#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the pending benches of `bench_checklist.json` on a shared cluster as
GPUs free up, one bench at a time.

Every `--interval-s` the watcher reads the cluster's nodes and running pods
and counts the free GPUs on each schedulable node. A bench fits when:

  * multinode: one node has 8 free GPUs (the TP=8 decode) and `encoders`
    other nodes have at least one free GPU each;
  * singlenode: one node has 8 free GPUs;

and its model is fully on the PVC (every shard its safetensors index lists).
Among the benches that fit, the lowest `priority` runs; the checklist ranks
the multi-node benches ahead of the single-node ones. The watcher pins the
decode (or single) pod to the 8-GPU node and the encoders to the nodes it
counted, through `k8s_bench.py --gpu-node/--encoder-nodes`, so the plan it
checked is the plan that runs.

Nodes in the checklist's `exclude_nodes` (known-broken hardware) are never
counted or used.

Several watchers can share one checklist, so benches run in parallel on
different nodes: a bench is claimed under a file lock before it starts, and
the nodes of every running bench are left out of the next placement.

Nothing waits on the cluster: a bench that does not fit right now is simply
not launched, and no GPU is held while waiting. A failed bench is marked
failed with its log path and is not retried automatically.

The cluster is addressed through a private kubeconfig holding only the
checklist's context (`oc config view --minify --flatten`), so running the
watcher never changes the current context of any other session.

    python watcher.py                  # loop until every bench is done/failed
    python watcher.py --once --dry-run # print the decision for the cluster now
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKLIST = HERE / "bench_checklist.json"
RESULTS = HERE / "results" / "pokprod"
GPUS_PER_DECODE = 8
# Pod slots a node must have free to host a bench server: its own pod, with
# room left for the bench's GPU-less proxy or client landing there too.
MIN_POD_SLOTS = 3
# Statuses a bench waits under. `pending-tuned` marks benches with per-bench
# `arms`/`extra_args`, which only this version of the watcher applies.
QUEUED = ("pending", "pending-tuned")
# Models already seen fully on the PVC; a downloaded model stays downloaded.
_ON_PVC: set[str] = set()


def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[watcher {stamp}] {msg}", flush=True)


@contextlib.contextmanager
def checklist_lock():
    """Exclusive access to the checklist across watcher processes."""
    with open(CHECKLIST.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def claim(bench: dict) -> bool:
    """Mark `bench` running in the checklist file if it is still pending and
    none of its nodes went to another running bench meanwhile.

    Several watchers may share the checklist; the lock makes the
    read-check-write atomic, so two of them never start the same bench or
    place two benches on one node.
    """
    with checklist_lock():
        current = load_checklist()
        entry = next(b for b in current["benches"] if b["id"] == bench["id"])
        if entry["status"] != bench.get("_queue", "pending"):
            return False
        if {bench["gpu_node"], *bench["encoder_nodes"]} & busy_nodes(current):
            return False
        entry.update({k: v for k, v in bench.items() if k != "_queue"})
        save_checklist(current)
        return True


def busy_nodes(checklist: dict) -> set[str]:
    """Every node a running bench was placed on."""
    return {
        n
        for b in checklist["benches"]
        if b["status"] == "running"
        for n in [b.get("gpu_node"), *b.get("encoder_nodes", [])]
        if n
    }


def load_checklist() -> dict:
    return json.loads(CHECKLIST.read_text())


def save_checklist(checklist: dict) -> None:
    tmp = CHECKLIST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(checklist, indent=2) + "\n")
    tmp.replace(CHECKLIST)


def private_kubeconfig(context: str) -> str:
    """A kubeconfig holding only `context`, so `oc` here never touches the
    shared current-context."""
    out = subprocess.run(
        ["oc", "config", "view", "--minify", "--flatten", "--context", context],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    fd, path = tempfile.mkstemp(prefix="watcher-kubeconfig-", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(out)
    os.chmod(path, 0o600)
    return path


def oc_json(kubeconfig: str, *argv: str, timeout: int = 180) -> dict:
    out = subprocess.run(
        ["oc", "--kubeconfig", kubeconfig, *argv, "-o", "json"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    ).stdout
    return json.loads(out)


def _gpus(pod: dict) -> int:
    return sum(
        int((c.get("resources", {}).get("requests") or {}).get("nvidia.com/gpu", 0))
        for c in pod["spec"]["containers"]
    )


def free_gpus(kubeconfig: str, exclude: set[str]) -> dict[str, int]:
    """Free GPUs per schedulable, Ready GPU node outside `exclude`.

    A node at its pod limit takes no new pod however many GPUs it has free,
    so one with fewer than `MIN_POD_SLOTS` free pod slots counts as having no
    free GPUs.
    """
    nodes = oc_json(kubeconfig, "get", "nodes")["items"]
    pods = oc_json(
        kubeconfig,
        "get",
        "pods",
        "-A",
        "--field-selector=status.phase!=Succeeded,status.phase!=Failed",
    )["items"]
    used: dict[str, int] = {}
    pod_count: dict[str, int] = {}
    for pod in pods:
        node = pod["spec"].get("nodeName")
        if node:
            used[node] = used.get(node, 0) + _gpus(pod)
            pod_count[node] = pod_count.get(node, 0) + 1
    free: dict[str, int] = {}
    for node in nodes:
        total = int(node["status"]["allocatable"].get("nvidia.com/gpu", 0))
        if (
            not total
            or node["spec"].get("unschedulable")
            or node["metadata"]["name"] in exclude
        ):
            continue
        ready = any(
            c["type"] == "Ready" and c["status"] == "True"
            for c in node["status"].get("conditions", [])
        )
        if not ready:
            continue
        name = node["metadata"]["name"]
        slots = int(node["status"]["allocatable"].get("pods", 0)) - pod_count.get(
            name, 0
        )
        free[name] = total - used.get(name, 0) if slots >= MIN_POD_SLOTS else 0
    return free


def models_on_pvc(kubeconfig: str, namespace: str, repos: set[str]) -> set[str]:
    """Which `repos` are fully downloaded on the PVC: every shard their
    `model.safetensors.index.json` lists is present. Read-only, through a
    throwaway CPU-only pod. A model found once is remembered: the probe pod
    takes seconds to schedule, and a free node may not wait that long."""
    missing = repos - _ON_PVC
    if not missing:
        return set(repos)
    checks = []
    for repo in sorted(missing):
        snap = f"/bench/hf/hub/models--{repo.replace('/', '--')}/snapshots/*"
        checks.append(
            f"for s in {snap}; do i=$s/model.safetensors.index.json; "
            f'[ -f "$i" ] || continue; '
            f'want=$(grep -o \'"[^"]*\\.safetensors"\' "$i" | sort -u | wc -l); '
            f'have=$(ls "$s"/*.safetensors 2>/dev/null | wc -l); '
            f'[ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && '
            f"echo {shlex.quote(repo)} && break; done"
        )
    # Per process: parallel watchers would otherwise delete each other's.
    name = f"ec-watcher-probe-{os.getpid()}"
    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name, "namespace": namespace, "labels": {"app": name}},
        "spec": {
            "restartPolicy": "Never",
            "serviceAccountName": "ec-bench",
            "containers": [
                {
                    "name": "probe",
                    "image": "busybox:1.36",
                    "command": ["sh", "-c", "; ".join(checks) + "; true"],
                    "resources": {"requests": {"cpu": "10m", "memory": "16Mi"}},
                    "volumeMounts": [
                        {"name": "bench", "mountPath": "/bench", "readOnly": True}
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "bench",
                    "persistentVolumeClaim": {
                        "claimName": "ec-bench-data",
                        "readOnly": True,
                    },
                }
            ],
        },
    }
    oc = ["oc", "--kubeconfig", kubeconfig, "-n", namespace]
    subprocess.run(
        [*oc, "delete", "pod", name, "--ignore-not-found", "--wait=true"],
        capture_output=True,
        timeout=120,
    )
    subprocess.run(
        [*oc, "apply", "-f", "-"],
        input=json.dumps(manifest),
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    try:
        subprocess.run(
            [
                *oc,
                "wait",
                f"pod/{name}",
                "--for=jsonpath={.status.phase}=Succeeded",
                "--timeout=180s",
            ],
            capture_output=True,
            timeout=240,
        )
        out = subprocess.run(
            [*oc, "logs", name], capture_output=True, text=True, timeout=60
        ).stdout
    finally:
        subprocess.run(
            [*oc, "delete", "pod", name, "--ignore-not-found", "--wait=false"],
            capture_output=True,
            timeout=60,
        )
    _ON_PVC.update(line.strip() for line in out.splitlines() if line.strip() in repos)
    return repos & _ON_PVC


def sweep_run(kubeconfig: str, namespace: str, run_id: str) -> None:
    """Remove whatever one bench run left behind.

    Each run tears itself down, but a run killed mid-way leaves its
    Deployments (and their GPUs) behind. Only resources labeled with this
    run's own id are touched, so benches running in parallel are left alone.
    """
    from k8s_bench import APP_LABEL

    subprocess.run(
        [
            "oc",
            "--kubeconfig",
            kubeconfig,
            "-n",
            namespace,
            "delete",
            "deployment,service,pod,configmap",
            "-l",
            f"app={APP_LABEL},run-id={run_id}",
            "--ignore-not-found",
            "--wait=true",
        ],
        capture_output=True,
        timeout=600,
    )


def plan(bench: dict, free: dict[str, int]) -> tuple[str, list[str]] | None:
    """(gpu_node, encoder_nodes) for `bench` on `free`, or None if it does not
    fit. Takes the fullest other nodes for encoders, so a later bench keeps
    the most room."""
    gpu_nodes = sorted(n for n, f in free.items() if f >= GPUS_PER_DECODE)
    if not gpu_nodes:
        return None
    gpu_node = gpu_nodes[0]
    if bench["kind"] == "singlenode":
        return gpu_node, []
    others = sorted(
        (n for n, f in free.items() if n != gpu_node and f >= 1),
        key=lambda n: (-free[n], n),
    )
    if len(others) < bench["encoders"]:
        return None
    return gpu_node, others[: bench["encoders"]]


def bench_argv(
    checklist: dict,
    bench: dict,
    gpu_node: str,
    encoders: list[str],
    run_id: str,
    out_dir: Path,
) -> list[str]:
    d = checklist["defaults"]
    model = checklist["models"][bench["model"]]
    wl = checklist["workloads"][bench["workload"]]
    argv = [
        sys.executable,
        str(HERE / "k8s_bench.py"),
        "--namespace",
        d["namespace"],
        "--image",
        d["image"],
        "--model",
        model["model"],
        "--run-id",
        run_id,
        "--out-dir",
        str(out_dir),
        "--workload-dir",
        wl["dir"].format(model=bench["model"]),
        *wl["build"],
        "--request-rates",
        d["request_rates"],
        "--gpu-node",
        gpu_node,
        "--exclude-nodes",
        ",".join(d["exclude_nodes"]),
        *d["common_args"],
        *model["extra_args"],
        *bench.get("extra_args", []),
    ]
    if bench["kind"] == "multinode":
        argv += [
            "--arms",
            bench.get("arms", d["multinode_arms"]),
            "--num-encoders",
            str(bench["encoders"]),
            "--encoder-gpus",
            "1",
            "--decode-gpus",
            str(GPUS_PER_DECODE),
            "--encoder-nodes",
            ",".join(encoders),
        ]
    else:
        argv += [
            "--arms",
            bench.get("arms", d["singlenode_arms"]),
            "--decode-gpus",
            str(GPUS_PER_DECODE),
        ]
    return argv


def run_bench(
    checklist: dict,
    bench: dict,
    gpu_node: str,
    encoders: list[str],
    kubeconfig: str,
    dry_run: bool,
) -> bool | None:
    """Run `bench` to completion: True if it passed, False if it failed, None
    if another watcher took it or its nodes first (nothing ran)."""
    run_id = f"w{datetime.datetime.now():%m%d%H%M%S}-{bench['encoders']}e"
    out_dir = RESULTS / bench["model"] / bench["id"]
    argv = bench_argv(checklist, bench, gpu_node, encoders, run_id, out_dir)
    placement = f"on decode={gpu_node} encoders={encoders or '-'}"
    if dry_run:
        log(f"would launch {bench['id']} {placement}")
        log("  " + shlex.join(argv))
        return True
    log_path = out_dir / f"{run_id}.log"
    bench.update(
        status="running",
        run_id=run_id,
        gpu_node=gpu_node,
        encoder_nodes=encoders,
        log=str(log_path),
        started=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    if not claim(bench):
        log(
            f"{bench['id']} not launched: it is no longer queued, or its nodes "
            "went to another watcher; trying the next"
        )
        return None
    log(f"launching {bench['id']} {placement}")
    log("  " + shlex.join(argv))
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, KUBECONFIG=kubeconfig)
    with open(log_path, "w") as f:
        rc = subprocess.run(
            argv, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT
        ).returncode
    # The checklist may have been edited while the bench ran, by the user or
    # another watcher; merge only this bench's own fields, under the lock.
    with checklist_lock():
        current = load_checklist()
        entry = next(b for b in current["benches"] if b["id"] == bench["id"])
        entry.update(
            {
                k: bench[k]
                for k in ("run_id", "gpu_node", "encoder_nodes", "log", "started")
            }
        )
        entry["status"] = "done" if rc == 0 else "failed"
        entry["exit_code"] = rc
        entry["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        save_checklist(current)
    if rc != 0:
        sweep_run(kubeconfig, checklist["defaults"]["namespace"], run_id)
    log(f"{bench['id']} {'done' if rc == 0 else f'FAILED (exit {rc}), see {log_path}'}")
    return rc == 0


def tick(kubeconfig: str, dry_run: bool) -> bool:
    """One scheduling pass. Returns False once nothing is left to run."""
    checklist = load_checklist()
    pending = sorted(
        (b for b in checklist["benches"] if b["status"] in QUEUED),
        key=lambda b: (b["priority"], b["id"]),
    )
    for b in pending:
        b["_queue"] = b["status"]
    if not pending:
        log("checklist complete")
        return False
    busy = busy_nodes(checklist)
    free = free_gpus(kubeconfig, set(checklist["defaults"]["exclude_nodes"]) | busy)
    gpu_nodes = [n for n, f in free.items() if f >= GPUS_PER_DECODE]
    log(
        f"{len(pending)} pending; 8-free nodes {gpu_nodes or 'none'}; "
        f"nodes with >=1 free {sum(1 for f in free.values() if f >= 1)}"
    )
    if not gpu_nodes:
        return True
    repos = {checklist["models"][b["model"]]["model"] for b in pending}
    have = models_on_pvc(kubeconfig, checklist["defaults"]["namespace"], repos)
    for bench in pending:
        if checklist["models"][bench["model"]]["model"] not in have:
            continue
        placed = plan(bench, free)
        if not placed:
            continue
        if run_bench(checklist, bench, *placed, kubeconfig, dry_run) is not None:
            return True
        # Lost to another watcher: its nodes are taken now, so place the
        # next bench on what is left.
        free = {n: f for n, f in free.items() if n not in busy_nodes(load_checklist())}
    log(
        "an 8-GPU node is free, but no pending bench fits (model not on the PVC "
        "yet, or too few other nodes for its encoders)"
    )
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--interval-s", type=int, default=60)
    p.add_argument("--once", action="store_true", help="one pass, then exit")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the launch decision without running anything",
    )
    args = p.parse_args()
    kubeconfig = private_kubeconfig(load_checklist()["defaults"]["context"])
    try:
        while True:
            try:
                more = tick(kubeconfig, args.dry_run)
            except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as exc:
                log(
                    f"cluster read failed ({exc.__class__.__name__}: "
                    f"{str(exc)[:200]}); retrying next interval"
                )
                more = True
            if args.once or not more:
                return 0
            time.sleep(args.interval_s)
    finally:
        os.unlink(kubeconfig)


if __name__ == "__main__":
    sys.exit(main())
