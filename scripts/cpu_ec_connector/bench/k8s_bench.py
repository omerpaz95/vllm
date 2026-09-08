#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-node EC connector benchmark on OpenShift.

The arms, server flags, load driving, log accounting and gates are
`run_bench.py`'s; this driver only changes where the servers run. Each server
becomes a Deployment and a Service, with the encoder(s) and the decode
instance kept on different nodes by pod anti-affinity, and the load generator
becomes a GPU-less pod on the same image. Everything runs from a laptop with
`oc`:

    ECCPUConnector over NIXL   cpu-data, cpu-grid    pod-to-pod side channel
    ECExampleConnector         example-data, -grid   ReadWriteMany PVC
    baseline, offload          one pod               no peer

One PVC, mounted at the same path in every pod, carries the HF cache (so the
model is downloaded once), the workload, each run's scratch and the example
connector's storage. Server logs stay in each pod: the container command
tees its output to a file that `oc exec` reads, so the byte-offset log
accounting `run_bench` does works unchanged.

Typical use:

    python k8s_bench.py --namespace my-ns --dry-run \\
        --arms baseline,cpu-grid,example-grid
    python k8s_bench.py --namespace my-ns --arms baseline,cpu-grid \\
        --gen-workload "--pool-size 96 --buckets 2048x2048:1.0 --num-requests 400"

`bench.json` has `run_bench`'s schema plus a `placement` block per arm that
records the node each pod landed on.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import run_bench
import yaml
from run_bench import (
    ARMS,
    BENCH_DIR,
    PROXY,
    BenchServer,
    ServerMismatchError,
    System,
    Target,
)

APP_LABEL = "vllm-ec-bench"
GPU_TIER_LABEL = "gpu"
PVC_MOUNT = "/bench"
SCRIPTS_MOUNT = "/bench-scripts"
POD_LOG = "/tmp/server.log"
K8S_DIR = BENCH_DIR / "k8s"
PROXY_IN_IMAGE = f"/vllm-workspace/{PROXY}"
GIB = 1024**3
_DRY_RUN_EC_CPU_BYTES = 8 * GIB
_SHM_HEADROOM_BYTES = 8 * GIB
_POD_MEMORY_HEADROOM_BYTES = 32 * GIB
_CONFIGMAP_LIMIT_BYTES = 1024 * 1024
# What runs inside the pods; the drivers stay on the laptop.
_POD_SCRIPTS = ("gen_workload.py", "ec_log_stats.py", "run_bench.py")
_HEALTH_POLL_S = 5.0
_STOP_TIMEOUT_S = 300
_RUN_ID_RE = re.compile(r"^[a-z0-9]([-a-z0-9]{0,28}[a-z0-9])?$")
_SIZE_RE = re.compile(r"^(\d+)(Gi|Mi)$")


def parse_size(spec: str) -> int:
    m = _SIZE_RE.match(spec)
    if not m:
        raise SystemExit(f"[k8s] size {spec!r} must look like 64Gi or 512Mi")
    return int(m.group(1)) * (GIB if m.group(2) == "Gi" else 1024**2)


def gib(num_bytes: int) -> str:
    return f"{-(-num_bytes // GIB)}Gi"


def load_template(name: str) -> list[dict[str, Any]]:
    with (K8S_DIR / name).open() as f:
        return list(yaml.safe_load_all(f))


def labels(args: argparse.Namespace, role: str) -> dict[str, str]:
    return {"app": APP_LABEL, "run-id": args.run_id, "role": role}


def selector(args: argparse.Namespace, role: str = "") -> str:
    parts = [f"app={APP_LABEL}", f"run-id={args.run_id}"]
    if role:
        parts.append(f"role={role}")
    return ",".join(parts)


def env_var(name: str, value: str) -> dict[str, Any]:
    return {"name": name, "value": value}


def env_pairs(pairs: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for pair in pairs:
        name, _, value = pair.partition("=")
        out.append(env_var(name, value))
    return out


# --------------------------------------------------------------------------
# oc
# --------------------------------------------------------------------------


class Oc:
    """The few oc verbs this driver needs, namespace-scoped."""

    def __init__(self, namespace: str) -> None:
        if not shutil.which("oc"):
            raise SystemExit(
                "[k8s] `oc` is not on PATH; only --dry-run works without it"
            )
        self.namespace = namespace

    def run(
        self,
        *argv: str,
        stdin: str | None = None,
        check: bool = True,
        timeout: int = 600,
    ) -> subprocess.CompletedProcess[str]:
        cmd = ["oc", "-n", self.namespace, *argv]
        result = subprocess.run(
            cmd, input=stdin, capture_output=True, text=True, timeout=timeout
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"oc failed ({result.returncode}): {' '.join(cmd)}\n"
                f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
            )
        return result

    def apply(self, docs: list[dict[str, Any]]) -> None:
        self.run("apply", "-f", "-", stdin=yaml.safe_dump_all(docs, sort_keys=False))

    def delete(self, kinds: str, sel: str, *, timeout_s: int = _STOP_TIMEOUT_S) -> None:
        self.run(
            "delete",
            kinds,
            "-l",
            sel,
            "--ignore-not-found",
            "--wait=true",
            f"--timeout={timeout_s}s",
            check=False,
            timeout=timeout_s + 60,
        )

    def pods(self, sel: str) -> list[dict[str, Any]]:
        out = self.run("get", "pod", "-l", sel, "-o", "json").stdout
        return json.loads(out).get("items", []) if out.strip() else []

    def exists(self, kind: str, name: str) -> bool:
        return self.run("get", kind, name, check=False).returncode == 0

    def rollout_status(self, deployment: str, timeout_s: int) -> None:
        result = self.run(
            "rollout",
            "status",
            f"deployment/{deployment}",
            f"--timeout={timeout_s}s",
            check=False,
            timeout=timeout_s + 60,
        )
        if result.returncode != 0:
            wide = self.run("get", "pod", "-l", f"app={APP_LABEL}", "-o", "wide")
            events = self.run(
                "get", "events", "--sort-by=.lastTimestamp", check=False
            ).stdout.splitlines()[-15:]
            raise RuntimeError(
                f"{deployment} did not roll out within {timeout_s}s: "
                f"{result.stderr.strip()[-500:]}\n{wide.stdout}\n" + "\n".join(events)
            )

    def wait_pod_ready(self, pod: str, timeout_s: int) -> None:
        self.run(
            "wait",
            f"pod/{pod}",
            "--for=condition=Ready",
            f"--timeout={timeout_s}s",
            timeout=timeout_s + 60,
        )

    def wait_gone(self, sel: str, timeout_s: int = _STOP_TIMEOUT_S) -> None:
        deadline = time.monotonic() + timeout_s
        while self.pods(sel):
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"pods matching {sel} still exist after {timeout_s}s"
                )
            time.sleep(2.0)


class PodTarget(Target):
    """`oc exec` into a pod or a deployment's pod, in one namespace."""

    def __init__(self, namespace: str, ref: str) -> None:
        super().__init__(ref)
        self.namespace = namespace

    def _argv(self, script: str) -> list[str]:
        assert self.pod is not None
        argv = ["bash", "-lc", script]
        return ["oc", "exec", "-n", self.namespace, self.pod, "--", *argv]


def pod_status(pod: dict[str, Any]) -> dict[str, Any]:
    status = pod.get("status", {})
    containers = status.get("containerStatuses") or [{}]
    return {
        "pod": pod["metadata"]["name"],
        "node": pod["spec"].get("nodeName", ""),
        "pod_ip": status.get("podIP", ""),
        "phase": status.get("phase", ""),
        "restarts": containers[0].get("restartCount", 0),
        "image_id": containers[0].get("imageID", ""),
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def node_affinity(args: argparse.Namespace) -> dict[str, Any]:
    if not args.exclude_nodes:
        return {}
    return {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchExpressions": [
                            {
                                "key": "kubernetes.io/hostname",
                                "operator": "NotIn",
                                "values": args.exclude_nodes,
                            }
                        ]
                    }
                ]
            }
        }
    }


def gpu_anti_affinity(args: argparse.Namespace) -> dict[str, Any]:
    """Every GPU pod of this run on a different node than every other."""
    return {
        "podAntiAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": [
                {
                    "labelSelector": {
                        "matchLabels": {
                            "app": APP_LABEL,
                            "run-id": args.run_id,
                            "tier": GPU_TIER_LABEL,
                        }
                    },
                    "topologyKey": "kubernetes.io/hostname",
                }
            ]
        }
    }


def shm_bytes(args: argparse.Namespace, server: BenchServer) -> int:
    """The EC region is an mmap under /dev/shm, and torch_shm and NIXL use it
    too; a connector-less server still needs room for the worker IPC."""
    if args.shm_size:
        return parse_size(args.shm_size)
    region = args.ec_cpu_bytes if server.ec_config else 0
    return int(region * 1.25) + _SHM_HEADROOM_BYTES


def common_env(args: argparse.Namespace) -> list[dict[str, Any]]:
    env = [
        # First, so `$(POD_IP)` in a later value expands.
        {"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}},
        env_var("VLLM_USE_V2_MODEL_RUNNER", "1"),
        env_var("VLLM_LOGGING_LEVEL", "DEBUG"),
        env_var("VLLM_SERVER_DEV_MODE", "1"),
        env_var("HF_HOME", args.hf_home),
        {
            "name": "HF_TOKEN",
            "valueFrom": {
                "secretKeyRef": {"name": args.hf_secret, "key": args.hf_secret_key}
            },
        },
    ]
    if args.ucx_tls:
        env.append(env_var("UCX_TLS", args.ucx_tls))
    env.extend(env_pairs(tuple(args.pod_env)))
    return env


class PodServer(BenchServer):
    """A `BenchServer` whose process is the container of a Deployment.

    The container runs the command `run_bench` builds, teeing its output to
    `POD_LOG`, so `verify()`, `reset_caches()` and the log accounting reach
    it through `oc exec` unchanged. `start()` applies the rendered manifests
    and `stop()` deletes them by label. The producer's side channel binds the
    pod IP (`VLLM_EC_SIDE_CHANNEL_HOST=$(POD_IP)`, from the downward API): a
    ZMQ ROUTER has to bind an address the pod owns, which a Service name is
    not, and the consumer dials whatever the producer announced.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.log_path = POD_LOG
        self.target = PodTarget(self.args.namespace, f"deployment/{self.resource}")
        self.oc: Oc | None = None
        self.arm = ""
        self.placement: dict[str, Any] = {}
        self.starts = 0

    @property
    def resource(self) -> str:
        return f"ec-bench-{self.args.run_id}-{self.name}"

    @property
    def base_url(self) -> str:
        return f"http://{self.resource}.{self.args.namespace}.svc:{self.port}"

    @property
    def selector(self) -> str:
        return selector(self.args, self.name)

    @property
    def side_channel_port(self) -> int:
        for pair in self.extra_env:
            name, _, value = pair.partition("=")
            if name == "VLLM_EC_SIDE_CHANNEL_PORT":
                return int(value)
        return 0

    def launch_script(self, *, instrument: bool = False) -> str:
        """The container command. `exec` keeps the server as PID 1 so the
        pod's SIGTERM reaches it; the process substitution keeps the log in
        a file the byte-offset reads can seek."""
        serve = [self.command] if self.command else self._serve_args()
        return f"exec {' '.join(serve)} > >(tee {POD_LOG}) 2>&1"

    def _container(self) -> dict[str, Any]:
        ports = [{"name": "http", "containerPort": self.port}]
        if self.side_channel_port:
            ports.append(
                {"name": "side-channel", "containerPort": self.side_channel_port}
            )
        env = common_env(self.args) + env_pairs(self.extra_env)
        mounts = [{"name": "bench", "mountPath": PVC_MOUNT}]
        if self.is_vllm:
            shm = shm_bytes(self.args, self)
            memory = self.args.pod_memory or gib(shm + _POD_MEMORY_HEADROOM_BYTES)
            resources = {
                "requests": {
                    "cpu": str(self.args.pod_cpus),
                    "memory": memory,
                    "nvidia.com/gpu": str(self.tp),
                },
                "limits": {"nvidia.com/gpu": str(self.tp)},
            }
            mounts.append({"name": "dshm", "mountPath": "/dev/shm"})
        else:
            resources = {"requests": {"cpu": "4", "memory": "8Gi"}}
            mounts.append({"name": "scripts", "mountPath": SCRIPTS_MOUNT})
        return {
            "image": self.args.image,
            "command": ["bash", "-c", self.launch_script()],
            "ports": ports,
            "env": env,
            "resources": resources,
            "volumeMounts": mounts,
        }

    def _volumes(self) -> list[dict[str, Any]]:
        volumes: list[dict[str, Any]] = [pvc_volume(self.args)]
        if self.is_vllm:
            volumes.append(
                {
                    "name": "dshm",
                    "emptyDir": {
                        "medium": "Memory",
                        "sizeLimit": gib(shm_bytes(self.args, self)),
                    },
                }
            )
        else:
            volumes.append(scripts_volume(self.args))
        return volumes

    def render(self) -> list[dict[str, Any]]:
        docs = load_template("vllm-deployment.yaml")
        by_kind = {d["kind"]: d for d in docs}
        dep, svc = by_kind["Deployment"], by_kind["Service"]
        pod_labels = labels(self.args, self.name)
        if self.is_vllm:
            pod_labels["tier"] = GPU_TIER_LABEL
        for doc in docs:
            doc["metadata"]["name"] = self.resource
            doc["metadata"]["namespace"] = self.args.namespace
            doc["metadata"]["labels"] = labels(self.args, self.name)
        dep["spec"]["selector"]["matchLabels"] = labels(self.args, self.name)
        template = dep["spec"]["template"]
        template["metadata"]["labels"] = pod_labels
        spec = template["spec"]
        spec["containers"][0].update(self._container())
        spec["volumes"] = self._volumes()
        affinity = node_affinity(self.args)
        if self.is_vllm:
            if self.args.node_selector:
                spec["nodeSelector"] = dict(self.args.node_selector)
            if not self.args.same_node:
                affinity.update(gpu_anti_affinity(self.args))
        if affinity:
            spec["affinity"] = affinity
        if not self.args.run_as_root:
            spec.pop("securityContext", None)
        svc["spec"]["selector"] = labels(self.args, self.name)
        svc["spec"]["ports"] = [
            {"name": "http", "port": self.port, "targetPort": self.port}
        ]
        return docs

    def _oc(self) -> Oc:
        if self.oc is None:
            self.oc = Oc(self.args.namespace)
        return self.oc

    def start(self, *, instrument: bool = False) -> None:
        if instrument:
            raise SystemExit("[k8s] --frag instrumentation is not supported in pods")
        self.stop()
        self.starts += 1
        print(f"[bench] applying {self.resource}")
        self._oc().apply(self.render())
        self._oc().rollout_status(self.resource, self.args.startup_timeout_s)
        self.placement = self.status()
        print(
            f"[bench] {self.resource}: pod {self.placement['pod']} on node "
            f"{self.placement['node']} ({self.placement['pod_ip']})"
        )

    def status(self) -> dict[str, Any]:
        pods = self._oc().pods(self.selector)
        if not pods:
            return {}
        return pod_status(pods[0])

    def save_log(self) -> None:
        """Keep the pod's log locally: the pod, and with it the log, is about
        to be deleted, and a failed gate is diagnosed from it."""
        log_dir = self.args.out_dir / "logs" / self.args.run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{self.arm or 'arm'}-{self.name}-{self.starts}.log"
        result = self._oc().run(
            "logs", f"deployment/{self.resource}", check=False, timeout=600
        )
        path.write_text(result.stdout)

    def stop(self) -> None:
        oc = self._oc()
        if oc.pods(self.selector):
            final = self.status()
            if final:
                self.placement = {**self.placement, **final}
            self.save_log()
        oc.delete("deployment,service", self.selector)
        oc.wait_gone(self.selector)

    def wait_healthy(self) -> None:
        deadline = time.monotonic() + self.args.startup_timeout_s
        probe = (
            f"curl -s -o /dev/null -w '%{{http_code}}' "
            f"http://127.0.0.1:{self.port}/health || true"
        )
        while time.monotonic() < deadline:
            if self.target.sh(probe, check=False).stdout.strip() == "200":
                return
            status = self.status()
            if status.get("restarts", 0) or status.get("phase") == "Failed":
                self.save_log()
                raise RuntimeError(
                    f"{self.resource} restarted or failed before becoming healthy "
                    f"({status}); see {self.args.out_dir}/logs/{self.args.run_id}"
                )
            time.sleep(_HEALTH_POLL_S)
        raise RuntimeError(
            f"{self.resource} not healthy within {self.args.startup_timeout_s}s; "
            f"see `oc logs -n {self.args.namespace} deployment/{self.resource}`"
        )


class ClientPod:
    """The GPU-less pod that generates the workload and drives the load."""

    def __init__(self, args: argparse.Namespace, oc: Oc | None) -> None:
        self.args = args
        self.oc = oc
        self.resource = f"ec-bench-{args.run_id}-client"
        self.target = PodTarget(args.namespace, f"pod/{self.resource}")

    @property
    def selector(self) -> str:
        return selector(self.args, "client")

    def render(self) -> list[dict[str, Any]]:
        (pod,) = load_template("bench-client-pod.yaml")
        pod["metadata"].update(
            {
                "name": self.resource,
                "namespace": self.args.namespace,
                "labels": labels(self.args, "client"),
            }
        )
        spec = pod["spec"]
        container = spec["containers"][0]
        container["image"] = self.args.image
        container["env"] = common_env(self.args)
        container["volumeMounts"] = [
            {"name": "bench", "mountPath": PVC_MOUNT},
            {"name": "scripts", "mountPath": SCRIPTS_MOUNT},
        ]
        spec["volumes"] = [pvc_volume(self.args), scripts_volume(self.args)]
        affinity = node_affinity(self.args)
        if affinity:
            spec["affinity"] = affinity
        if not self.args.run_as_root:
            spec.pop("securityContext", None)
        return [pod]

    def start(self) -> None:
        assert self.oc is not None
        self.stop()
        print(f"[k8s] applying {self.resource}")
        self.oc.apply(self.render())
        self.oc.wait_pod_ready(self.resource, self.args.startup_timeout_s)
        self.target.sh(f"mkdir -p {self.args.work_dir} {self.args.hf_home}")

    def stop(self) -> None:
        assert self.oc is not None
        self.oc.delete("pod", self.selector)
        self.oc.wait_gone(self.selector)


def scripts_configmap_name(args: argparse.Namespace) -> str:
    return f"ec-bench-{args.run_id}-scripts"


def pvc_volume(args: argparse.Namespace) -> dict[str, Any]:
    return {"name": "bench", "persistentVolumeClaim": {"claimName": args.pvc_name}}


def scripts_volume(args: argparse.Namespace) -> dict[str, Any]:
    return {"name": "scripts", "configMap": {"name": scripts_configmap_name(args)}}


def render_scripts_configmap(args: argparse.Namespace) -> dict[str, Any]:
    """The bench scripts and this checkout's EPD proxy, for the pods.

    The proxy travels with the bench rather than coming from the image: the
    `--no-rewrite` semantics, the `Rewrote N` line and the `STAGE` timings
    this harness reads are properties of the checked-out proxy, and an image
    built from another commit may log differently.
    """
    files = {name: (BENCH_DIR / name).read_text() for name in _POD_SCRIPTS}
    files["disagg_epd_proxy.py"] = (BENCH_DIR.parents[2] / PROXY).read_text()
    total = sum(len(v.encode()) for v in files.values())
    if total > _CONFIGMAP_LIMIT_BYTES:
        raise SystemExit(
            f"[k8s] the bench scripts total {total} bytes, over the ConfigMap "
            f"limit of {_CONFIGMAP_LIMIT_BYTES}"
        )
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": scripts_configmap_name(args),
            "namespace": args.namespace,
            "labels": labels(args, "scripts"),
        },
        "data": files,
    }


def render_pvc(args: argparse.Namespace) -> dict[str, Any]:
    (pvc,) = load_template("pvc.yaml")
    pvc["metadata"].update(
        {
            "name": args.pvc_name,
            "namespace": args.namespace,
            # No run-id: the claim outlives runs, so `--cleanup` skips it.
            "labels": {"app": APP_LABEL, "role": "pvc"},
        }
    )
    pvc["spec"]["storageClassName"] = args.storage_class
    pvc["spec"]["resources"]["requests"]["storage"] = args.pvc_size
    return pvc


def build_pod_system(target: Target, args: argparse.Namespace, name: str) -> System:
    sys_ = run_bench.build_system(target, args, name, server_cls=PodServer)
    for server in sys_.servers:
        assert isinstance(server, PodServer)
        server.arm = name
    return sys_


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------


def placement_record(name: str, sys_: System, args: argparse.Namespace) -> dict:
    pods = []
    for server in sys_.servers:
        assert isinstance(server, PodServer)
        pods.append({"role": server.name, **server.placement})
    gpu_nodes = [
        p["node"] for p, s in zip(pods, sys_.servers) if s.is_vllm and p.get("node")
    ]
    return {
        "arm": name,
        "same_node_requested": args.same_node,
        "gpu_nodes_distinct": len(set(gpu_nodes)) == len(gpu_nodes),
        "pods": pods,
    }


def check_placement(record: dict[str, Any]) -> None:
    """The anti-affinity is `required`, so the scheduler already enforced
    this; the check turns the proof into a gate and catches restarts."""
    name = record["arm"]
    restarted = [p["role"] for p in record["pods"] if p.get("restarts")]
    if restarted:
        raise ServerMismatchError(
            f"{name}: {restarted} restarted during the arm, so part of the "
            "measurement ran against a fresh process"
        )
    if not record["same_node_requested"] and not record["gpu_nodes_distinct"]:
        nodes = {p["role"]: p.get("node") for p in record["pods"]}
        raise ServerMismatchError(f"{name}: GPU pods shared a node: {nodes}")


def print_placement(placement: dict[str, list[dict[str, Any]]]) -> None:
    print("\n[k8s] placement")
    for name, records in placement.items():
        for record in records:
            where = ", ".join(
                f"{p['role']}@{p.get('node') or '?'}" for p in record["pods"]
            )
            verdict = (
                "same node requested"
                if record["same_node_requested"]
                else ("distinct" if record["gpu_nodes_distinct"] else "SHARED")
            )
            print(f"[k8s] {name}: {where} ({verdict})")


# --------------------------------------------------------------------------
# Workload and preflight
# --------------------------------------------------------------------------


def ensure_workload(client: ClientPod, args: argparse.Namespace) -> None:
    manifest = f"{args.workload_dir}/manifest.json"
    if client.target.exists(manifest):
        return
    if not args.gen_workload:
        raise SystemExit(
            f"[k8s] no manifest.json in {args.workload_dir} on the PVC; pass "
            '--gen-workload "<gen_workload.py args>" to build it in-cluster'
        )
    print(f"[k8s] generating the workload in {args.workload_dir}")
    client.target.sh(
        f"mkdir -p {args.workload_dir} && {args.python} "
        f"{SCRIPTS_MOUNT}/gen_workload.py --out-dir {args.workload_dir} "
        f"{args.gen_workload}",
        timeout=args.gen_timeout_s,
    )


def preflight(oc: Oc, client: ClientPod, args: argparse.Namespace) -> None:
    who = oc.run("whoami", check=False)
    if who.returncode != 0:
        raise SystemExit(f"[k8s] not logged in: {who.stderr.strip()[:200]}")
    print(f"[k8s] authenticated as {who.stdout.strip()} in {args.namespace}")
    if not oc.exists("secret", args.hf_secret):
        raise SystemExit(f"[k8s] secret {args.hf_secret} not found in {args.namespace}")
    stale = [
        p
        for p in oc.pods(f"app={APP_LABEL}")
        if p["metadata"].get("labels", {}).get("run-id") != args.run_id
    ]
    if stale:
        names = ", ".join(p["metadata"]["name"] for p in stale)
        print(
            f"[k8s] WARNING: leftover bench pods hold GPUs: {names}; sweep with "
            "--cleanup",
            file=sys.stderr,
        )
    probe = client.target.sh(f"{args.python} -c 'import vllm' && echo ok", check=False)
    if "ok" not in probe.stdout:
        raise SystemExit(
            f"[k8s] {args.python} cannot import vllm in {args.image} "
            f"(rc={probe.returncode}); set --python to the image's interpreter.\n"
            f"{probe.stderr[-500:]}"
        )
    if any(ARMS[a].connector == "ECCPUConnector" for a in args.arms):
        probe = client.target.sh(
            f"{args.python} -c 'import nixl' && echo ok", check=False
        )
        if "ok" not in probe.stdout:
            raise SystemExit(
                f"[k8s] `import nixl` fails in {args.image}; the CPU arms need "
                "an image built with INSTALL_KV_CONNECTORS=true (the release "
                "images are). --skip-preflight overrides.\n"
                f"{probe.stderr[-500:]}"
            )


def ensure_pvc(oc: Oc, args: argparse.Namespace) -> None:
    if oc.exists("pvc", args.pvc_name):
        print(f"[k8s] using existing PVC {args.pvc_name}")
        return
    print(f"[k8s] creating PVC {args.pvc_name} ({args.storage_class}, {args.pvc_size})")
    oc.apply([render_pvc(args)])


def cleanup(args: argparse.Namespace) -> int:
    oc = Oc(args.namespace)
    sel = selector(args) if args.cleanup_run_id else f"app={APP_LABEL}"
    print(f"[k8s] deleting deployment,service,pod,configmap with {sel}")
    oc.delete("deployment,service,pod,configmap", sel)
    if args.delete_pvc:
        oc.run("delete", "pvc", args.pvc_name, "--ignore-not-found")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    run_bench.add_common_options(p)
    k = p.add_argument_group("cluster")
    k.add_argument("--namespace", required=True)
    k.add_argument("--image", default="vllm/vllm-openai:nightly")
    k.add_argument(
        "--hf-secret", default="llm-d-hf-token", help="secret holding HF_TOKEN"
    )
    k.add_argument("--hf-secret-key", default="HF_TOKEN")
    k.add_argument("--num-encoders", type=int, default=1)
    k.add_argument(
        "--encoder-gpus",
        type=int,
        default=1,
        help="nvidia.com/gpu per encoder pod; also its tensor-parallel size",
    )
    k.add_argument(
        "--decode-gpus",
        type=int,
        default=1,
        help="nvidia.com/gpu for the decode and single-instance pods; also TP",
    )
    k.add_argument(
        "--encoder-gpu-memory-utilization",
        type=float,
        default=0.0,
        help="0 = --gpu-memory-utilization; each encoder pod owns its GPUs",
    )
    k.add_argument("--storage-class", default="ibm-spectrum-scale-fileset")
    k.add_argument("--pvc-size", default="500Gi")
    k.add_argument("--pvc-name", default="ec-bench-data", help="reused if present")
    k.add_argument(
        "--same-node",
        action="store_true",
        help="control run: no anti-affinity, pods may share a node",
    )
    k.add_argument(
        "--node-selector",
        default="",
        help="key=value[,key=value] for the GPU pods",
    )
    k.add_argument("--exclude-nodes", default="", help="comma-separated hostnames")
    k.add_argument(
        "--shm-size",
        default="",
        help="/dev/shm per GPU pod (e.g. 64Gi); default 1.25x the EC region + 8Gi",
    )
    k.add_argument(
        "--pod-memory",
        default="",
        help="memory request per GPU pod; default shm + 32Gi",
    )
    k.add_argument("--pod-cpus", type=int, default=8)
    k.add_argument(
        "--ucx-tls",
        default="tcp,sm",
        help="UCX_TLS for every pod; '' leaves it unset so UCX may pick RDMA",
    )
    k.add_argument(
        "--pod-env",
        action="append",
        default=[],
        help="extra NAME=VALUE for every pod (e.g. UCX_NET_DEVICES=mlx5_0:1)",
    )
    k.add_argument(
        "--proxy-from-image",
        action="store_true",
        help=f"run {PROXY_IN_IMAGE} instead of this checkout's proxy",
    )
    k.add_argument(
        "--run-as-root",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="runAsUser 0 (needs the anyuid SCC); off relies on fsGroup for the PVC",
    )
    k.add_argument("--skip-preflight", action="store_true")
    k.add_argument(
        "--workload-dir",
        default=f"{PVC_MOUNT}/wl",
        help="on the PVC; every pod sees the same path",
    )
    k.add_argument(
        "--gen-workload",
        default="",
        help="gen_workload.py arguments, to build --workload-dir in-cluster if "
        "it has no manifest",
    )
    k.add_argument("--gen-timeout-s", type=int, default=7200)
    k.add_argument("--keep", action="store_true", help="leave the client pod up")
    k.add_argument(
        "--cleanup",
        action="store_true",
        help="delete every bench object in the namespace (the PVC stays)",
    )
    k.add_argument("--cleanup-run-id", action="store_true", help="only --run-id")
    k.add_argument(
        "--python",
        default="",
        help="interpreter inside the image for the servers, the proxy and the "
        "load generator; default python3, which the vllm-openai image symlinks "
        "into /usr/bin",
    )
    k.add_argument("--delete-pvc", action="store_true")
    p.set_defaults(startup_timeout_s=1800)
    args = p.parse_args()

    if not _RUN_ID_RE.match(args.run_id):
        raise SystemExit(
            f"[k8s] --run-id {args.run_id!r} must be a DNS label: lowercase "
            "alphanumerics and '-', at most 30 characters"
        )
    parse_size(args.pvc_size)
    args.node_selector = dict(
        kv.split("=", 1) for kv in args.node_selector.split(",") if kv.strip()
    )
    args.exclude_nodes = [n.strip() for n in args.exclude_nodes.split(",") if n.strip()]
    for pair in args.pod_env:
        if "=" not in pair:
            raise SystemExit(f"[k8s] --pod-env {pair!r} is not NAME=VALUE")

    # What run_bench's server construction reads, pinned for pods. python3 is
    # symlinked into /usr/bin in the vllm-openai image, so it resolves even
    # when a shell profile has reset PATH and dropped /opt/venv/bin.
    args.python = args.python or "python3"
    args.work_dir = f"{PVC_MOUNT}/runs/{args.run_id}"
    args.hf_home = f"{PVC_MOUNT}/hf"
    args.proxy_script = f"{SCRIPTS_MOUNT}/disagg_epd_proxy.py"
    if args.proxy_from_image:
        args.proxy_script = PROXY_IN_IMAGE
    args.proxy_host = "0.0.0.0"
    args.side_channel_host = "$(POD_IP)"
    args.frag = False
    args.patch_dir = ""
    gpus = ",".join(str(i) for i in range(args.decode_gpus))
    args.gpu = args.decode_gpu = gpus
    args.encoder_devices = ";".join(
        [",".join(str(i) for i in range(args.encoder_gpus))] * args.num_encoders
    )
    args.encoder_gpu_memory_utilization = (
        args.encoder_gpu_memory_utilization or args.gpu_memory_utilization
    )
    return run_bench.finalize_args(args)


def render_all(args: argparse.Namespace, client: ClientPod) -> dict[str, list[dict]]:
    """Every manifest a run would apply, keyed by file stem."""
    out: dict[str, list[dict[str, Any]]] = {
        "pvc": [render_pvc(args)],
        "scripts-configmap": [render_scripts_configmap(args)],
        "client-pod": client.render(),
    }
    for name in args.arms:
        sys_ = build_pod_system(client.target, args, name)
        for server in sys_.servers:
            assert isinstance(server, PodServer)
            out[f"{name}/{server.name}"] = server.render()
    return out


def dry_run(args: argparse.Namespace, client: ClientPod) -> int:
    if not args.ec_cpu_bytes:
        args.ec_cpu_bytes = _DRY_RUN_EC_CPU_BYTES
        print(
            f"[k8s] dry run: ec_cpu_bytes unknown without the manifest; rendering "
            f"with {gib(args.ec_cpu_bytes)}"
        )
    render_dir = args.out_dir / "manifests" / args.run_id
    for stem, docs in render_all(args, client).items():
        path = render_dir / f"{stem}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump_all(docs, sort_keys=False))
        print(f"\n=== {path} ===")
        for doc in docs:
            if doc["kind"] == "ConfigMap":
                print(f"ConfigMap {doc['metadata']['name']}: {sorted(doc['data'])}")
            else:
                print(yaml.safe_dump(doc, sort_keys=False), end="")
    rate0, conc0 = args.load_points[0]
    num_prompts = args.num_prompts or 0
    for name in args.arms:
        sys_ = build_pod_system(client.target, args, name)
        for server in sys_.servers:
            print(f"\n=== {name}: {server.name} container command ===")
            print(server.launch_script())
        print(f"\n=== {name}: load (in {client.resource}) ===")
        print(
            sys_.front.bench_serve_script(
                num_prompts, rate0, f"{args.work_dir}/bench.json", conc0
            )
        )
    print(f"\n[k8s] dry run: rendered {render_dir}; nothing applied")
    return 0


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.cleanup:
        return cleanup(args)
    oc = None if args.dry_run else Oc(args.namespace)
    client = ClientPod(args, oc)
    if args.dry_run:
        return dry_run(args, client)
    assert oc is not None

    if not args.skip_preflight:
        who = oc.run("whoami", check=False)
        if who.returncode != 0:
            raise SystemExit(f"[k8s] not logged in: {who.stderr.strip()[:200]}")
    ensure_pvc(oc, args)
    oc.apply([render_scripts_configmap(args)])
    client.start()
    if not args.skip_preflight:
        preflight(oc, client, args)
    ensure_workload(client, args)
    expected, num_prompts = run_bench.prepare_workload(client.target, args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "bench.json"
    results: list[dict[str, Any]] = []
    placement: dict[str, list[dict[str, Any]]] = {}
    built: list[System] = []

    def build(target: Target, a: argparse.Namespace, name: str) -> System:
        sys_ = build_pod_system(target, a, name)
        built.append(sys_)
        return sys_

    def persist() -> None:
        out_path.write_text(
            json.dumps(
                {
                    "args": {k: str(v) for k, v in vars(args).items()},
                    "manifest_expected": expected,
                    "results": results,
                    "placement": placement,
                },
                indent=2,
            )
        )

    topology = "same-node" if args.same_node else "multinode"
    try:
        for name in args.arms:
            try:
                new = run_bench.run_arm(client.target, args, name, num_prompts, build)
            finally:
                records = [placement_record(name, s, args) for s in built]
                placement[name] = records
                built.clear()
            for record in records:
                check_placement(record)
            for entry in new:
                if entry["topology"] == "epd":
                    entry["topology"] = topology
            results.extend(new)
            persist()
    except Exception:
        persist()
        print(f"[k8s] partial results saved to {out_path}", file=sys.stderr)
        raise
    finally:
        if args.keep:
            print(f"[k8s] --keep: {client.resource} and the ConfigMap stay up")
        else:
            client.stop()
            oc.delete("configmap", selector(args, "scripts"))

    run_bench.print_table(results)
    print_placement(placement)
    gates_ok = run_bench.check_gates(results)
    print(f"\n[bench] wrote {out_path}")
    return 0 if gates_ok else 1


if __name__ == "__main__":
    sys.exit(main())
