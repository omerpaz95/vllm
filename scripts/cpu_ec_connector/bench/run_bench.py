#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving benchmark for the encoder-cache (EC) connectors.

Replays one workload (from `gen_workload.py`) against six arms that differ
only in topology, connector and what the decode instance receives:

    baseline      one instance, no connector: encodes and decodes itself
    offload       one instance, ECCPUConnector in ec_both: repeats reload from
                  the CPU region instead of re-running the vision tower
    cpu-data      encoder + decode, ECCPUConnector over NIXL; decode gets the
                  pixels plus the connector's transfer params
    cpu-grid      as cpu-data, but the proxy substitutes the image grid for
                  the pixels (the rewrite from PR #50390)
    example-data  encoder + decode, ECExampleConnector over shared storage,
                  pixels forwarded
    example-grid  as example-data, grid substituted

`data` vs `grid` isolates what the grid substitution saves with the transport
held fixed; `cpu` vs `example` compares the transports. `--no-rewrite` on the
EPD proxy is the `data` mode: since the proxy started carrying the connector's
handles regardless of rewriting, no proxy modification is needed.

Load comes from `vllm bench serve` over the `custom_image` dataset, so the
latencies are the ones a client sees. Alongside them the connector's own DEBUG
accounting is read from the server logs, and every arm is gated on having done
what its name says: a connector arm must have loaded entries, a rewrite arm
must have rewritten most of the workload, a fan-out must have reached every
encoder, and every request must have completed. A run that fails a gate is
not a measurement, and this raises instead of printing a delta.

Each arm's servers verify from their fresh logs that they are what was asked
for, because a survivor from the previous arm would answer /health and be
measured under the wrong name.

The servers this launches run on the target: locally, or inside a pod with
`--pod`. All of the target's scratch state lives under `--work-dir`.

Typical use:

    python run_bench.py --workload-dir /data/wl --out-dir results \
        --arms baseline,offload,cpu-grid --max-concurrency 1,4,8

`--frag` runs the offload arm with a region deliberately smaller than the
working set and descriptor counting enabled, to see whether entries still
collapse to one descriptor once the region has churned.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ec_log_stats import (
    decay_report,
    rewritten_items,
    stage_summary,
    summarize,
    window_stats,
)

PROXY = "examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py"
BENCH_DIR = Path(__file__).resolve().parent
QUEUE_SAMPLE_INTERVAL_S = 1.0
_HEALTH_POLL_S = 5.0
# Generous because an orderly exit has to tear down CUDA and, on a connector
# arm, unlink the EC region; escalating early is what leaks multi-GiB /dev/shm
# files.
_STOP_TIMEOUT_S = 180
_DETACH_TIMEOUT_S = 120
_PGID_READ_ATTEMPTS = 10
_PGID_READ_DELAY_S = 1.0
_EC_REGION_MARKER = "Created EC mmap file"
_GPU_PROCESSOR_MARKER = "Running the multi-modal processor on cuda"
# Either line proves the log belongs to a server that got as far as serving;
# the exact wording is version-dependent, uvicorn's is stable.
_STARTUP_MARKERS = ("Application startup complete", "Starting vLLM server on")


@dataclass(frozen=True)
class Arm:
    topology: str  # "single" or "epd"
    connector: str  # "", "ECCPUConnector" or "ECExampleConnector"
    rewrite: bool = False  # EPD only: the proxy substitutes the grid

    @property
    def expects_loads(self) -> bool:
        return bool(self.connector)


ARMS: dict[str, Arm] = {
    "baseline": Arm("single", ""),
    "offload": Arm("single", "ECCPUConnector"),
    "cpu-data": Arm("epd", "ECCPUConnector"),
    "cpu-grid": Arm("epd", "ECCPUConnector", rewrite=True),
    "example-data": Arm("epd", "ECExampleConnector"),
    "example-grid": Arm("epd", "ECExampleConnector", rewrite=True),
}


class ServerMismatchError(RuntimeError):
    """The running system is not the arm that was requested."""


class Target:
    """Runs shell commands locally or inside a pod."""

    def __init__(self, pod: str | None) -> None:
        self.pod = pod

    def _argv(self, script: str) -> list[str]:
        # Not a login shell: a container's /etc/profile resets PATH for root
        # and drops the image's venv, which is how `python` goes missing.
        argv = ["bash", "-c", script]
        if self.pod:
            return ["oc", "exec", self.pod, "--", *argv]
        return argv

    def sh(
        self, script: str, *, timeout: int = 600, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            self._argv(script), capture_output=True, text=True, timeout=timeout
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"command failed ({result.returncode}): {script}\n"
                f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
            )
        return result

    def sh_detached(self, script: str) -> None:
        """Run a command that leaves a daemon behind.

        Output must not be captured: a backgrounded server inherits the pipes
        and never closes them, so waiting on them would block until the timeout
        even though the launching shell exited immediately.
        """
        subprocess.run(
            self._argv(script),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_DETACH_TIMEOUT_S,
            check=True,
        )

    def file_size(self, path: str) -> int:
        out = self.sh(f"stat -c %s {shlex.quote(path)} 2>/dev/null || echo 0").stdout
        return int(out.strip() or 0)

    def read_bytes(self, path: str, start: int, end: int) -> str:
        if end <= start:
            return ""
        quoted = shlex.quote(path)
        return self.sh(f"tail -c +{start + 1} {quoted} | head -c {end - start}").stdout

    def read_text(self, path: str) -> str:
        return self.sh(f"cat {shlex.quote(path)} 2>/dev/null || true").stdout

    def exists(self, path: str) -> bool:
        probe = f"test -s {shlex.quote(path)} && echo ok"
        return bool(self.sh(probe, check=False).stdout.strip())


@dataclass
class BenchServer:
    """One process this harness starts and stops: a vLLM server, or the proxy.

    `command` runs something that is not a vLLM server (the proxy) through the
    same launch and teardown path. `ec_config` is the instance's
    `--ec-transfer-config`, or None for no connector.
    """

    target: Target
    args: argparse.Namespace
    name: str
    port: int
    gpu: str = ""
    command: str = ""
    ec_config: dict[str, Any] | None = None
    gpu_util: float = 0.0
    tp: int = 1
    batched_tokens: int = 0
    extra_args: tuple[str, ...] = ()
    extra_env: tuple[str, ...] = ()
    match: tuple[str, str] = ()
    log_path: str = field(init=False)

    def __post_init__(self) -> None:
        self.log_path = f"{self.args.work_dir}/logs/{self.name}.log"
        if self.ec_config is not None:
            self.ec_config = {"engine_id": self.engine_id, **self.ec_config}

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def pid_file(self) -> str:
        return f"{self.args.work_dir}/{self.name}.pid"

    @property
    def pgid_file(self) -> str:
        return f"{self.args.work_dir}/{self.name}.pgid"

    @property
    def engine_id(self) -> str:
        return f"ec-bench-{self.args.run_id}-{self.name}"

    @property
    def is_vllm(self) -> bool:
        return not self.command

    @property
    def kill_match(self) -> tuple[str, str]:
        """Two substrings that together identify this process and nothing else.

        The port alone also appears in the proxy's `--encode-servers-urls`, and
        the command alone would match every vLLM server on a shared host.
        """
        return self.match or ("cli.main serve", f"--port {self.port}")

    def launch_script(self, *, instrument: bool) -> str:
        """Build the launch command.

        The `cd` matters: from the repository root, `import vllm` resolves the
        source directory as a namespace package. DEBUG matters: the
        connector's transfer accounting is on debug lines. The V2 model runner
        is required by the CPU connector, so every arm runs it.
        """
        env = [
            "VLLM_USE_V2_MODEL_RUNNER=1",
            "VLLM_LOGGING_LEVEL=DEBUG",
            "VLLM_SERVER_DEV_MODE=1",
        ]
        if self.gpu:
            env.insert(0, f"CUDA_VISIBLE_DEVICES={self.gpu}")
        if self.args.hf_home:
            env.append(f"HF_HOME={self.args.hf_home}")
        if instrument:
            env += [
                f"PYTHONPATH={self.args.patch_dir}",
                f"EC_BENCH_FRAG_FILE={self.args.frag_file}",
            ]
        env += list(self.extra_env)
        serve = [self.command] if self.command else self._serve_args()
        # setsid puts the process in a new session so its children -- for a vLLM
        # server the API server, EngineCore and workers -- share one process
        # group that teardown can signal as a unit. The setsid'd shell records
        # its OWN pid, which is the group id, then execs the command into it;
        # reading the group back with `ps` would race setsid.
        #
        # Env assignments precede setsid because setsid execs its first
        # argument, so `setsid VAR=x cmd` would look for a program named "VAR=x".
        #
        # Only the setsid command is backgrounded. `a && b && c & d` would
        # background the whole `&&` list as one subshell, so `$!` would name
        # that subshell and `d` would race its `mkdir`.
        inner = f"echo $$ > {self.pgid_file}; exec " + " ".join(serve)
        return "\n".join(
            [
                f"mkdir -p {self.args.work_dir}/logs && cd {self.args.work_dir} "
                "|| exit 1",
                " ".join(env)
                + f" setsid bash -c {shlex.quote(inner)}"
                + f" < /dev/null > {self.log_path} 2>&1 &",
                f"echo $! > {self.pid_file}; disown",
            ]
        )

    def _serve_args(self) -> list[str]:
        serve = [
            f"{self.args.python} -m vllm.entrypoints.cli.main serve {self.args.model}",
            f"--port {self.port}",
            "--dtype bfloat16",
            f"--max-model-len {self.args.max_model_len}",
            f"--gpu-memory-utilization "
            f"{self.gpu_util or self.args.gpu_memory_utilization}",
            f"--max-num-batched-tokens "
            f"{self.batched_tokens or self.args.max_num_batched_tokens}",
            f"--max-num-seqs {self.args.max_num_seqs}",
            f"--tensor-parallel-size {self.tp}",
            # Identical across arms, and both are required by the accounting
            # this harness reads: iteration details print encoder inputs, and
            # excluding video drops the encoder budget floor 32768 -> 16384.
            "--enable-logging-iteration-details",
            """--limit-mm-per-prompt '{"video":0}'""",
        ]
        if self.ec_config is not None:
            serve.append(
                f"--ec-transfer-config {shlex.quote(json.dumps(self.ec_config))}"
            )
        serve.extend(self.extra_args)
        if self.args.serve_args:
            serve.append(self.args.serve_args)
        return serve

    def start(self, *, instrument: bool = False) -> None:
        self.target.sh(
            f"rm -f {self.log_path} {self.args.frag_file} {self.pgid_file}",
            check=False,
        )
        self.stop()
        print(f"[bench] launching {self.name}")
        self.target.sh_detached(self.launch_script(instrument=instrument))
        # Without a recorded pid, stop() falls back to a pattern match alone;
        # fail here instead of discovering it at teardown.
        pid = self.target.read_text(self.pid_file).strip()
        if not pid.isdigit():
            raise RuntimeError(
                f"launch did not record a pid in {self.pid_file} "
                f"(got {pid!r}); see {self.log_path}"
            )
        pgid = ""
        for _ in range(_PGID_READ_ATTEMPTS):
            pgid = self.target.read_text(self.pgid_file).strip()
            if pgid.isdigit():
                break
            time.sleep(_PGID_READ_DELAY_S)
        if pgid.isdigit():
            print(f"[bench] {self.name} pid {pid}, process group {pgid}")
        else:
            print(
                f"[bench] WARNING: no process group recorded for pid {pid}; "
                "teardown will signal the parent and a port-scoped pattern "
                "only, which can leave EngineCore children running",
                file=sys.stderr,
            )

    def stop(self) -> None:
        """Stop the whole process tree, waiting for it to actually exit.

        SIGTERM to the process group, not the parent alone: EngineCore and the
        workers are children, and signalling only the parent leaves them
        holding the GPU and the port -- which presents as the next arm's server
        never becoming healthy. SIGTERM before SIGKILL so the EC region's
        cleanup runs and unlinks its /dev/shm file. The pattern fallback is
        scoped to this port rather than every vLLM server on the host.
        """
        p1, p2 = self.kill_match
        pattern = f"[{p1[0]}]{p1[1:]}.*{p2}"
        script = f"""
        pid=$(cat {self.pid_file} 2>/dev/null || true)
        pgid=$(cat {self.pgid_file} 2>/dev/null || true)
        own=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
        # Never signal our own group: that would kill this shell and its parent.
        if [ -n "$pgid" ] && [ "$pgid" = "$own" ]; then pgid=""; fi
        if [ -n "$pgid" ]; then
            kill -TERM -"$pgid" 2>/dev/null || true
        else
            if [ -n "$pid" ]; then kill -TERM "$pid" 2>/dev/null || true; fi
            pkill -f "{pattern}" 2>/dev/null || true
        fi
        # Liveness must ignore zombies: a reaped-late <defunct> child still
        # matches, which would make every teardown look hung and escalate to
        # SIGKILL even though the tree exited cleanly. Lines in this shell's
        # own group are skipped so the ps/awk pipeline cannot match itself.
        for _ in $(seq {_STOP_TIMEOUT_S}); do
            alive=$(ps -eo pgid=,stat=,args= | awk \
                -v g="$pgid" -v own="$own" \
                -v p1="{p1}" -v p2="{p2}" '
                $1 == own {{ next }}
                $2 ~ /^Z/ {{ next }}
                ($1 == g && g != "") {{ n++; next }}
                (index($0, p1) > 0 && index($0, p2) > 0) {{ n++ }}
                END {{ print n + 0 }}')
            if [ "$alive" = "0" ]; then
                rm -f {self.pid_file} {self.pgid_file}
                echo stopped
                exit 0
            fi
            sleep 1
        done
        echo escalated
        if [ -n "$pgid" ]; then kill -KILL -"$pgid" 2>/dev/null || true; fi
        if [ -n "$pid" ]; then kill -KILL "$pid" 2>/dev/null || true; fi
        pkill -9 -f "{pattern}" 2>/dev/null || true
        rm -f {self.pid_file} {self.pgid_file}
        """
        result = self.target.sh(script, check=False, timeout=_STOP_TIMEOUT_S + 60)
        if "escalated" in result.stdout:
            detail = (
                f"; the EC region's /dev/shm file for {self.engine_id} may have leaked"
                if self.ec_config
                else ""
            )
            print(
                f"[bench] WARNING: {self.name} needed SIGKILL after "
                f"{_STOP_TIMEOUT_S}s{detail}",
                file=sys.stderr,
            )

    def wait_healthy(self) -> None:
        deadline = time.monotonic() + self.args.startup_timeout_s
        probe = (
            f"curl -s -o /dev/null -w '%{{http_code}}' {self.base_url}/health || true"
        )
        while time.monotonic() < deadline:
            if self.target.sh(probe, check=False).stdout.strip() == "200":
                return
            time.sleep(_HEALTH_POLL_S)
        raise RuntimeError(
            f"{self.name} not healthy within {self.args.startup_timeout_s}s "
            f"(see {self.log_path})"
        )

    def wait_for_log(self, marker: str, timeout_s: int = 120) -> None:
        """For the proxy, which serves no /health endpoint of its own."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if marker in self.target.read_text(self.log_path):
                return
            time.sleep(2.0)
        raise RuntimeError(f"{self.name}: {marker!r} never appeared in {self.log_path}")

    def verify(self) -> None:
        """Confirm the live server is the one just launched, from its own log.

        A survivor from the previous arm would answer /health and have its
        numbers recorded under this arm's name. For the CPU connector the
        region's creation line, tagged with this run's engine id, also proves
        the connector is active and the region is not a reused one.
        """
        log = self.target.read_text(self.log_path)
        if not any(marker in log for marker in _STARTUP_MARKERS):
            raise ServerMismatchError(
                f"{self.log_path} contains none of {_STARTUP_MARKERS}: the "
                "server answering /health is not the one just launched"
            )
        has_region = _EC_REGION_MARKER in log and self.engine_id in log
        connector = (self.ec_config or {}).get("ec_connector", "")
        if connector == "ECCPUConnector" and not has_region:
            raise ServerMismatchError(
                f"{self.name}: no '{_EC_REGION_MARKER}' for {self.engine_id}; "
                "the CPU connector is not active, or a region was reused"
            )
        if not connector and _EC_REGION_MARKER in log:
            raise ServerMismatchError(
                f"{self.name}: an EC region was created, so a connector is "
                "active in an instance that is supposed to have none"
            )
        print(f"[bench] verified {self.name} from {self.log_path}")

    def reset_caches(self) -> None:
        """Drop the prefix and processor caches (VLLM_SERVER_DEV_MODE routes).

        There is no route for the EC region or shared storage; those persist
        for the life of the process, which is what --restart-per-load-point
        is for.
        """
        for endpoint in ("reset_prefix_cache", "reset_mm_cache"):
            self.target.sh(
                f"curl -s -X POST {self.base_url}/{endpoint} >/dev/null || true",
                check=False,
            )

    def bench_serve_script(
        self,
        num_prompts: int,
        rate: str,
        out: str,
        concurrency: int = 0,
        dataset: str = "workload.jsonl",
    ) -> str:
        cmd = [
            f"{self.args.python} -m vllm.entrypoints.cli.main bench serve",
            "--backend openai-chat",
            f"--base-url {self.base_url}",
            "--endpoint /v1/chat/completions",
            f"--model {self.args.model}",
            "--dataset-name custom_image",
            f"--dataset-path {self.args.workload_dir}/{dataset}",
            # The emitted order IS the workload; shuffling would destroy the
            # reuse distances the manifest predicts.
            "--disable-shuffle",
            "--custom-ensure-client-side-data",
            f"--custom-output-len {self.args.output_len}",
            f"--num-prompts {num_prompts}",
            f"--request-rate {rate}",
            "--percentile-metrics ttft,tpot,itl,e2el",
            "--metric-percentiles 50,95,99",
            f"--seed {self.args.seed}",
            "--save-result",
            f"--result-filename {out}",
        ]
        if concurrency:
            cmd.append(f"--max-concurrency {concurrency}")
        env = f"HF_HOME={self.args.hf_home} " if self.args.hf_home else ""
        return f"cd {self.args.work_dir} && {env}{' '.join(cmd)}"


@dataclass
class System:
    """Every process one arm needs, and which of them plays which role."""

    front: BenchServer  # what the load generator talks to
    consumer: BenchServer  # where connector loads land and the LM runs
    encoders: list[BenchServer] = field(default_factory=list)
    proxy: BenchServer | None = None

    @property
    def servers(self) -> list[BenchServer]:
        return [*self.encoders, self.consumer] + ([self.proxy] if self.proxy else [])

    @property
    def vllm_servers(self) -> list[BenchServer]:
        return [s for s in self.servers if s.is_vllm]


def _cpu_connector_config(args: argparse.Namespace, role: str) -> dict[str, Any]:
    extra: dict[str, Any] = {"ec_cpu_bytes": args.ec_cpu_bytes}
    if role != "ec_both":
        # Two processes cannot share one instance's mmap region, so the
        # peer-to-peer transport carries entries across: the producer announces
        # its side-channel address in ec_transfer_params and the consumer
        # dials it. Set inside extra config; ECTransferConfig rejects it as a
        # top-level key.
        extra["ec_enable_nixl"] = True
    return {
        "ec_connector": "ECCPUConnector",
        "ec_role": role,
        "ec_connector_extra_config": extra,
    }


def _connector_config(
    args: argparse.Namespace, arm: Arm, role: str
) -> dict[str, Any] | None:
    if not arm.connector:
        return None
    if arm.connector == "ECCPUConnector":
        return _cpu_connector_config(args, role)
    return {
        "ec_connector": arm.connector,
        "ec_role": role,
        "ec_connector_extra_config": {"shared_storage_path": args.shared_storage_path},
    }


def _gpus(spec: str) -> list[str]:
    """A comma-separated device list; its length is the tensor-parallel size."""
    return [g.strip() for g in spec.split(",") if g.strip()]


def build_system(
    target: Target,
    args: argparse.Namespace,
    name: str,
    server_cls: type[BenchServer] = BenchServer,
) -> System:
    """Every server one arm needs; `server_cls` decides where they run."""
    arm = ARMS[name]
    if arm.topology == "single":
        server = server_cls(
            target=target,
            args=args,
            name="single",
            port=args.port,
            gpu=args.gpu,
            tp=len(_gpus(args.gpu)),
            ec_config=_connector_config(args, arm, "ec_both"),
            extra_args=tuple(filter(None, [args.decode_serve_args])),
        )
        return System(front=server, consumer=server)

    # One encoder per `;`-separated group; the GPUs inside a group are that
    # encoder's tensor-parallel set. A GPU named by several groups is shared,
    # and those encoders split its memory.
    groups = [_gpus(g) for g in args.encoder_devices.split(";") if _gpus(g)]
    if not groups:
        raise SystemExit("[bench] --encoder-devices names no device")
    share = {gpu: sum(gpu in g for g in groups) for g in groups for gpu in g}
    encoder_ports = {args.encoder_port + i for i in range(len(groups))}
    for label, port in (("decode", args.decode_port), ("proxy", args.proxy_port)):
        if port in encoder_ports:
            raise SystemExit(
                f"[bench] the {label} port {port} lies inside the encoder range "
                f"{min(encoder_ports)}-{max(encoder_ports)}; move --{label}-port "
                "or --encoder-port"
            )
    if args.decode_port == args.proxy_port:
        raise SystemExit("[bench] --decode-port and --proxy-port are the same")

    encoders: list[BenchServer] = []
    for index, group in enumerate(groups):
        util = args.encoder_gpu_memory_utilization or round(
            args.gpu_memory_utilization / max(share[gpu] for gpu in group), 3
        )
        env: tuple[str, ...] = ()
        if arm.connector == "ECCPUConnector":
            # Each producer binds its own side channel and announces it through
            # ec_transfer_params; the consumer needs no side-channel setting.
            env = (
                f"VLLM_EC_SIDE_CHANNEL_HOST={args.side_channel_host}",
                f"VLLM_EC_SIDE_CHANNEL_PORT={args.side_channel_port + index}",
            )
        # The encoder needs a token budget several images deep or it can never
        # batch two image requests, which pins it to one image per step and
        # turns every rate=inf number into queue depth.
        extra = [
            "--no-enable-prefix-caching",
            f"--mm-processor-device {args.mm_processor_device}",
            # torch_shm is the one transport that carries device tensors; any
            # other copies the result to host and `auto` declines the
            # accelerator.
            f"--mm-tensor-ipc {args.mm_tensor_ipc}",
            # A different knob despite the similar name: with "shm" the engine
            # keeps no receiver cache, the processed data is refilled in the
            # worker, and the connector reports no grid for the proxy to
            # substitute.
            f"--mm-processor-cache-type {args.mm_processor_cache_type}",
        ]
        if args.mm_encoder_only:
            extra.append("--mm-encoder-only")
        if args.encoder_enforce_eager:
            extra.append("--enforce-eager")
        if args.encoder_serve_args:
            extra.append(args.encoder_serve_args)
        encoders.append(
            server_cls(
                target=target,
                args=args,
                name=f"encoder{index}",
                port=args.encoder_port + index,
                gpu=",".join(group),
                tp=len(group),
                gpu_util=util,
                batched_tokens=args.encoder_max_num_batched_tokens,
                ec_config=_connector_config(args, arm, "ec_producer"),
                extra_env=env,
                extra_args=tuple(extra),
            )
        )
    decode = server_cls(
        target=target,
        args=args,
        name="decode",
        port=args.decode_port,
        gpu=args.decode_gpu,
        tp=len(_gpus(args.decode_gpu)),
        ec_config=_connector_config(args, arm, "ec_consumer"),
        # Without this the decode instance rejects an image_embeds part, which
        # is what a rewritten request is made of. Set in every EPD arm: a flag
        # that differs between arms is a confound rather than a switch.
        extra_args=tuple(filter(None, ["--enable-mm-embeds", args.decode_serve_args])),
    )
    encode_urls = ",".join(e.base_url for e in encoders)
    proxy_cmd = " ".join(
        [
            f"{args.python} {args.proxy_script}",
            f"--host {args.proxy_host} --port {args.proxy_port}",
            f"--encode-servers-urls {encode_urls}",
            "--prefill-servers-urls disable",
            f"--decode-servers-urls {decode.base_url}",
        ]
        + ([] if arm.rewrite else ["--no-rewrite"])
    )
    proxy = server_cls(
        target=target,
        args=args,
        name="proxy",
        port=args.proxy_port,
        command=proxy_cmd,
        match=("disagg_epd_proxy.py", f"--port {args.proxy_port}"),
    )
    return System(front=proxy, consumer=decode, encoders=encoders, proxy=proxy)


# --------------------------------------------------------------------------
# Queue-depth sampling
# --------------------------------------------------------------------------

# Samples vllm:num_requests_{running,waiting} from every instance's /metrics.
# Queue depth is what distinguishes "the system is working" from "the system
# is a queue with a benchmark attached", which is what invalidated the first
# rate=inf measurements. Takes "name=base_url" pairs and emits
# "epoch,name,metric,value".
_QUEUE_SAMPLER = r"""
while true; do
    NOW=$(date +%s.%N)
    for pair in "$@"; do
        NAME=${pair%%=*}
        URL=${pair#*=}
        curl -s --max-time 2 "${URL}/metrics" 2>/dev/null \
            | grep -E '^vllm:num_requests_(running|waiting)[ {]' \
            | awk -v t="$NOW" -v n="$NAME" \
                '{split($1, f, "{"); print t "," n "," f[1] "," $NF}' \
            >> "$OUT"
    done
    sleep "$INTERVAL"
done
"""


def start_queue_sampler(target: Target, args: argparse.Namespace, sys_: System):
    pairs = " ".join(f"{s.name}={s.base_url}" for s in sys_.vllm_servers)
    csv = args.queue_csv
    target.sh(f"rm -f {csv}", check=False)
    target.sh_detached(
        f"INTERVAL={QUEUE_SAMPLE_INTERVAL_S} OUT={csv} setsid bash -c "
        f"{shlex.quote(_QUEUE_SAMPLER)} _ {pairs} < /dev/null > /dev/null 2>&1 & "
        f"echo $! > {csv}.pid; disown"
    )


def stop_queue_sampler(target: Target, args: argparse.Namespace) -> None:
    csv = args.queue_csv
    target.sh(
        f"pid=$(cat {csv}.pid 2>/dev/null || true); "
        f'if [ -n "$pid" ]; then kill -TERM -"$pid" 2>/dev/null || '
        f'kill "$pid" 2>/dev/null || true; fi; rm -f {csv}.pid',
        check=False,
    )


def queue_stats(csv_text: str, t_start: float, t_end: float) -> dict[str, Any]:
    """Peak and mean queue depth per instance over one load point.

    `waiting` is the number of requests admitted but not yet running: if it
    stays near zero the measurement reflects work, and if it tracks the offered
    concurrency the measurement reflects the queue instead.
    """
    series: dict[tuple[str, str], list[float]] = {}
    for line in csv_text.splitlines():
        parts = line.split(",")
        if len(parts) < 4:
            continue
        # Prometheus label blocks contain commas, so the metric field cannot be
        # assumed comma-free: take the ends and treat the middle as the name.
        stamp, name, value = parts[0], parts[1], parts[-1]
        metric = ",".join(parts[2:-1])
        try:
            when, amount = float(stamp), float(value)
        except ValueError:
            continue
        if not t_start <= when <= t_end:
            continue
        key = (name, metric.split("{")[0].replace("vllm:num_requests_", ""))
        series.setdefault(key, []).append(amount)
    out: dict[str, Any] = {"samples": sum(len(v) for v in series.values())}
    for (name, metric), values in sorted(series.items()):
        out[f"{name}_{metric}_max"] = round(max(values), 1)
        out[f"{name}_{metric}_mean"] = round(sum(values) / len(values), 2)
    return out


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------


def image_refs_in_prefix(target: Target, workload_dir: str, num_prompts: int) -> int:
    """Image references in the first `num_prompts` lines of the workload.

    The manifest counts the whole workload, so a short run's rewrite count
    must be measured against the prefix it actually sent. `--disable-shuffle`
    means the client replays the file in order, so the prefix is exact.
    """
    text = target.read_text(f"{workload_dir}/workload.jsonl")
    refs = 0
    for index, line in enumerate(text.splitlines()):
        if index >= num_prompts:
            break
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        refs += sum(
            1
            for part in record.get("content", [])
            if part.get("type") in ("image", "image_url", "image_embeds")
        )
    return refs


def check_rewrite(
    name: str, rewrote: int, image_refs: int, floor: float, when: str
) -> float:
    """Coverage of the grid substitution, raising when it is too low to measure.

    The proxy logs "Rewrote N" for any N >= 1 and falls back per item, so
    presence proves nothing about how much of the workload was covered. An
    item is rewritable only when the encoder reported a grid for it, which
    needs its processed data on the scheduler side -- missing under
    `mm_processor_cache_type=shm`, and legitimately absent for a repeat the
    encoder served from its processor cache. `floor=0` only checks presence,
    which is what the warmup's handful of requests can support.
    """
    rewrite = ARMS[name].rewrite
    coverage = rewrote / image_refs if image_refs else 0.0
    if not rewrite:
        if rewrote:
            raise ServerMismatchError(
                f"{name} ({when}): --no-rewrite was passed, but the proxy still "
                f"rewrote {rewrote} item(s)"
            )
        return 0.0
    if not rewrote:
        raise ServerMismatchError(
            f"{name} ({when}): the proxy rewrote nothing, so the grid "
            "substitution never engaged"
        )
    if coverage < floor:
        raise ServerMismatchError(
            f"{name} ({when}): the proxy rewrote {rewrote} of {image_refs} image "
            f"references ({coverage:.0%}), below the {floor:.0%} floor. Check the "
            "encoder is not running with mm_processor_cache_type=shm; lower "
            "--min-rewrite-coverage only after that."
        )
    return coverage


def check_loads(name: str, when: str, consumer: dict[str, Any]) -> None:
    """A connector arm that transferred nothing looks, in the timings, exactly
    like the no-connector arm; assert the direction both ways."""
    loads = consumer["ec_load_entries"]
    if ARMS[name].expects_loads and not loads:
        raise ServerMismatchError(
            f"{name} ({when}): the consumer loaded no encodings, so nothing was "
            "transferred and this arm measures local recompute"
        )
    if not ARMS[name].expects_loads and loads:
        raise ServerMismatchError(
            f"{name} ({when}): the consumer loaded {loads} encodings, but this "
            "arm is not supposed to transfer anything"
        )


def check_fanout(name: str, when: str, per_encoder: dict[str, int]) -> None:
    """If the proxy's round-robin left an encoder idle, fewer encoders were
    measured than configured, with the rest burning memory for nothing."""
    idle = [enc for enc, done in per_encoder.items() if not done]
    if len(per_encoder) > 1 and idle:
        raise ServerMismatchError(
            f"{name} ({when}): {idle} computed no encoder inputs, so the fan-out "
            f"reached only {len(per_encoder) - len(idle)} of {len(per_encoder)} "
            "encoders"
        )


def verify_epd_engaged(
    name: str, args: argparse.Namespace, sys_: System, when: str
) -> None:
    """Fail unless the transform device and the rewrite switch took effect.

    The GPU transform declines silently when its preconditions are unmet, and
    a proxy that rewrote nothing looks identical in the timings to one that
    was asked not to.
    """
    encoder = sys_.encoders[0]
    on_gpu = _GPU_PROCESSOR_MARKER in encoder.target.read_text(encoder.log_path)
    if (args.mm_processor_device == "cuda") != on_gpu:
        raise ServerMismatchError(
            f"{name}: asked for the image transform on {args.mm_processor_device}, "
            f"but the encoder log {'shows' if on_gpu else 'does not show'} "
            f"{_GPU_PROCESSOR_MARKER!r}; the encoder log states the reason"
        )
    assert sys_.proxy is not None
    rewrote = rewritten_items(sys_.proxy.target.read_text(sys_.proxy.log_path))
    check_rewrite(name, rewrote, args.image_refs, 0.0, when)
    print(f"[bench] verified {name}: proxy rewrote {rewrote} item(s) during {when}")


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def parse_frag(text: str) -> dict[str, Any]:
    """Aggregate the descriptor-count JSONL the instrumentation patch writes."""
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        return {"rows": 0, "note": "no descriptor data (patch not active?)"}
    by_caller: dict[str, dict[str, Any]] = {}
    for row in rows:
        agg = by_caller.setdefault(
            row["caller"], {"blocks": 0, "descriptors": 0, "calls": 0, "series": []}
        )
        agg["blocks"] += row["blocks"]
        agg["descriptors"] += row["descriptors"]
        agg["calls"] += row["calls"]
        agg["series"].append((row["t_s"], row["blocks_per_descriptor"]))
    for agg in by_caller.values():
        agg["series"].sort()
        agg["blocks_per_descriptor_mean"] = (
            round(agg["blocks"] / agg["descriptors"], 2) if agg["descriptors"] else 0.0
        )
        first, last = agg["series"][0][1], agg["series"][-1][1]
        agg["blocks_per_descriptor_first"] = first
        agg["blocks_per_descriptor_last"] = last
        agg["verdict"] = (
            "runs held up" if last >= 0.9 * first else "runs shortened (fragmenting)"
        )
    return {"rows": len(rows), "by_caller": by_caller}


def measure_point(
    target: Target,
    args: argparse.Namespace,
    name: str,
    sys_: System,
    num_prompts: int,
    rate: str,
    conc: int,
) -> dict[str, Any]:
    """One load point: drive the workload once and account for it.

    `target` is where the load generator runs; each server's log is read
    through that server's own target, which differs when the servers are
    pods.
    """
    when = f"rate={rate} c={conc}"
    for server in sys_.vllm_servers:
        server.reset_caches()
    marks = {s.name: s.target.file_size(s.log_path) for s in sys_.servers}
    out_path = f"{args.work_dir}/bench_{name}_{rate}_c{conc}.json"
    print(f"[bench] {name}: {when} ({conc or 'unbounded'}), {num_prompts} prompts")
    t_start = time.time()
    target.sh(
        sys_.front.bench_serve_script(num_prompts, rate, out_path, conc),
        timeout=args.bench_timeout_s,
    )
    t_end = time.time()
    # Completion reports land after the last response.
    time.sleep(args.settle_s)
    slices = {
        s.name: s.target.read_bytes(
            s.log_path, marks[s.name], s.target.file_size(s.log_path)
        )
        for s in sys_.servers
    }
    raw = target.read_text(out_path)
    client = json.loads(raw) if raw.strip() else {}
    done = client.get("completed", 0)
    if done < num_prompts:
        raise ServerMismatchError(
            f"{name} ({when}): only {done} of {num_prompts} requests completed, so "
            f"the latencies describe the few that survived; see "
            f"{sys_.consumer.log_path}"
        )
    consumer = summarize(slices[sys_.consumer.name])
    check_loads(name, when, consumer)
    per_encoder = {
        e.name: summarize(slices[e.name])["encoder_inputs_computed"]
        for e in sys_.encoders
    }
    check_fanout(name, when, per_encoder)
    entry: dict[str, Any] = {
        "arm": name,
        "topology": ARMS[name].topology,
        "request_rate": rate,
        "concurrency": conc,
        "client": client,
        # Where loads land and the language model runs: the single instance,
        # or the decode instance.
        "server": consumer,
        "encoders": len(sys_.encoders),
        "per_encoder_inputs": per_encoder,
        "queue": queue_stats(target.read_text(args.queue_csv), t_start, t_end),
    }
    if sys_.proxy is not None:
        rewrote = rewritten_items(slices["proxy"])
        entry["encoder"] = summarize(slices[sys_.encoders[0].name])
        # Per-stage attribution the proxy logs, which says whether a win came
        # from the decode side or elsewhere.
        entry["stages"] = stage_summary(slices["proxy"])
        entry["rewritten"] = rewrote
        entry["rewrite_coverage"] = round(
            check_rewrite(
                name, rewrote, args.image_refs, args.min_rewrite_coverage, when
            ),
            4,
        )
    if args.frag:
        log = slices[sys_.consumer.name]
        entry["windows"] = window_stats(log, args.frag_window_s)
        entry["decay"] = decay_report(entry["windows"], "load")
        entry["descriptors"] = parse_frag(target.read_text(args.frag_file))
    return entry


def run_points(
    target: Target,
    args: argparse.Namespace,
    name: str,
    num_prompts: int,
    points: list[tuple[str, int]],
    build: Callable[[Target, argparse.Namespace, str], System] = build_system,
) -> list[dict[str, Any]]:
    """Start the arm's system once, measure `points` on it, tear it down."""
    sys_ = build(target, args, name)
    if ARMS[name].topology == "epd":
        # A config must not inherit encodings the previous one saved: those
        # would be free hits it never paid for.
        target.sh(
            f"rm -rf {args.shared_storage_path} && mkdir -p {args.shared_storage_path}",
            check=False,
        )
    try:
        for server in sys_.vllm_servers:
            server.start(instrument=args.frag)
            server.wait_healthy()
            server.verify()
        if sys_.proxy is not None:
            sys_.proxy.start()
            sys_.proxy.wait_for_log("Uvicorn running")

        # The warmup drives images that are in no measured request, so it pays
        # every first-request cost (lazy kernel loading, the first NIXL
        # session, region page faults) without seeding a cache the
        # measurement then hits.
        print(f"[bench] {name}: warmup ({args.warmup_prompts} requests)")
        target.sh(
            sys_.front.bench_serve_script(
                args.warmup_prompts,
                "inf",
                f"{args.work_dir}/warmup.json",
                dataset="warmup.jsonl",
            ),
            timeout=args.bench_timeout_s,
            check=False,
        )
        if sys_.proxy is not None:
            verify_epd_engaged(name, args, sys_, "warmup")

        results = []
        start_queue_sampler(target, args, sys_)
        try:
            for rate, conc in points:
                results.append(
                    measure_point(target, args, name, sys_, num_prompts, rate, conc)
                )
        finally:
            stop_queue_sampler(target, args)
        return results
    finally:
        for server in reversed(sys_.servers):
            server.stop()


def run_arm(
    target: Target,
    args: argparse.Namespace,
    name: str,
    num_prompts: int,
    build: Callable[[Target, argparse.Namespace, str], System] = build_system,
) -> list[dict[str, Any]]:
    """Measure one arm across every load point.

    With `--restart-per-load-point` each point gets fresh servers, so the EC
    region (which no reset route touches) starts empty every time and every
    point pays its saves. Otherwise the first point pays them and later points
    run against a warm region: arm-to-arm comparison at one point stays fair,
    the within-arm scaling curve does not.
    """
    if args.restart_per_load_point:
        results = []
        for point in args.load_points:
            results.extend(run_points(target, args, name, num_prompts, [point], build))
        return results
    return run_points(target, args, name, num_prompts, args.load_points, build)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _metric(client: dict[str, Any], key: str) -> Any:
    value = client.get(key)
    return round(value, 2) if isinstance(value, (int, float)) else "-"


def _max_encoder_queue(queue: dict[str, Any]) -> Any:
    depths = [
        v for k, v in queue.items() if k.startswith("encoder") and "waiting_max" in k
    ]
    return max(depths) if depths else "-"


def print_table(results: list[dict[str, Any]]) -> None:
    """One row per (arm, load point), with throughput ratios against baseline.

    `saves`/`loads` are connector entries on the consumer side; `enc_inputs`
    is what the consumer still computed itself. `encode`/`dec_ttfb` are the
    proxy's stage medians, EPD arms only. x_base compares whatever GPU sets
    --gpu, --encoder-devices and --decode-gpu gave each arm, so a baseline on
    one GPU against an EPD pair on two credits disaggregation with the extra
    hardware.
    """
    header = (
        f"{'arm':<13} {'rate':>5} {'conc':>5} {'ttft_p50':>9} {'ttft_p99':>9} "
        f"{'out_tok/s':>10} {'x_base':>7} {'saves':>6} {'loads':>6} "
        f"{'enc_inputs':>10} {'encode':>8} {'dec_ttfb':>9} {'encQmax':>8} "
        f"{'decQmax':>8} {'rewrote':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    base = {
        (r["request_rate"], r["concurrency"]): r["client"].get("output_throughput")
        for r in results
        if r["arm"] == "baseline"
    }
    for r in results:
        client, server = r["client"], r["server"]
        stages, queue = r.get("stages", {}), r.get("queue", {})
        ref = base.get((r["request_rate"], r["concurrency"]))
        got = client.get("output_throughput")
        ratio = f"{got / ref:.2f}" if isinstance(ref, float) and got and ref else "-"
        consumer = "single" if r["topology"] == "single" else "decode"
        print(
            f"{r['arm']:<13} {str(r['request_rate']):>5} {r['concurrency']:>5} "
            f"{_metric(client, 'median_ttft_ms'):>9} "
            f"{_metric(client, 'p99_ttft_ms'):>9} "
            f"{_metric(client, 'output_throughput'):>10} {ratio:>7} "
            f"{server['ec_save_entries']:>6} {server['ec_load_entries']:>6} "
            f"{server['encoder_inputs_computed']:>10} "
            f"{stages.get('encode_ms_median', '-'):>8} "
            f"{stages.get('decode_ttfb_ms_median', '-'):>9} "
            f"{_max_encoder_queue(queue):>8} "
            f"{queue.get(f'{consumer}_waiting_max', '-'):>8} "
            f"{r.get('rewritten', '-'):>8}"
        )


def check_gates(results: list[dict[str, Any]]) -> bool:
    """Every connector arm must have computed fewer encoder inputs on its
    consumer than the baseline did at the same load point, and loaded
    something; otherwise the arms differ by noise and configuration rather
    than by the mechanism under test."""
    base = {
        (r["request_rate"], r["concurrency"]): r["server"]["encoder_inputs_computed"]
        for r in results
        if r["arm"] == "baseline"
    }
    if not base:
        print("\n[bench] no baseline arm; the encoder-compute gate needs one")
        return True
    ok = True
    for r in results:
        if not ARMS[r["arm"]].expects_loads:
            continue
        ref = base.get((r["request_rate"], r["concurrency"]))
        if ref is None:
            continue
        loads = r["server"]["ec_load_entries"]
        computed = r["server"]["encoder_inputs_computed"]
        passed = loads > 0 and computed < ref
        ok = ok and passed
        avoided = f"{(1 - computed / ref) * 100:.1f}%" if ref else "-"
        print(
            f"[bench] {r['arm']} rate={r['request_rate']} c={r['concurrency']}: "
            f"{'PASS' if passed else 'FAIL'} ({loads} loads; consumer computed "
            f"{computed} vs baseline {ref}, avoided {avoided})"
        )
    return ok


def print_frag(results: list[dict[str, Any]]) -> None:
    for entry in results:
        print(f"\n[bench] bandwidth decay: {entry['decay']}")
        for caller, agg in entry["descriptors"].get("by_caller", {}).items():
            print(
                f"[bench] {caller}: blocks/descriptor "
                f"{agg['blocks_per_descriptor_first']} -> "
                f"{agg['blocks_per_descriptor_last']} "
                f"(mean {agg['blocks_per_descriptor_mean']}) -- {agg['verdict']}"
            )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def add_common_options(p: argparse.ArgumentParser) -> None:
    """Options that describe the arms, the servers and the load points, which
    hold wherever the servers run. Placement options belong to the driver."""
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--out-dir", type=Path, default=Path("bench_results"))
    p.add_argument(
        "--arms",
        default=",".join(ARMS),
        help=f"comma-separated subset of {', '.join(ARMS)}",
    )
    p.add_argument("--request-rates", default="inf")
    p.add_argument(
        "--max-concurrency",
        default="1,4,8",
        help="comma-separated in-flight limits to sweep. 0 means unbounded, "
        "which at --request-rate inf floods the system and makes every latency "
        "a queue measurement",
    )
    p.add_argument("--num-prompts", type=int, default=0, help="0 = whole workload")
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument(
        "--restart-per-load-point",
        action="store_true",
        help="fresh servers for every load point, so the EC region starts "
        "empty each time and every point pays its saves",
    )
    p.add_argument("--ec-cpu-bytes", type=int, default=0, help="0 = from manifest")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument(
        "--encoder-max-num-batched-tokens",
        type=int,
        default=65536,
        help="token budget for an encode-only instance; must exceed one image's "
        "token count several times over or it cannot batch image requests",
    )
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument(
        "--serve-args",
        default="",
        help="extra `vllm serve` arguments appended to every instance",
    )
    p.add_argument(
        "--decode-serve-args",
        default="",
        help="extra arguments for the decode and single-instance servers",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--startup-timeout-s", type=int, default=900)
    p.add_argument("--bench-timeout-s", type=int, default=3600)
    p.add_argument("--settle-s", type=float, default=5.0)
    p.add_argument("--port", type=int, default=8100, help="single-instance arms")
    epd = p.add_argument_group("EPD arms")
    epd.add_argument(
        "--encoder-serve-args",
        default="",
        help="extra `vllm serve` arguments for the encoder instances",
    )
    epd.add_argument(
        "--mm-encoder-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip the language model on encoder instances (~16 GB -> ~1.4 GB), "
        "which is what makes several encoders fit one GPU",
    )
    epd.add_argument(
        "--encoder-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="the EPD example requires eager mode on encoder instances",
    )
    epd.add_argument("--encoder-port", type=int, default=8101, help="+i per encoder")
    epd.add_argument("--decode-port", type=int, default=8200)
    epd.add_argument("--proxy-port", type=int, default=8000)
    epd.add_argument("--side-channel-port", type=int, default=5577, help="+i")
    epd.add_argument("--shared-storage-path", default="", help="example connector")
    epd.add_argument(
        "--mm-processor-device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="where the encoder's image transform runs; verified from its log",
    )
    epd.add_argument(
        "--mm-tensor-ipc",
        default="torch_shm",
        help="torch_shm is required for the transform to run on the accelerator",
    )
    epd.add_argument(
        "--mm-processor-cache-type",
        default="lru",
        choices=("lru", "shm"),
        help="pinned on the encoder because it decides whether the grid is "
        "reported at all; 'shm' moves the refill into the worker, where the "
        "connector cannot see it",
    )
    epd.add_argument(
        "--min-rewrite-coverage",
        type=float,
        default=0.5,
        help="fraction of image references the proxy must rewrite for a grid "
        "arm to count as engaged; repeats served from the encoder's processor "
        "cache legitimately keep their pixels",
    )
    p.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dry-run", action="store_true")


def add_single_node_options(p: argparse.ArgumentParser) -> None:
    """Where the servers run and which GPUs they get, for one host."""
    p.add_argument("--pod", default="", help="run the servers inside this pod (oc)")
    p.add_argument(
        "--python",
        default="",
        help="interpreter on the target; defaults to this one locally, or "
        "`python` inside a pod",
    )
    p.add_argument(
        "--vllm-repo",
        default=str(BENCH_DIR.parents[2]),
        help="checkout on the target, for the EPD proxy script",
    )
    p.add_argument(
        "--work-dir",
        default="/tmp/ec_bench",
        help="scratch directory on the target: logs, pid files, bench outputs",
    )
    p.add_argument("--hf-home", default="", help="HF_HOME for the servers")
    p.add_argument("--workload-dir", required=True, help="from gen_workload.py")
    p.add_argument(
        "--gpu",
        default="0",
        help="GPUs for the single-instance arms, comma-separated; the count is "
        "the tensor-parallel size",
    )
    p.add_argument(
        "--frag",
        action="store_true",
        help="offload arm only, region under the working set, descriptor "
        "counting on: measures whether entries stay contiguous as it churns",
    )
    p.add_argument("--frag-window-s", type=float, default=30.0)
    p.add_argument(
        "--patch-dir",
        default=str(BENCH_DIR / "patches"),
        help="directory holding the --frag sitecustomize.py, on the target",
    )
    epd = p.add_argument_group("EPD placement")
    epd.add_argument(
        "--encoder-devices",
        default="0",
        help="one encoder instance per `;`-separated group, each group a "
        "comma-separated GPU list whose length is that encoder's tensor-parallel "
        "size: '0;0' is two encoders sharing GPU 0 (they split its memory), "
        "'0,1;2,3' is two TP=2 encoders. Accepts indices or MIG UUIDs",
    )
    epd.add_argument(
        "--encoder-gpu-memory-utilization",
        type=float,
        default=0.0,
        help="per-encoder memory share; 0 divides --gpu-memory-utilization by "
        "the number of encoders sharing that device",
    )
    epd.add_argument(
        "--decode-gpu",
        default="1",
        help="GPUs for the decode instance, comma-separated; the count is the "
        "tensor-parallel size",
    )
    epd.add_argument(
        "--proxy-host",
        default="127.0.0.1",
        help="address the EPD proxy binds; the load generator dials it",
    )
    epd.add_argument(
        "--side-channel-host",
        default="127.0.0.1",
        help="address each producer binds its side channel to and announces "
        "to the consumer",
    )


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Derive the load points and scratch paths; validate the arm list."""
    args.shared_storage_path = args.shared_storage_path or f"{args.work_dir}/shared"
    args.queue_csv = f"{args.work_dir}/queue.csv"
    args.frag_file = f"{args.work_dir}/frag.jsonl"
    rates = [r.strip() for r in args.request_rates.split(",") if r.strip()]
    concurrencies = [int(c) for c in args.max_concurrency.split(",") if c.strip()]
    args.load_points = [(r, c) for r in rates for c in concurrencies or [0]]
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.frag:
        args.arms = ["offload"]
    unknown = set(args.arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"[bench] unknown arm(s): {sorted(unknown)}")
    return args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_common_options(p)
    add_single_node_options(p)
    args = p.parse_args()
    args.python = args.python or ("python" if args.pod else sys.executable)
    args.proxy_script = f"{args.vllm_repo}/{PROXY}"
    return finalize_args(args)


def load_manifest(target: Target, args: argparse.Namespace) -> dict[str, Any]:
    raw = target.read_text(f"{args.workload_dir}/manifest.json")
    manifest = json.loads(raw) if raw.strip() else {}
    if not manifest:
        raise SystemExit(
            f"[bench] no manifest.json in {args.workload_dir}; run gen_workload.py"
        )
    # A manifest can outlive its images. Check they are there rather than
    # failing deep inside the load generator with a confusing error.
    pool = manifest.get("pool") or []
    probes = sorted({pool[0]["path"], pool[-1]["path"]}) if pool else []
    missing = [path for path in probes if not target.exists(path)]
    if missing:
        raise SystemExit(
            f"[bench] the manifest references images that are not on the target "
            f"({missing}); rebuild the pool with gen_workload.py"
        )
    return manifest


def prepare_workload(
    target: Target, args: argparse.Namespace
) -> tuple[dict[str, Any], int]:
    """Read the workload's manifest and settle what the run takes from it.

    Returns the manifest's `expected` block and the request count; sets
    `image_refs`, `warmup_prompts` and a manifest-derived `ec_cpu_bytes`.
    """
    manifest = load_manifest(target, args)
    expected = manifest["expected"]
    num_prompts = args.num_prompts or manifest["sequence"]["requests"]
    args.image_refs = image_refs_in_prefix(target, args.workload_dir, num_prompts)
    warmup = target.read_text(f"{args.workload_dir}/warmup.jsonl")
    args.warmup_prompts = sum(1 for line in warmup.splitlines() if line.strip())
    if not args.warmup_prompts:
        raise SystemExit(
            f"[bench] no warmup.jsonl in {args.workload_dir}; rebuild the workload "
            "with the current gen_workload.py, which holds out warmup images"
        )
    if not args.ec_cpu_bytes:
        args.ec_cpu_bytes = expected[
            "fragmentation_arm_ec_cpu_bytes" if args.frag else "suggested_ec_cpu_bytes"
        ]
    print(
        f"[bench] workload {num_prompts} requests ({args.image_refs} image refs), "
        f"working set {expected['working_set_bytes'] / 1024**3:.2f} GiB, "
        f"ec_cpu_bytes {args.ec_cpu_bytes / 1024**3:.2f} GiB, max hit rate "
        f"{expected['max_hit_rate'] * 100:.1f}%"
    )
    return expected, num_prompts


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    target = Target(args.pod or None)
    expected, num_prompts = prepare_workload(target, args)

    if args.dry_run:
        rate0, conc0 = args.load_points[0]
        for name in args.arms:
            sys_ = build_system(target, args, name)
            for server in sys_.servers:
                print(f"\n=== {name}: {server.name} ===")
                print(server.launch_script(instrument=args.frag))
            print(f"\n=== {name}: load ===")
            print(
                sys_.front.bench_serve_script(
                    num_prompts, rate0, f"{args.work_dir}/bench.json", conc0
                )
            )
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / ("frag.json" if args.frag else "bench.json")

    def persist(results: list[dict[str, Any]]) -> None:
        """After every arm, so a failure does not discard the earlier arms."""
        out_path.write_text(
            json.dumps(
                {
                    "args": {k: str(v) for k, v in vars(args).items()},
                    "manifest_expected": expected,
                    "results": results,
                },
                indent=2,
            )
        )

    results: list[dict[str, Any]] = []
    try:
        for name in args.arms:
            results.extend(run_arm(target, args, name, num_prompts))
            persist(results)
    except Exception:
        persist(results)
        print(f"[bench] partial results saved to {out_path}", file=sys.stderr)
        raise

    print_table(results)
    if args.frag:
        print_frag(results)
        gates_ok = True
    else:
        gates_ok = check_gates(results)
    print(f"\n[bench] wrote {out_path}")
    return 0 if gates_ok else 1


if __name__ == "__main__":
    sys.exit(main())
