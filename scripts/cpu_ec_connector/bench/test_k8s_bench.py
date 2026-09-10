# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Schema checks on what `k8s_bench.py --dry-run` renders.

No cluster is needed: the manifests are built in-process and checked for the
invariants a first cluster run would otherwise discover the slow way. Runs
under pytest, or directly as a script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import k8s_bench  # noqa: E402
from k8s_bench import PVC_MOUNT, ClientPod, PodServer, build_pod_system  # noqa: E402

BASE_ARGV = [
    "k8s_bench.py",
    "--namespace",
    "ns1",
    "--run-id",
    "t1",
    "--arms",
    "baseline,cpu-grid,example-data",
    "--num-encoders",
    "2",
    "--encoder-gpus",
    "2",
    "--decode-gpus",
    "4",
    "--ec-cpu-bytes",
    str(16 * 1024**3),
]


def parse(*extra: str):
    argv, sys.argv = sys.argv, [*BASE_ARGV, *extra]
    try:
        return k8s_bench.parse_args()
    finally:
        sys.argv = argv


def rendered(args, arm: str) -> dict[str, list[dict]]:
    sys_ = build_pod_system(ClientPod(args, None).target, args, arm)
    out = {}
    for server in sys_.servers:
        assert isinstance(server, PodServer)
        docs = yaml.safe_load_all(yaml.safe_dump_all(server.render()))
        out[server.name] = list(docs)
    return out


def deployment(docs: list[dict]) -> dict:
    return next(d for d in docs if d["kind"] == "Deployment")


def service(docs: list[dict]) -> dict:
    return next(d for d in docs if d["kind"] == "Service")


def container(docs: list[dict]) -> dict:
    return deployment(docs)["spec"]["template"]["spec"]["containers"][0]


def env_of(docs: list[dict]) -> dict[str, dict]:
    return {e["name"]: e for e in container(docs)["env"]}


def test_labels_and_names():
    args = parse()
    for arm in args.arms:
        for name, docs in rendered(args, arm).items():
            for doc in docs:
                meta = doc["metadata"]
                assert meta["namespace"] == "ns1"
                assert meta["name"] == f"ec-bench-t1-{name}"
                assert meta["labels"] == {
                    "app": "vllm-ec-bench",
                    "run-id": "t1",
                    "role": name,
                }
            dep = deployment(docs)
            pod_labels = dep["spec"]["template"]["metadata"]["labels"]
            assert (
                dep["spec"]["selector"]["matchLabels"]
                == service(docs)["spec"]["selector"]
            )
            assert set(dep["spec"]["selector"]["matchLabels"]) <= set(pod_labels)


def test_gpu_count_matches_tp_flag():
    args = parse()
    expect = {"single": 4, "decode": 4, "encoder0": 2, "encoder1": 2}
    for arm in args.arms:
        for name, docs in rendered(args, arm).items():
            if name == "proxy":
                assert "nvidia.com/gpu" not in container(docs)["resources"].get(
                    "limits", {}
                )
                continue
            resources = container(docs)["resources"]
            assert resources["limits"]["nvidia.com/gpu"] == str(expect[name])
            assert resources["requests"]["nvidia.com/gpu"] == str(expect[name])
            command = container(docs)["command"][2]
            assert f"--tensor-parallel-size {expect[name]}" in command
            assert "CUDA_VISIBLE_DEVICES" not in command


def test_anti_affinity_on_gpu_pods_unless_same_node():
    args = parse()
    docs = rendered(args, "cpu-grid")
    for name in ("encoder0", "encoder1", "decode"):
        spec = deployment(docs[name])["spec"]["template"]["spec"]
        term = spec["affinity"]["podAntiAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ][0]
        assert term["topologyKey"] == "kubernetes.io/hostname"
        assert term["labelSelector"]["matchLabels"]["tier"] == "gpu"
        assert term["labelSelector"]["matchLabels"]["run-id"] == "t1"
        assert spec["containers"][0]["resources"]["limits"]["nvidia.com/gpu"]
        assert (
            deployment(docs[name])["spec"]["template"]["metadata"]["labels"]["tier"]
            == "gpu"
        )
    proxy_spec = deployment(docs["proxy"])["spec"]["template"]["spec"]
    assert "affinity" not in proxy_spec
    assert (
        "tier"
        not in deployment(docs["proxy"])["spec"]["template"]["metadata"]["labels"]
    )

    same = parse("--same-node")
    for name, doc in rendered(same, "cpu-grid").items():
        assert "podAntiAffinity" not in deployment(doc)["spec"]["template"]["spec"].get(
            "affinity", {}
        )


def test_node_selection():
    args = parse("--exclude-nodes", "n1,n2", "--node-selector", "gpu=h100,pool=a")
    docs = rendered(args, "cpu-grid")
    spec = deployment(docs["decode"])["spec"]["template"]["spec"]
    assert spec["nodeSelector"] == {"gpu": "h100", "pool": "a"}
    expr = spec["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"][0]
    assert expr == {
        "key": "kubernetes.io/hostname",
        "operator": "NotIn",
        "values": ["n1", "n2"],
    }
    proxy_spec = deployment(docs["proxy"])["spec"]["template"]["spec"]
    assert "nodeSelector" not in proxy_spec
    assert "nodeAffinity" in proxy_spec["affinity"]
    client = ClientPod(args, None).render()[0]
    assert "nodeAffinity" in client["spec"]["affinity"]


def test_side_channel_wiring():
    args = parse()
    docs = rendered(args, "cpu-grid")
    for index, name in enumerate(("encoder0", "encoder1")):
        env = container(docs[name])["env"]
        names = [e["name"] for e in env]
        by_name = {e["name"]: e for e in env}
        assert by_name["POD_IP"]["valueFrom"]["fieldRef"]["fieldPath"] == "status.podIP"
        assert by_name["VLLM_EC_SIDE_CHANNEL_HOST"]["value"] == "$(POD_IP)"
        assert names.index("POD_IP") < names.index("VLLM_EC_SIDE_CHANNEL_HOST")
        port = int(by_name["VLLM_EC_SIDE_CHANNEL_PORT"]["value"])
        assert port == args.side_channel_port + index
        ports = {p["name"]: p["containerPort"] for p in container(docs[name])["ports"]}
        assert ports["side-channel"] == port
    decode_env = env_of(docs["decode"])
    assert "VLLM_EC_SIDE_CHANNEL_HOST" not in decode_env
    assert "VLLM_EC_SIDE_CHANNEL_PORT" not in decode_env
    assert "side-channel" not in {p["name"] for p in container(docs["decode"])["ports"]}
    # The example connector needs no side channel at all.
    example = rendered(args, "example-data")
    assert "VLLM_EC_SIDE_CHANNEL_PORT" not in env_of(example["encoder0"])
    assert "shared_storage_path" in container(example["encoder0"])["command"][2]
    assert args.shared_storage_path.startswith(PVC_MOUNT)


def test_ports_consistent_across_container_service_and_proxy():
    args = parse()
    docs = rendered(args, "cpu-grid")
    urls = {}
    for name, doc in docs.items():
        http = next(p for p in container(doc)["ports"] if p["name"] == "http")
        svc_port = service(doc)["spec"]["ports"][0]
        assert svc_port["port"] == svc_port["targetPort"] == http["containerPort"]
        assert f"--port {http['containerPort']}" in container(doc)["command"][2]
        urls[name] = f"http://ec-bench-t1-{name}.ns1.svc:{http['containerPort']}"
    proxy_cmd = container(docs["proxy"])["command"][2]
    assert f"--encode-servers-urls {urls['encoder0']},{urls['encoder1']}" in proxy_cmd
    assert f"--decode-servers-urls {urls['decode']}" in proxy_cmd
    assert "--host 0.0.0.0" in proxy_cmd
    assert "/bench-scripts/disagg_epd_proxy.py" in proxy_cmd
    assert "--no-rewrite" not in proxy_cmd
    assert (
        "--no-rewrite"
        in container(rendered(args, "example-data")["proxy"])["command"][2]
    )
    sys_ = build_pod_system(ClientPod(args, None).target, args, "cpu-grid")
    assert sys_.front.base_url == urls["proxy"]
    assert urls["proxy"] in sys_.front.bench_serve_script(10, "inf", "/x.json", 1)


def test_pvc_is_rwx_and_mounted_everywhere():
    args = parse("--storage-class", "nfs-client-pokprod", "--pvc-size", "1Gi")
    pvc = k8s_bench.render_pvc(args)
    assert pvc["spec"]["accessModes"] == ["ReadWriteMany"]
    assert pvc["spec"]["storageClassName"] == "nfs-client-pokprod"
    assert pvc["spec"]["resources"]["requests"]["storage"] == "1Gi"
    assert "run-id" not in pvc["metadata"]["labels"]
    for arm in args.arms:
        for doc in rendered(args, arm).values():
            mounts = {m["name"]: m["mountPath"] for m in container(doc)["volumeMounts"]}
            assert mounts["bench"] == PVC_MOUNT
            volumes = {
                v["name"]: v
                for v in deployment(doc)["spec"]["template"]["spec"]["volumes"]
            }
            assert (
                volumes["bench"]["persistentVolumeClaim"]["claimName"]
                == "ec-bench-data"
            )
    client = ClientPod(args, None).render()[0]
    mounts = {
        m["name"]: m["mountPath"]
        for m in client["spec"]["containers"][0]["volumeMounts"]
    }
    assert mounts == {"bench": PVC_MOUNT, "scripts": "/bench-scripts"}
    assert args.hf_home.startswith(PVC_MOUNT)
    assert args.work_dir == f"{PVC_MOUNT}/runs/t1"
    assert args.workload_dir.startswith(PVC_MOUNT)


def test_shm_and_env_on_vllm_pods():
    args = parse("--ucx-tls", "rc,sm", "--pod-env", "UCX_NET_DEVICES=mlx5_0:1")
    docs = rendered(args, "cpu-grid")
    for name in ("encoder0", "decode"):
        env = env_of(docs[name])
        assert env["VLLM_USE_V2_MODEL_RUNNER"]["value"] == "1"
        assert env["VLLM_LOGGING_LEVEL"]["value"] == "DEBUG"
        assert env["VLLM_SERVER_DEV_MODE"]["value"] == "1"
        assert env["HF_HOME"]["value"] == f"{PVC_MOUNT}/hf"
        assert env["HF_TOKEN"]["valueFrom"]["secretKeyRef"] == {
            "name": "llm-d-hf-token",
            "key": "HF_TOKEN",
        }
        assert env["UCX_TLS"]["value"] == "rc,sm"
        assert env["UCX_NET_DEVICES"]["value"] == "mlx5_0:1"
        volumes = {
            v["name"]: v
            for v in deployment(docs[name])["spec"]["template"]["spec"]["volumes"]
        }
        # 16 GiB region * 1.25 + 8 GiB headroom.
        assert volumes["dshm"]["emptyDir"] == {"medium": "Memory", "sizeLimit": "28Gi"}
        assert container(docs[name])["resources"]["requests"]["memory"] == "60Gi"
        assert "tee /tmp/server.log" in container(docs[name])["command"][2]
    baseline = rendered(args, "baseline")["single"]
    dshm = next(
        v
        for v in deployment(baseline)["spec"]["template"]["spec"]["volumes"]
        if v["name"] == "dshm"
    )
    assert dshm["emptyDir"]["sizeLimit"] == "8Gi"
    assert "--ec-transfer-config" not in container(baseline)["command"][2]
    fixed = parse("--shm-size", "64Gi", "--pod-memory", "200Gi")
    encoder = rendered(fixed, "cpu-grid")["encoder0"]
    assert container(encoder)["resources"]["requests"]["memory"] == "200Gi"


def test_server_flags_mirror_run_bench():
    args = parse()
    docs = rendered(args, "cpu-grid")
    encoder = container(docs["encoder0"])["command"][2]
    for flag in (
        "--mm-encoder-only",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-type lru",
        "--mm-tensor-ipc torch_shm",
        "--max-num-batched-tokens 65536",
        "--enable-logging-iteration-details",
        """--limit-mm-per-prompt '{"video":0}'""",
        '"ec_role": "ec_producer"',
        '"ec_enable_nixl": true',
        '"engine_id": "ec-bench-t1-encoder0"',
    ):
        assert flag in encoder, flag
    decode = container(docs["decode"])["command"][2]
    assert "--enable-mm-embeds" in decode
    assert '"ec_role": "ec_consumer"' in decode
    assert "--mm-encoder-only" not in decode
    assert decode.startswith("exec python -m vllm.entrypoints.cli.main serve ")


def test_scripts_configmap_and_proxy_from_image():
    args = parse()
    cm = k8s_bench.render_scripts_configmap(args)
    assert cm["metadata"]["name"] == "ec-bench-t1-scripts"
    for name in ("run_bench.py", "gen_workload.py", "ec_log_stats.py"):
        assert name in cm["data"]
    assert "disagg_epd_proxy.py" in cm["data"]
    assert "def " in cm["data"]["disagg_epd_proxy.py"]
    proxy = rendered(args, "cpu-grid")["proxy"]
    mounts = {m["name"]: m["mountPath"] for m in container(proxy)["volumeMounts"]}
    assert mounts["scripts"] == "/bench-scripts"
    from_image = parse("--proxy-from-image")
    cmd = container(rendered(from_image, "cpu-grid")["proxy"])["command"][2]
    assert "/vllm-workspace/examples/disaggregated/" in cmd


def test_run_id_and_size_validation():
    import pytest

    with pytest.raises(SystemExit):
        parse("--run-id", "Bad_ID")
    with pytest.raises(SystemExit):
        parse("--pvc-size", "500")
    with pytest.raises(SystemExit):
        parse("--pod-env", "NOEQUALS")


def test_dry_run_writes_manifests(tmp_path):
    args = parse("--dry-run", "--out-dir", str(tmp_path))
    assert k8s_bench.dry_run(args, ClientPod(args, None)) == 0
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.yaml"))
    assert "manifests/t1/pvc.yaml" in files
    assert "manifests/t1/client-pod.yaml" in files
    assert "manifests/t1/scripts-configmap.yaml" in files
    assert "manifests/t1/cpu-grid/encoder1.yaml" in files
    assert "manifests/t1/baseline/single.yaml" in files
    for path in tmp_path.rglob("*.yaml"):
        assert list(yaml.safe_load_all(path.read_text()))


def test_placement_gate():
    record = {
        "arm": "cpu-grid",
        "same_node_requested": False,
        "gpu_nodes_distinct": False,
        "pods": [{"role": "encoder0", "node": "a"}, {"role": "decode", "node": "a"}],
    }
    import pytest

    with pytest.raises(k8s_bench.ServerMismatchError):
        k8s_bench.check_placement(record)
    k8s_bench.check_placement({**record, "same_node_requested": True})
    restarted = {**record, "gpu_nodes_distinct": True}
    restarted["pods"] = [{"role": "decode", "node": "b", "restarts": 1}]
    with pytest.raises(k8s_bench.ServerMismatchError):
        k8s_bench.check_placement(restarted)


if __name__ == "__main__":
    import tempfile
    import types

    try:
        import pytest  # noqa: F401
    except ImportError:
        # Enough of pytest for the checks above to run without it.
        class _Raises:
            def __init__(self, exc):
                self.exc = exc

            def __enter__(self):
                return self

            def __exit__(self, kind, *_):
                if kind is None:
                    raise AssertionError(f"{self.exc.__name__} not raised")
                return issubclass(kind, self.exc)

        sys.modules["pytest"] = types.SimpleNamespace(raises=_Raises)  # type: ignore

    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
            print(f"ok   {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failed else 0)
