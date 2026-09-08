# EC connector benchmark on OpenShift (multi-node)

`k8s_bench.py` runs the arms of [`run_bench.py`](../README.md) with the
encoder(s) and the decode instance on **different nodes**. It reuses
`run_bench.py` for everything that is not placement: the server flags, the
load generator, the log accounting, the gates and the table. The driver runs
on your laptop with `oc`; the servers are Deployments, the load generator is
a GPU-less pod on the same image.

| arm | what crosses the node boundary |
|---|---|
| `cpu-data`, `cpu-grid` | `ECCPUConnector` over NIXL; the consumer dials the producer's pod IP |
| `example-data`, `example-grid` | `ECExampleConnector` over a ReadWriteMany PVC |
| `baseline`, `offload` | nothing: one pod (the `x_base` reference) |

## Prerequisites

- `oc` logged in (`oc whoami`) with `pods/exec` in the namespace.
- A secret holding `HF_TOKEN` (default name `llm-d-hf-token`, key
  `HF_TOKEN`; `--hf-secret`/`--hf-secret-key` to change).
- A ReadWriteMany-capable storage class. Default
  `ibm-spectrum-scale-fileset`; `nfs-client-pokprod` is the other RWX option
  on that cluster (`--storage-class`). The PVC (`--pvc-name`, default
  `ec-bench-data`, `--pvc-size` 500Gi) is created once and reused: it holds
  the HF cache, the workload, per-run scratch and the example connector's
  storage, mounted at `/bench` in every pod.
- As many GPU nodes as GPU pods: with anti-affinity on, `N` encoders and
  one decode need `N+1` nodes. `--same-node` turns it off (a control run).
- The image must carry `nixl` for the CPU arms. The release
  `vllm/vllm-openai` images are built with `INSTALL_KV_CONNECTORS=true`; the
  preflight checks `import nixl` in the client pod. Prefer pinning a digest
  over `nightly`, so both nodes run the same build (the placement block
  records each pod's `imageID`).
- `runAsUser: 0` is set (the `anyuid` SCC); `--no-run-as-root` drops it and
  relies on the namespace's fsGroup for the PVC.
- Pod-to-pod traffic on the side-channel port (5577+i), the vLLM ports and
  UCX's dynamic ports must be allowed by any NetworkPolicy.

## Running

```bash
cd scripts/cpu_ec_connector/bench

# 1. Always dry-run first: renders every manifest under
#    <out-dir>/manifests/<run-id>/ and prints every container command and
#    the load command. Needs no cluster and no oc.
python k8s_bench.py --namespace my-ns --dry-run \
    --arms baseline,cpu-grid --out-dir results

# 2. NIXL (CPU connector) arms, 2 encoders x 1 GPU, decode on 1 GPU, workload
#    generated in-cluster on the first run (it is kept on the PVC).
python k8s_bench.py --namespace my-ns --out-dir results \
    --arms baseline,cpu-data,cpu-grid --num-encoders 2 \
    --encoder-gpus 1 --decode-gpus 1 --max-concurrency 1,4,8 \
    --num-prompts 120 --restart-per-load-point \
    --gen-workload "--pool-size 96 --buckets 2048x2048:1.0 --num-requests 400 --reuse zipf:1.1"

# 3. Example connector arms over the shared filesystem.
python k8s_bench.py --namespace my-ns --out-dir results \
    --arms baseline,example-data,example-grid --max-concurrency 1,4,8

# RDMA instead of TCP for UCX: clear UCX_TLS and name the device.
python k8s_bench.py ... --ucx-tls "" --pod-env UCX_NET_DEVICES=mlx5_0:1
```

`--serve-args`, `--encoder-serve-args`, `--decode-serve-args`, `--model`,
`--request-rates`, `--max-concurrency`, `--num-prompts`,
`--restart-per-load-point`, `--ec-cpu-bytes` and the rest of `run_bench.py`'s
arm and load options apply unchanged. `--encoder-gpus`/`--decode-gpus` set
both `nvidia.com/gpu` and `--tensor-parallel-size`.

## Checking placement

While an arm runs:

```bash
oc get pod -n my-ns -l app=vllm-ec-bench -o wide
```

After the run, `results/bench.json` carries a `placement` block per arm with
each pod's node, IP, restart count and image digest, and the driver prints
`[k8s] placement` lines; an arm whose GPU pods shared a node (without
`--same-node`) or whose pod restarted fails with `ServerMismatchError`.
Result entries for the EPD arms have `topology: multinode` (`same-node`
under `--same-node`).

Each server's log is saved to `results/logs/<run-id>/<arm>-<role>-<n>.log`
before its pod is deleted.

## Cleaning up

Servers are torn down after every arm, the client pod and ConfigMap at the
end (`--keep` leaves them). To sweep leftovers from any run, or one run:

```bash
python k8s_bench.py --namespace my-ns --cleanup
python k8s_bench.py --namespace my-ns --cleanup --cleanup-run-id --run-id 20260908-1200
python k8s_bench.py --namespace my-ns --cleanup --delete-pvc   # also the PVC
```

## Design notes

- **Side channel = pod IP.** The producer's ZMQ ROUTER *binds*
  `VLLM_EC_SIDE_CHANNEL_HOST`, so it has to be an address the pod owns; a
  Service name is not. The manifests set `POD_IP` from the downward API and
  `VLLM_EC_SIDE_CHANNEL_HOST=$(POD_IP)`; the consumer needs nothing, it
  dials what `ec_transfer_params` announced. NIXL/UCX likewise connect pod to
  pod.
- **Proxy from the checkout.** The EPD proxy travels in the scripts
  ConfigMap (`/bench-scripts/disagg_epd_proxy.py`), because the `Rewrote N`
  and `STAGE` lines the accounting reads belong to this checkout's proxy.
  `--proxy-from-image` runs the image's copy instead.
- **Logs stay in the pod.** The container command is
  `exec python -m vllm... > >(tee /tmp/server.log) 2>&1`, so `oc exec`
  reads the same file `run_bench` would on one host; `oc logs` shows the
  same text.
- **PVC layout.** `/bench/hf` (HF_HOME), `/bench/wl` (workload),
  `/bench/runs/<run-id>/` (client outputs, `queue.csv`, `shared/` for the
  example connector).

## Not yet verified on a cluster

Everything here has been exercised only through `--dry-run`, the render
checks in `test_k8s_bench.py` and `run_bench.py`'s single-node dry-run. A
first cluster run must confirm:

- `vllm bench serve` and `gen_workload.py` run in the GPU-less client pod
  on the official image (platform detection without a GPU).
- `import nixl` in the chosen image, and UCX picking a transport that
  crosses nodes with `UCX_TLS=tcp,sm` (or the RDMA override).
- `oc exec deployment/<name>` and `oc rollout status` behave as assumed,
  and the pod's own Service name resolves from inside it (`reset_caches`
  and the health probe use it).
- `$(POD_IP)` expands in the container env (POD_IP is listed first).
- `/dev/shm` default (1.25x the EC region + 8 GiB) and the memory request
  (shm + 32 GiB) fit the nodes; `--shm-size`/`--pod-memory` override.
- The HF cache on GPFS/NFS: servers start one at a time so the first
  downloads the model; the 1800 s startup timeout covers it.
- `runAsUser: 0` is permitted, and the PVC is writable.
- The proxy pod's `--host 0.0.0.0` bind and its Service are reachable from
  the client pod; the `Uvicorn running` line appears in its log.
- Deleting a Deployment SIGTERMs vLLM as PID 1 cleanly within the 60 s
  grace period.
