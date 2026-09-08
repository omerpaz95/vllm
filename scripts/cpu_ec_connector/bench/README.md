# EC connector benchmark

Measures the encoder-cache (EC) connectors on a multimodal serving workload:
whether offloading or transferring encoder outputs beats recomputing them,
and how the CPU/NIXL transport compares to the reference shared-filesystem
one. Everything runs against `main`; no patches to vLLM are needed.

## Arms

| arm | topology | connector | decode instance receives |
|---|---|---|---|
| `baseline` | one instance | none | n/a, encodes and decodes itself |
| `offload` | one instance | `ECCPUConnector`, `ec_both` | n/a, repeats reload from the CPU region |
| `cpu-data` | encoder + decode | `ECCPUConnector` over NIXL | pixels + `ec_transfer_params` |
| `cpu-grid` | encoder + decode | `ECCPUConnector` over NIXL | `image_grid_thw` + `ec_transfer_params` |
| `example-data` | encoder + decode | `ECExampleConnector` | pixels + `ec_transfer_params` |
| `example-grid` | encoder + decode | `ECExampleConnector` | grid + `ec_transfer_params` |

`data` vs `grid` isolates what the grid substitution (PR #50390) saves with
the transport held fixed; `cpu` vs `example` compares the transports. The
`data` arms are the EPD proxy's `--no-rewrite` mode: the proxy attaches the
connector's handles to the decode body whether or not it rewrites, so the
CPU connector's consumer can locate the producer's entry either way.

`x_base` compares whatever GPU sets `--gpu`, `--encoder-devices` and
`--decode-gpu` gave each arm, so a baseline on one GPU against an EPD pair
on two credits disaggregation with the extra hardware; give the baseline the
same GPU list as the decode instance for a resource-matched comparison.

## Files

| file | role |
|---|---|
| `gen_workload.py` | builds the image pool, the `custom_image` JSONL and `manifest.json` |
| `run_bench.py` | server lifecycle, load driving, log accounting, gates, table |
| `ec_log_stats.py` | log parsers: EC transfers (both connectors), encoder inputs, proxy `STAGE` lines, rewrite counts |
| `phase0_hit_check.py` | correctness gate for one `ec_both` instance: a repeat pass must reload, not recompute |
| `micro_swap_blocks.py` | descriptor-layout microbenchmark of the region's batched copies |
| `patches/sitecustomize.py` | descriptor-count instrumentation for `--frag`; wraps `_coalesce_runs` without touching the connector. Not yet exercised. |

## Running

```bash
# 1. Workload: 96 images at 2048x2048, 400 requests, zipf reuse. The default
#    --photo-source is `synth`: generated fractal noise, no download, each
#    image's JPEG calibrated to 0.17 MB/MP (what a real-photo pool measured
#    at q85), about 4 s per image. `dir:<path>` uses a directory of photos
#    (searched recursively; add --allow-upscale for sources smaller than the
#    bucket) and `hf-tar:<repo>[:<file>]` streams a tar.gz from Hugging Face.
python gen_workload.py --out-dir /data/wl --pool-size 96 \
    --buckets 2048x2048:1.0 --num-requests 400 --reuse zipf:1.1 --self-check

#    The same pool from real LSDIR photos (gated on Hugging Face, academic
#    research licence; needs a token whose account accepted the terms, and
#    `env -u HF_TOKEN` if a stored login should win over the environment):
python gen_workload.py --photo-source hf-tar:ofsoundof/LSDIR:shard-00.tar.gz \
    --allow-upscale --out-dir /data/wl --pool-size 96 \
    --buckets 2048x2048:1.0 --num-requests 400 --reuse zipf:1.1 --self-check

# 2. Always dry-run first: it prints every launch command and the load
#    command, and has caught port collisions before they cost a run.
python run_bench.py --workload-dir /data/wl --out-dir results --dry-run

# 3. Sweep concurrency across all six arms, fresh servers per load point.
python run_bench.py --workload-dir /data/wl --out-dir results \
    --max-concurrency 1,4,8 --num-prompts 120 --restart-per-load-point

# A larger model: GPU lists set the tensor-parallel size of every instance.
# Here the baseline and the decode instance get TP=4 and two TP=2 encoders
# share GPUs 0-3; --serve-args / --encoder-serve-args / --decode-serve-args
# pass anything else through to `vllm serve`.
python run_bench.py --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
    --gpu 0,1,2,3 --encoder-devices "0,1;2,3" --decode-gpu 4,5,6,7 \
    --workload-dir /data/wl --out-dir results --max-concurrency 1,4,8

# Inside a pod (oc): the servers and the load generator run there.
python run_bench.py --pod my-pod --python /venv/bin/python \
    --vllm-repo /workspace/vllm --work-dir /workspace/ec_bench \
    --workload-dir /workspace/wl --out-dir results --arms cpu-grid,example-grid
```

`--frag` runs the `offload` arm with a region smaller than the working set
and descriptor counting on, to see whether entries still collapse to one
descriptor after the region has churned. `--patch-dir` must point at
`patches/` on the target.

The archived real-photo results used `hf-tar:ofsoundof/LSDIR:shard-00.tar.gz`,
which is gated and licensed for academic research only. Every measured
quantity is pixel-count driven, and an EC entry's size is purely resolution
(5,329 embeddings x 7,168 B = 38.2 MB at 2048x2048), so what a source has to
get right is the JPEG size that sets wire payload and decode time; the
synthetic source is calibrated to that, and an enlarged photo compresses the
same as a native crop.

## What is gated

A run that fails any of these raises instead of printing a delta, because the
timings would then measure something other than the arm's name:

- every request completed (`completed == num_prompts`)
- each server's fresh log carries a startup line, and for the CPU connector
  the region creation line tagged with this run's engine id (a survivor
  from the previous arm would otherwise answer `/health`)
- connector arms loaded entries on the consumer; the baseline loaded none
- `grid` arms rewrote at least `--min-rewrite-coverage` of the image
  references; `data` arms rewrote nothing
- with several encoders, each computed something
- the encoder's image transform ran on the requested `--mm-processor-device`
- after the run, every connector arm's consumer computed fewer encoder inputs
  than the baseline at the same load point

## Configuration that matters

| setting | why |
|---|---|
| `--encoder-max-num-batched-tokens 65536` | 8192 against 5,350 tokens/image forbids batching two image requests and pins the encoder at one image per step; every `rate=inf` number is then queue depth |
| `--max-concurrency 1,4,8` | never `--request-rate inf` unbounded: TTFT becomes queue depth divided by service rate |
| `--enable-mm-embeds` on decode | otherwise every grid request is HTTP 400; set in every EPD arm so it is not a confound |
| `--mm-processor-cache-type lru` on encoders | with `shm` the engine keeps no receiver cache, the connector reports no grid, and the grid arm silently becomes the data arm |
| `--limit-mm-per-prompt '{"video":0}'` | drops the encoder-cache floor from 32768 to 16384 embeddings |
| `--enable-logging-iteration-details` + `VLLM_LOGGING_LEVEL=DEBUG` | the accounting this reads lives on those lines |
| `--mm-encoder-only` + `--enforce-eager` on encoders | ~16 GB to ~1.4 GB, and the EPD example requires eager encoders |
| `VLLM_USE_V2_MODEL_RUNNER=1` | required by the CPU connector; set on every arm |
| `ec_enable_nixl` inside `ec_connector_extra_config` | `ECTransferConfig` rejects it as a top-level key |

Queue depth is sampled from every instance's `/metrics` once a second and
reported per load point (`encQmax`, `decQmax`). If `waiting` tracks the
offered concurrency, that point measures the queue, not the work.

## Caveats to state with any result

- **The EC region is never reset by a route.** Without
  `--restart-per-load-point`, the first load point pays the saves and later
  points run against a warm region; the `saves` column shows which regime a
  point was in. Arm-to-arm comparison at one point is still fair.
- **The warmup uses held-out images** (`warmup.jsonl`, built from
  `--warmup-images` extra pool slots that no measured request references),
  so first-request costs are paid without seeding any cache the measurement
  hits. A workload directory without `warmup.jsonl` predates this and must
  be rebuilt.
- **Region larger vs smaller than the working set** answer different
  questions (all hits vs continuous eviction). `--ec-cpu-bytes` defaults to
  the manifest's 1.25x working set; the frag arm uses 0.5x.
- **A multi-image request must fit the consumer's encoder cache**, or it is
  chunked across steps. The manifest's `max_embeds_per_request` is the floor.
- **Log-text parsing depends on unpromised strings.** Every parser is paired
  with a gate that fails when it matched nothing.

## Prior results

Grid vs pixels with `ECExampleConnector` at 2048x2048, two replications,
queue-verified: throughput 1.23-1.28x at c=1 decaying to ~1.02x at c=32 as
the decode GPU saturates, reproducing PR #50390's published 1.18-1.31x. The
CPU-connector comparison was run on an older proxy whose `--no-rewrite`
dropped the transfer params, so the CPU `data` arm transferred nothing there;
rerun it before quoting. Nothing has been measured with more than one
encoder, and `--frag` has not been exercised.

## Traps

- **Verify the pod runs the code you think**, by content rather than SHA:
  `grep get_from_extra_config vllm/distributed/ec_transfer/ec_connector/cpu/scheduler/__init__.py`.
- **`pkill -f <pattern>` matches its own shell.** The teardown here brackets
  the pattern and refuses to signal its own process group.
- **Zombies count as alive to `pgrep`**, which makes every teardown look
  hung and escalate to SIGKILL; SIGKILL is how multi-GiB `/dev/shm` files
  leak. The liveness check skips `Z` state.
- **Kill the process group.** Signalling the API server alone leaves
  EngineCore holding the GPU and the port, which presents as the next arm's
  server never becoming healthy.
- **`subprocess.run(capture_output=True)` never returns for a launched
  daemon.** Fire-and-forget launches use DEVNULL.
- **`HF_TOKEN` in the environment overrides a stored `hf auth login`**; run
  `gen_workload.py` with `env -u HF_TOKEN` to use the stored one.
- **LSDIR's `val1.tar.gz` contains downscaled copies beside the HR files**;
  train shards are flat HR.
