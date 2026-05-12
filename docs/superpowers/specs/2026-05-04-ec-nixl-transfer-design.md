# EC Connector — NIXL transfer from E node to PD node

- **Branch:** `ec_connector_nixl_transfer` (based on `cpu_ec_connector`)
- **Scope:** add the transport mechanism that moves encoder outputs from a remote Encoder (E) node into the consumer (PD) node's CPU encoder-cache region, using NIXL for the data plane and ZMQ for the control plane.
- **Out of scope (future PRs):**
    - Producer worker-side `save_caches` (GPU→CPU staging of encoder output into the producer's in-memory dict). Tests in this PR populate that dict directly.
    - Plumbing `encoder_node_address` onto the per-request metadata (will ship in a separate PR via `kv_transfer_params`). This PR uses a temporary connector-extra-config fallback.
    - Block eviction / free-on-request-finished for the consumer's CPU encoder cache.

## Background

The `cpu_ec_connector` branch introduced:

- `ECSharedRegion` — flat mmap-backed `(num_blocks, block_size_bytes)` CPU region at `/dev/shm/vllm_ec_{engine_id}.mmap`, pinned (by rank 0) for fast GPU DMA. All workers mmap the same file and see the same bytes.
- `ECCPUConnector` (`vllm/distributed/ec_transfer/ec_connector/cpu_connector.py`) — consumer-side CPU→GPU loader (`start_load_caches`). Scheduler-side methods and `save_caches` are `NotImplementedError`.
- `ECConnectorBase` (`vllm/distributed/ec_transfer/ec_connector/base.py`) — abstract base shared across EC connectors.

This PR fills in the scheduler-side methods and adds remote transport between two `ECCPUConnector` instances (one `ec_producer`, one `ec_consumer`).

## Goals

1. Consumer scheduler can request an encoding it doesn't have locally by `mm_hash` over a side channel.
2. Producer scheduler fulfils that request by reading its locally-held encoding tensor and pushing it, via NIXL, into the consumer's CPU mmap at a consumer-chosen set of block indices.
3. Consumer's existing `start_load_caches` then pushes CPU→GPU, unchanged.

## Non-goals

- No reliability/retry layer beyond a single failure bool.
- No producer-side caching/eviction policy.
- No TP-size matching between E and PD. The transfer protocol is TP-agnostic.
- No migration of NIXL data through the worker processes — all NIXL calls happen in scheduler processes.

## Architecture

```text
┌────────── Consumer (PD) node ─────────┐       ┌────────── Producer (E) node ──────────┐
│ Scheduler process                     │       │ Scheduler process                     │
│  ECCPUConnector (is_consumer=True)    │  ZMQ  │  ECCPUConnector (is_producer=True)    │
│   ├ _region: ECSharedRegion           │ ◄───► │                                       │
│   │    (mmap + free-list;             │       │   ├ _nixl: NixlWrapper                │
│   │     region.alloc/free called      │       │   ├ _encodings: dict[mm_hash, Tensor] │
│   │     by scheduler only)            │       │   │   (future PR fills; this PR       │
│   ├ _nixl: NixlWrapper                │       │   │    uses test fixtures)            │
│   ├ _encoding_map: mm_hash→block_ids  │       │                                       │
│   ├ _ready: set of arrived mm_hashes  │       │                                       │
│   └ _dealers: one DEALER per E node   │       │   ├ _waiting: parked xfer requests    │
│                                       │ NIXL  │   ├ _router: ROUTER listener          │
│                                       │ ◄──── │   ├ _listener_thread                  │
│                                       │       │   └ _stop_event                       │
│                                       │       │                                       │
│ Worker process(es) (unchanged)        │       │ Worker process(es)                    │
│  _region + start_load_caches (CPU→GPU)│       │  (no code in this PR)                 │
└───────────────────────────────────────┘       └───────────────────────────────────────┘
```

Key architectural choices:

- **NIXL lives in the scheduler process.** Workers never touch NIXL. The scheduler opens the same `/dev/shm` mmap workers use (the creator/joiner race in `ECSharedRegion` already handles this) and registers that region once with NIXL at startup.
- **Producer uses a plain `dict[str, torch.Tensor]` for source data** — not a second `ECSharedRegion`. NIXL memory registration happens per-transfer against the source tensor and is released at transfer end. Simpler producer state; a future PR can optimise if the per-transfer registration cost matters.
- **Control plane is scheduler ↔ scheduler over ZMQ** using DEALER (consumer) ↔ ROUTER (producer). One ROUTER listener thread on producer side; no dedicated thread on consumer side.
- **Single failure bool in acks.** No NACK types, no error codes, no retry.

## Data flow (one end-to-end transfer)

1. Engine asks consumer scheduler `has_cache_item(mm_hash)` → returns `False` (not seen before).
2. Engine calls `update_state_after_alloc(request, index)`. Consumer scheduler:
   - Resolves the E node address for this `mm_hash` (temporary MVP: fallback to `ec_connector_extra_config["default_encoder_node"] = "host:port"`; future PR reads it from `kv_transfer_params`).
   - Allocates N blocks via `self._region.alloc(n)` where `N = request.mm_features[index].mm_position.length`.
   - Gets or creates a DEALER to that E node.
   - Sends `XferReq(mm_hash, dst_block_indices, consumer_agent_name, consumer_nixl_metadata, consumer_mem_descriptor)` via the DEALER.
   - Records `mm_hash → dst_block_indices` in `_encoding_map`.
3. Subsequent `has_cache_item(mm_hash)` calls return `None` (in-flight) until the ack arrives.
4. Producer listener thread receives the `XferReq`, decodes it. If `mm_hash` is in `_encodings`, calls `_do_nixl_xfer(req, tensor)` inline; otherwise parks `(identity, req)` in `_waiting[mm_hash]`.
5. `_do_nixl_xfer` performs the full NIXL sequence (detail in §NIXL call sequence), returns `ok: bool`, sends `XferAck(mm_hash, ok)` back via the ROUTER using the saved DEALER identity.
6. Consumer's next `has_cache_item` or `build_connector_meta` call runs `_drain_acks()` — non-blocking `recv` on each DEALER. On `ok=True` ack, adds `mm_hash` to `_ready`. On `ok=False`, pops the entry from `_encoding_map` and returns its blocks to the allocator.
7. Next `build_connector_meta(scheduler_output)` returns `ECCPUConnectorMetadata(mm_hash_to_cpu_blocks={h: _encoding_map[h] for h in _ready})`, then pops those entries from both `_encoding_map` and `_ready`. Consumer worker's existing `start_load_caches` does the CPU→GPU copy; its `if mm_hash in encoder_cache: continue` provides belt-and-braces dedup.

## ZMQ protocol

```python
# vllm/distributed/ec_transfer/ec_connector/messages.py (new file)

EC_CONNECTOR_VERSION: int = 1

class XferReq(msgspec.Struct, tag="req"):
    mm_hash: str
    dst_block_indices: list[int]
    consumer_agent_name: str                     # NixlWrapper's uuid
    consumer_nixl_metadata: bytes                # from wrapper.get_agent_metadata()
    consumer_mem_descriptor: bytes               # serialized xfer_descs for consumer mmap
    compatibility_hash: str                      # first-message compat check
    connector_version: int = EC_CONNECTOR_VERSION

class XferAck(msgspec.Struct, tag="ack"):
    mm_hash: str
    ok: bool
```

Tagged-union encoding (msgspec's `tag=` field) allows decoders to pick the right struct without trial-and-error. Each side only expects one direction, so the decoder can use `msgspec.msgpack.Decoder(XferReq)` or `Decoder(XferAck)` directly.

### Sockets

- **Producer:** single ROUTER bound to `(VLLM_EC_SIDE_CHANNEL_HOST, VLLM_EC_SIDE_CHANNEL_PORT)`, created via `make_zmq_socket(ctx, path, zmq.ROUTER, bind=True)`. Single listener thread.
- **Consumer:** one DEALER per distinct E node, keyed by `(host, port)`, lazy-created via `make_zmq_socket(ctx, path, zmq.DEALER, bind=False)`. No dedicated thread; drained on-demand in `has_cache_item` and `build_connector_meta` via `recv(flags=zmq.NOBLOCK)`.

Note: `make_zmq_socket` (from `vllm.utils.network_utils`) is used directly here. The existing `zmq_ctx` helper at `vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py` rejects DEALER (it whitelists only ROUTER/REQ), so cannot be reused for our consumer side. The producer ROUTER side could use `zmq_ctx`, but using `make_zmq_socket` on both sides keeps the call sites symmetric.

### Failure handling

- **NIXL xfer fails on producer** → `XferAck(ok=False)`. Consumer frees blocks; `has_cache_item` returns `False`; engine falls back.
- **`mm_hash` absent from `_encodings` when XferReq arrives** → park in `_waiting`. Future `save_caches` (out of scope for this PR) drains waiting entries. Tests drive this path via a direct test hook.
- **Malformed ZMQ message** → log warning, drop, no ack. Consumer sees no state change.
- **Duplicate `XferReq` for an `mm_hash` already in `_waiting`** → overwrite (latest wins).
- **No ack ever arrives** → `_encoding_map` holds the entry indefinitely (never moves to `_ready`). Accepted MVP limitation; the engine's request-level timeout will surface the stall. A sweeper thread is a possible future addition.

### Compatibility hash

First `XferReq` from a consumer to a producer includes a SHA-256 `compatibility_hash` (mirrors `compute_nixl_compatibility_hash` in the KV NIXL metadata module but with a smaller factor set relevant to EC: `vllm_version`, `EC_CONNECTOR_VERSION`, `model_config.model`, `model_config.dtype`, `block_size_bytes`). Producer computes the same hash locally; on mismatch, `XferAck(ok=False)` and log the mismatch. This prevents silent misinterpretation when peers are built against different codebases.

## NIXL call sequence (verified against existing KV connector)

### Setup (producer `__init__`, scheduler role)

```python
from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

config = nixl_agent_config(num_threads=1, capture_telemetry=False)
self._nixl = NixlWrapper(str(uuid.uuid4()), config)
self._mem_type = current_platform.get_nixl_memory_type()   # "DRAM" for CPU
self._device_id = 0                                        # CPU
```

### Setup (consumer `__init__`, scheduler role)

```python
self._nixl = NixlWrapper(str(uuid.uuid4()), config)
# Register entire mmap region once, for the lifetime of the connector.
reg_tuples = [(region.base_ptr, region.total_size_bytes, 0, "")]
self._reg_descs = self._nixl.get_reg_descs(reg_tuples, "DRAM")
self._nixl.register_memory(self._reg_descs)

# Build xfer_descs for every block in the mmap (one per slot).
xfer_tuples = [
    (region.base_ptr + i * block_size_bytes, block_size_bytes, 0)
    for i in range(num_ec_blocks)
]
self._xfer_descs = self._nixl.get_xfer_descs(xfer_tuples, "DRAM")

# Serialize xfer_descs for inclusion in XferReq messages.
self._mem_descriptor_bytes = self._serialize_xfer_descs(self._xfer_descs)
self._agent_metadata = self._nixl.get_agent_metadata()
```

### Per-transfer (producer, inside `_do_nixl_xfer(req, tensor)`)

```python
n = len(req.dst_block_indices)
if tensor.nbytes != n * self._block_size_bytes:
    return False

# 1. Register source tensor with NIXL (transient).
reg_tuples = [(tensor.data_ptr(), tensor.nbytes, self._device_id, "")]
reg_descs = self._nixl.get_reg_descs(reg_tuples, self._mem_type)
self._nixl.register_memory(reg_descs)

# 2. Learn the consumer's agent.
remote_agent_name = self._nixl.add_remote_agent(req.consumer_nixl_metadata)

# 3. Prepare local xfer dlist (n block-sized slices of the source tensor).
local_tuples = [
    (tensor.data_ptr() + i * self._block_size_bytes,
     self._block_size_bytes, self._device_id)
    for i in range(n)
]
local_descs = self._nixl.get_xfer_descs(local_tuples, self._mem_type)
local_dlist = self._nixl.prep_xfer_dlist("NIXL_INIT_AGENT", local_descs)

# 4. Rehydrate the consumer's xfer_descs and prep remote dlist.
remote_descs = self._deserialize_xfer_descs(req.consumer_mem_descriptor)
remote_dlist = self._nixl.prep_xfer_dlist(remote_agent_name, remote_descs)

# 5. Build + start WRITE xfer.
xfer_handle = self._nixl.make_prepped_xfer(
    "WRITE",
    local_dlist,  list(range(n)),
    remote_dlist, req.dst_block_indices,
    notif_msg=b"",                        # ack is ZMQ, not NIXL notif
)
self._nixl.transfer(xfer_handle)

# 6. Poll state until done.
ok = False
try:
    while True:
        state = self._nixl.check_xfer_state(xfer_handle)
        if state == "DONE":
            ok = True; break
        if state != "PROC":
            break                          # error
        time.sleep(0.0005)
finally:
    self._nixl.release_xfer_handle(xfer_handle)
    self._nixl.release_dlist_handle(local_dlist)
    self._nixl.release_dlist_handle(remote_dlist)
    self._nixl.remove_remote_agent(remote_agent_name)
    self._nixl.deregister_memory(reg_descs)

return ok
```

## Scheduler-side logic

### Consumer

```python
# _encoding_map: dict[str, list[int]]  — mm_hash -> block indices (populated at alloc time)
# _ready:        set[str]               — mm_hashes whose xfer completed, not yet handed off

# has_cache_item
def has_cache_item(self, identifier: str) -> bool | None:
    self._drain_acks()
    if identifier in self._ready:
        return True
    if identifier in self._encoding_map:
        return None                                      # allocated, xfer still in flight
    return False

# update_state_after_alloc
def update_state_after_alloc(self, request: Request, index: int) -> None:
    feature = request.mm_features[index]
    mm_hash = feature.mm_hash or feature.identifier     # mm_hash is optional
    if mm_hash in self._encoding_map:
        return
    n = feature.mm_position.length
    indices = self._region.alloc(n)
    self._encoding_map[mm_hash] = indices               # single table, populated now

    addr = self._resolve_encoder_address(request, feature)
    dealer = self._get_or_create_dealer(addr)
    req = XferReq(
        mm_hash=mm_hash,
        dst_block_indices=indices,
        consumer_agent_name=self._nixl_agent_name,
        consumer_nixl_metadata=self._agent_metadata,
        consumer_mem_descriptor=self._mem_descriptor_bytes,
        compatibility_hash=self._compat_hash,
    )
    dealer.send(msgspec.msgpack.encode(req))

# _complete is called by _drain_acks on each received ack
def _complete(self, mm_hash: str, ok: bool) -> None:
    if mm_hash not in self._encoding_map:
        return                                          # stale / duplicate ack
    if ok:
        self._ready.add(mm_hash)
    else:
        indices = self._encoding_map.pop(mm_hash)
        self._region.free(indices)

# build_connector_meta
def build_connector_meta(self, scheduler_output: SchedulerOutput) -> ECCPUConnectorMetadata:
    self._drain_acks()
    mapping = {h: self._encoding_map.pop(h) for h in self._ready}
    self._ready.clear()
    return ECCPUConnectorMetadata(mm_hash_to_cpu_blocks=mapping)
```

`_resolve_encoder_address(request, feature)` returns the E node `(host, port)`. MVP implementation:

1. If `kv_transfer_params` on the request has an `encoder_node_address` entry, use that (future PR will set this).
2. Otherwise, read `ec_connector_extra_config["default_encoder_node"]` (format `"host:port"`). Required in MVP.
3. If neither is present, raise a clear error at startup.

### Producer

```python
# listener thread
def _run_listener(self) -> None:
    self._router.setsockopt(zmq.RCVTIMEO, 1000)
    while not self._stop_event.is_set():
        try:
            identity, _, payload = self._router.recv_multipart()
        except zmq.Again:
            continue
        try:
            req = msgspec.msgpack.decode(payload, type=XferReq)
        except (msgspec.DecodeError, msgspec.ValidationError):
            logger.warning("ec: dropped malformed XferReq")
            continue

        if req.compatibility_hash != self._compat_hash:
            self._send_ack(identity, req.mm_hash, ok=False)
            continue

        tensor = self._encodings.get(req.mm_hash)
        if tensor is None:
            self._waiting[req.mm_hash] = (identity, req)
            continue

        ok = self._do_nixl_xfer(req, tensor)
        self._send_ack(identity, req.mm_hash, ok)

def _drain_waiting(self, mm_hash: str) -> None:
    entry = self._waiting.pop(mm_hash, None)
    if entry is None: return
    identity, req = entry
    tensor = self._encodings.get(mm_hash)
    if tensor is None: return
    ok = self._do_nixl_xfer(req, tensor)
    self._send_ack(identity, mm_hash, ok)
```

### Block allocation lives on `ECSharedRegion`

The free-list for the mmap's block slots is member state of `ECSharedRegion` itself — the allocator's state belongs with the memory it tracks. No separate allocator class; two methods added to the existing region:

```python
# vllm/distributed/ec_transfer/ec_connector/ec_shared_region.py  (modified)

class ECSharedRegion:
    def __init__(self, instance_id, num_blocks, block_size_bytes):
        ...                                          # existing mmap setup unchanged
        self._free_blocks: list[int] = list(range(num_blocks))

    def alloc(self, n: int) -> list[int]:
        if len(self._free_blocks) < n:
            raise RuntimeError(
                f"ECSharedRegion exhausted: need {n} blocks, {len(self._free_blocks)} free"
            )
        out, self._free_blocks = self._free_blocks[:n], self._free_blocks[n:]
        return out

    def free(self, indices: list[int]) -> None:
        self._free_blocks.extend(indices)
```

Notes:

- `alloc(n)` returns `list[int]` (block indices), not a tensor view. Indices are what every downstream consumer of an allocation needs: they travel over ZMQ in `XferReq.dst_block_indices`, flow to workers via `ECCPUConnectorMetadata`, drive NIXL xfer-desc offset math, and get passed back to `free()`. None of these places can use a Python tensor handle. When bytes are needed directly (tests, debugging), `region.blocks[indices]` already returns a `(n, block_size_bytes)` view — no extra API required.
- `_free_blocks` is process-local Python state, not shared via the mmap. Workers that instantiate `ECSharedRegion` get their own (unused) copy — only the scheduler instance ever calls `alloc`/`free`.
- No locking: only one thread (the scheduler's main thread) mutates `_free_blocks`.
- Eviction beyond "raise when exhausted" is out of scope. Sizing `num_ec_blocks` is an operator concern in this PR.
- This is a small additive change to `ECSharedRegion` (currently on `cpu_ec_connector`). Bundled into this PR.

## Code reuse

Directly imported from existing vLLM code:

- `vllm.distributed.nixl_utils.NixlWrapper`, `nixl_agent_config` — NIXL Python bindings; handles the `nixl`/`rixl` (ROCm) import split and the `UCX_RCACHE_MAX_UNRELEASED` workaround.
- `vllm.utils.network_utils.make_zmq_socket`, `make_zmq_path` — ZMQ socket helpers supporting DEALER, ROUTER, and all needed buffer/HWM settings.
- `vllm.platforms.current_platform.get_nixl_memory_type()` — `"DRAM"` / `"VRAM"` string.

Patterns adapted (not imported) from `vllm/distributed/kv_transfer/kv_connector/v1/nixl/`:

- Background listener thread structure from `scheduler.py:_nixl_handshake_listener` — `RCVTIMEO` polling, `stop_event`, `ready_event`, `recv_multipart/send_multipart` with identity + empty-delimiter + payload.
- Compatibility-hash first-message handshake from `metadata.py:compute_nixl_compatibility_hash` — smaller factor set appropriate to EC.

Code specifically not reused:

- `zmq_ctx` helper in `nixl/utils.py` — rejects DEALER, unsuitable for consumer side.
- `NixlHandshakePayload`, `NixlAgentMetadata` — KV-specific fields (kv_cache_layout, block_lens, ssm_sizes, attn_backend_name). The EC schemas are new and simpler.

## File inventory

**Modified:**

- `vllm/distributed/ec_transfer/ec_connector/cpu_connector.py` — all scheduler-side additions.
- `vllm/distributed/ec_transfer/ec_connector/ec_shared_region.py` — add `_free_blocks`, `alloc`, `free`.
- `vllm/distributed/ec_transfer/ec_connector/base.py` — add default `shutdown()` no-op.
- `vllm/envs.py` — add `VLLM_EC_SIDE_CHANNEL_HOST`, `VLLM_EC_SIDE_CHANNEL_PORT`.

**New:**

- `vllm/distributed/ec_transfer/ec_connector/messages.py` — `XferReq`, `XferAck`, `EC_CONNECTOR_VERSION`, `compute_ec_compatibility_hash`.
- `tests/v1/ec_connector/unit/test_ec_transfer_scheduler.py` — unit tests with fake NIXL.
- `tests/v1/ec_connector/integration/test_ec_zmq_loopback.py` — two-connector in-process test with real ZMQ, fake NIXL.
- `tests/v1/ec_connector/_fakes.py` — `FakeNixlWrapper`.

**Unchanged:**

- `factory.py`, `example_connector.py`
- Any worker-process code.

## Testing strategy

### Unit — no network, no NIXL (`test_ec_transfer_scheduler.py`)

- `has_cache_item` state machine: `False` → `None` → `True` after simulated ack; → `False` after `ok=False` ack with blocks freed.
- `update_state_after_alloc`: builds a round-trippable `XferReq` with the expected `dst_block_indices`, `mm_hash`, compat hash.
- Duplicate `update_state_after_alloc` for same `mm_hash` is a no-op (no second XferReq, no second block allocation).
- `_drain_acks`: routes acks to the correct `mm_hash`; non-matching acks ignored.
- Producer listener: dispatches when `_encodings[mm_hash]` exists; parks otherwise; `_drain_waiting` fires the parked request on populate.
- NIXL failure path: fake raises in `make_prepped_xfer` → `XferAck(ok=False)`, consumer frees blocks.
- Block-size mismatch: `tensor.nbytes != n * block_size` → `ok=False`, no NIXL call made.
- Compat-hash mismatch: producer responds `ok=False` immediately, no NIXL call.
- `FreeListAllocator`: alloc / free round-trip; exhaustion raises.

### Integration — real ZMQ, fake NIXL (`test_ec_zmq_loopback.py`)

- Spin up two `ECCPUConnector` instances (producer + consumer) in the same process on `tcp://127.0.0.1:<random-port>`. Fake NIXL implements `register_memory`, `add_remote_agent`, `make_prepped_xfer`, `transfer`, `check_xfer_state` with a CPU-memcpy write path over a shared registry.
- `test_end_to_end_single_xfer`: populate `_encodings[hash]` with known bytes → consumer `update_state_after_alloc` → wait for ack (poll `has_cache_item` with short deadline) → consumer's `_region.blocks[indices]` equals the source bytes.
- `test_wait_then_drain`: send XferReq before `_encodings[hash]` exists; populate + `_drain_waiting`; verify ack arrives and bytes match.
- `test_two_concurrent_xfers_different_hashes`: both complete; state is clean; `build_connector_meta` returns both mappings.

### Real NIXL smoke test (optional, guarded)

- `test_ec_nixl_real.py` — single end-to-end test with actual NIXL over CPU memory, two processes via `multiprocessing.spawn`. Skipped if NIXL isn't importable. Sanity only; unit + integration tiers carry correctness.

### Fakes

`FakeNixlWrapper` in `tests/v1/ec_connector/_fakes.py` implements the method surface used by the connector (`get_reg_descs`, `register_memory`, `deregister_memory`, `get_xfer_descs`, `prep_xfer_dlist`, `release_dlist_handle`, `make_prepped_xfer`, `transfer`, `check_xfer_state`, `release_xfer_handle`, `add_remote_agent`, `remove_remote_agent`, `get_agent_metadata`). ~60 lines. WRITE xfers synchronously copy bytes from a shared registry so consumer mmap actually receives the expected bytes — lets tests assert real data arrival without any network or NIXL runtime.

## Open verification items (implementation-time)

These are items the design depends on but that I could not confirm without running the NIXL library:

1. **Serialization of xfer_descs between peers.** The consumer builds xfer_descs for its mmap via `get_xfer_descs(...)` and must send them to the producer over ZMQ. The equivalent KV flow doesn't use this pattern (KV ships full `NixlAgentMetadata` containing layout, and peers rebuild descs locally). The NIXL Python API likely exposes a serialize/deserialize method on the descs object; if it does not, the fallback is for the consumer to send raw `(base_ptr, num_blocks, block_size_bytes)` in `consumer_mem_descriptor` and have the producer reconstruct the descs locally via `get_xfer_descs(...)` using those numbers — which is fine because descs are just lists of `(addr, size, device_id)` tuples.
2. **NIXL ROCm/rixl parity.** The codepaths for ROCm use `rixl._api` instead of `nixl._api`. The vLLM `nixl_utils.py` module abstracts this, so the connector code should "just work" on both; the smoke test needs to be ROCm-aware (`@skipif(not HAS_NIXL)` covers this).
3. **`notif_msg` semantics with empty bytes.** We pass `notif_msg=b""` because we use ZMQ acks, not NIXL notifs. If NIXL requires a non-empty notif for `make_prepped_xfer` to succeed, we'll switch to a synthetic tag that we never poll for.

## Follow-up work

- `save_caches` on the producer worker (future PR): GPU encoder output → producer's `_encodings[mm_hash]` tensor; triggers `_drain_waiting`.
- `encoder_node_address` plumbed through `kv_transfer_params` (parallel PR); drop the `default_encoder_node` config fallback when that lands.
- Block free-on-request-finished (hook into the encoder-cache eviction path).
- Ack-timeout sweeper if stalls become operationally visible.
- Eviction policy for `_block_allocator` beyond "raise on exhaustion".
- Producer-side pre-registration of a pooled region if per-xfer NIXL registration overhead is measurable.
