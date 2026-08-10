# Native Pipeline Parallelism

Native pipeline parallelism is a model-independent runtime capability with a
strict qualification gate. A server cannot enter distributed execution until
topology, stage ownership, cache projection, weight loading, boundary transport,
and every rank's step descriptor agree on the same static plan.

The first executable configuration is Kimi-K3, BF16, `PP8 x TP8`. Other model
architectures, activation dtypes, and pipeline depths fail before distributed
startup. Executable does not mean cluster-qualified: the text, image, memory,
fault, and performance gates below must pass before the configuration is used as
a service.

## Rank Layout

Pipeline parallelism is the outermost rank dimension. Every stage has the same
width and occupies one contiguous range of global ranks:

```text
global_rank = stage_id * stage_world_size + stage_local_rank
stage_world_size = world_size / stage_count
```

Ranks with the same `stage_local_rank` form a pipeline lane. For a 64-rank,
eight-stage deployment with eight ranks per stage, the stage groups are
`[0..7]`, `[8..15]`, through `[56..63]`; lane 3 is
`(3, 11, 19, 27, 35, 43, 51, 59)`.

The top-level mapping keeps `world_size=64` because process launch, rendezvous,
and pipeline lanes are global. Attention, dense, MoE, context, and vision
collectives resolve against the eight-rank stage width and must never cross a
stage boundary. With `stage_count=1`, group membership and the mapping string
representation remain identical to the existing runtime.

## Static Contracts

The pipeline package defines four startup-time contracts:

- `PipelinePlan` owns an ordered, gap-free layer partition.
- `StagePlan` identifies embedding/head ownership and adjacent schemas.
- `ActivationSchema` defines an ordered tensor-like payload and a canonical
  SHA-256 digest.
- `StageModel` is the model-independent execution boundary.

Only stage 0 owns external model inputs, and only the final stage may return a
model output. Adjacent stages require exact schema-digest equality. Canonical
digests reject floating-point values so topology and plan identity do not vary
with JSON formatting or float serialization.

`WholeModelStage` and `LocalPipelineExecutor` preserve the current one-stage
call path. `LoopbackPipelineExecutor` executes multiple fake stages in one
process and validates every boundary before device transport is introduced.

## Wire And Failure Protocol

Every eager forward begins with a global Gloo all-gather of a fixed step
descriptor. It includes the process epoch, monotonic step id, forward mode,
batch geometry, plan digest, and a deterministic logical-batch fingerprint.
The fingerprint covers input and shifted tokens, request/cache-pool indices,
sampling parameters, and image identity/placement metadata. The same
all-gather carries a physical page-table fingerprint that is compared within
each eight-rank stage group; different stages may own different cache groups.
Every rank validates every stage, so any mismatch poisons the full instance
before activation NCCL starts.

Activation traffic uses a fixed 64-word versioned header followed by tensors in
the schema's static order. The receiver validates step, source and destination
stage, schema digest, field count, dimensions, and total elements before it
allocates or receives payload tensors. Sampling results use a separate NCCL
process-group role and a fixed 16-word header so activation P2P and result
broadcasts cannot reorder on one communicator.

A dedicated global Gloo group carries fault packets independently of the data
communicators. Any rank failure poisons every peer and arms a hard process-exit
deadline. A forward step also has an operator-configurable hard deadline via
`--pipeline-step-timeout-seconds`. The runtime does not attempt in-process NCCL
recovery; the supervisor must restart the whole 64-rank instance.
Graceful fault-listener shutdown is deadline-bound. A failed STOP handshake,
an active step during close, or a rank-local SIGTERM also poisons the instance
and retains the hard-exit deadline instead of returning from cleanup while a
peer remains blocked.

## Stage-Local Cache

The cache scheduler keeps logical page identifiers shared across the pipeline.
Each rank projects the complete logical layer ownership map into a local cache
layout:

- every logical layer has exactly one stage owner;
- remote-stage fields are omitted from the local physical arena;
- logical layer IDs map to dense local physical layer IDs;
- field, group, and logical page identifiers are preserved;
- K3 physical planes are compacted by local group occurrence after projection,
  removing holes left by global interleaving without changing logical IDs;
- a canonical rank manifest records the projected geometry and ownership.

The Kimi-K3 cache recipe keeps the global 25% padding guard, then validates the
qualified PP8 projection against exact per-stage plane counts and parent byte
sizes. It reaches a global minimum token-capacity consensus. Capacity planning
first reserves the AttnRes slab and received Wire tensors; at 8K/BF16 stage 7
reserves 16 hidden-width tensors, approximately 1.75 GiB. Logical-to-physical
pool access and live HBM still require device validation on every rank.

## Weight Ledger

`tokenspeed.tools.checkpoint_ledger` reads safetensors headers and an optional
Hugging Face index without importing Torch or loading tensor values. A strict
JSON plan assigns tensors to stages and describes TP/EP ownership and
partitioning. Output distinguishes:

- `exact`: one complete tensor on one rank;
- `replicated`: a complete tensor on multiple ranks;
- `sharded`: byte-provable TP/EP slices or expert ownership;
- `unknown`: any ownership, dtype, shape, or partition that cannot be proven.

Unknown entries remain visible and prevent a byte ledger from being considered
complete. Average model size is not an acceptable HBM gate; startup planning
uses the worst per-rank total.

```bash
python -m tokenspeed.tools.kimi_k3_checkpoint_plan \
  --config /path/to/config.json \
  --pipeline-parallel-size 8 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --output ownership-plan.json

python -m tokenspeed.tools.checkpoint_ledger \
  --checkpoint /path/to/model.safetensors.index.json \
  --plan ownership-plan.json \
  --summary-only \
  --output checkpoint-ledger-summary.json
```

Omit `--summary-only` when a full per-tensor/rank audit trail is required. The
summary path still validates and accounts for every source tensor but avoids
retaining millions of expanded TP records in memory.

The K3 generator mirrors the runtime's AttnRes-aligned layer partition and
accounts for the checkpoint-padded `A_log` buffer. PP8 qualification requires
`--mm-encoder-tp-mode data`: K3 has 12 vision heads, which cannot be
weight-sharded over TP8. In data mode each first-stage rank owns a complete
TP1 vision tower and the runtime distributes whole image items across ranks.

## Kimi-K3 First Target

The first device target is target-only, eager `PP8 x TP8` on 64 GPUs with
decoder ranges `[0,12)`, `[12,24)`, `[24,36)`, `[36,48)`, `[48,60)`,
`[60,72)`, `[72,84)`, and `[84,93)`. Pipeline stages match the eight-GPU
server boundary, keeping every layer's TP collective within one server. The
final stage has three fewer decoder layers to reserve room for final
normalization, the LM head, and sampling.

A K3 boundary cannot contain only the current hidden state. Correct execution
must carry the prefix stream plus every completed AttnRes snapshot. Across the
seven PP8 boundaries this grows from 2 through 8 BF16 hidden-width tensors, or
approximately 28 through 112 KiB per token per TP rank for hidden width 7168.
A model adapter owns this schema; the generic executor has no K3 branch.

Stage 0 owns text embeddings and the image path. Stage 7 owns final AttnRes,
normalization, the LM head, logits processing, and sampling. The first text
oracle runs without speculative decoding, followed by image parity after the
text path and stage-0 memory peak pass.

The initial launch must use eager target-only execution and explicitly select
the currently qualified surface:

```text
--world-size 64
--pipeline-parallel-size 8
--attn-tp-size 8
--mm-encoder-tp-mode data
--max-model-len 1048576
--max-total-tokens 1048576
--max-num-seqs 4
--chunked-prefill-size 8192
--kv-cache-dtype fp8
--attention-backend mla
--kda-backend fla
--moe-backend marlin
--enforce-eager
--disable-prefill-graph
--disable-overlap-schedule
--disable-autotune
--no-enable-prefix-caching
--disable-kvstore
--grammar-backend none
```

Startup rejects incompatible flags instead of silently changing them. Keep the
first qualification run deterministic and add features one contract and parity
gate at a time.

For Python-only qualification fixes, `docker/Dockerfile.runtime-overlay` may
layer the current runtime package over an already qualified SM120 image. This
path is valid only when the following source guard is empty relative to the
declared base revision:

```bash
git diff --exit-code <base-revision>..HEAD -- \
  tokenspeed-kernel tokenspeed-kernel-amd tokenspeed-mla tokenspeed-scheduler
```

The overlay removes the old package tree, force-reinstalls `python/` without
changing dependencies, runs `pip check`, and records both base and overlay
revisions as OCI labels. Any kernel, scheduler, dependency, or base-image
change requires the full NVIDIA Dockerfile instead.

The K3 checkpoint does not declare FP8 KV metadata, so the qualification
command must not leave `--kv-cache-dtype` at `auto` (BF16 for this checkpoint).
SM120 qualification pins the generic MLA history consumer because the current
`tokenspeed_mla` prefill kernel only implements SM100 and SM103. KDA similarly
uses the portable FLA path, while the MXFP4 SiTU experts use the qualified
SM90+ Marlin path. These are explicit compatibility choices, not the final
performance ceiling. The 1M model length remains visible while the initial 1M
global token pool admits either one full-context request or several shorter
requests; increase the global pool only after measured HBM headroom passes on
all stages.

Runtime profiling, pause/resume, memory release, and online weight mutation are
also rejected while PP is active. They need a stage-global transaction and
status-reduction protocol before their API result can be trusted. Use external
Nsight/CUPTI capture for the first PP8 qualification run.

## Qualification Events

Startup success boundaries emit compact, single-line JSON payloads through the
normal rank-local logger. Every payload has `schema="tokenspeed.qualification"`,
`schema_version=1`, `status="success"`, and a stable `event` name. The payloads
contain topology identifiers and digests only; they never include requests,
tokens, prompts, images, or model tensor values.

- `distributed_topology` is emitted after distributed initialization and the
  stage-local memory-balance check. It records global, local, stage, TP, and
  vision TP/DP rank and group membership, plus the CUDA device and world
  process-group backend.
- `pipeline_plan_consensus` is emitted on every rank only after the global plan
  comparison succeeds. `pipeline_plan_digest` is the complete lowercase
  SHA-256 digest.
- `kimi_k3_cache_abi_consensus` is emitted after both `layout` and `runtime`
  cache consensus. `consensus_phase` distinguishes the rounds, while
  `global_abi_digest` and `stage_abi_digest` retain the complete SHA-256 values.

A failed consensus raises before its corresponding success event. Qualification
collectors must require the expected event count and rank set, not just search
for one successful line.

## Bring-Up Gates

The first multi-stage runtime must fail startup when any of these are enabled:

- data or context parallelism combined with pipeline parallelism;
- target/draft speculative decoding;
- prefill/decode or encoder disaggregation;
- decode or prefill CUDA graphs;
- online weight replacement;
- an unrecognized model adapter, boundary schema, or cache layout;
- any unassigned or unknown checkpoint byte in the selected weight plan.

The restrictions are capability gates, not permanent architecture choices.
Each can be removed only with a dedicated ownership contract and parity test.

## Delivery Order

1. Preserve the existing path through the one-stage adapter and exhaustive
   topology tests.
2. Generate exact per-rank weight and stage-local cache manifests.
3. Run fake and loopback PP2 protocol tests, including malformed schemas and
   final-stage authority.
4. Build stage-local K3 modules and prove missing/unexpected weight coverage.
5. Add eager device transport and run target-only PP8 text parity.
6. Measure per-rank HBM after weight load and eager warmup; reject the profile
   on the worst rank.
7. Add image parity and profile stage 0.
8. Add graphs, deeper pipelines, and speculative decoding behind independent
   capability gates.

No throughput result is accepted before logits, generated text, cache
commit/rollback, cancellation, and restart behavior pass their parity gates.
