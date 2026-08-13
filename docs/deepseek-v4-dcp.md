# DeepSeek V4 Decode Context Parallelism

TokenSpeed implements decode context parallelism (DCP) inside an attention
tensor-parallel group. DCP does not add ranks or change model-weight sharding;
it partitions the physical DeepSeek V4 history caches that TP would otherwise
replicate.

For DCP width `D`, local DCP rank `r`, and cache interleave `I`, global logical
row `g` is owned by:

```text
owner(g) = floor(g / I) mod D
```

Its physical local logical row is:

```text
local(g) = floor(g / (D * I)) * I + (g mod I)
```

The inverse used by the sparse indexer is:

```text
global(local) = floor(local / I) * (D * I) + r * I + (local mod I)
```

## Attention data flow

Each DCP rank stores only its owned SWA, C4, C128, and indexer-cache rows. For
attention, ranks all-gather TP-local query heads, run sparse MLA against their
local KV shard, all-gather natural-log LSE states, and reduce-scatter the
normalized output heads. Attention sinks are enabled on exactly one KV shard
so they are not counted once per DCP rank.

Decode and prefill use separate collective workspaces. Prefill growth therefore
cannot increase the fixed collective shape used by single-token decode.

## Sparse indexer data flow

The C4 indexer computes logits and top-K against only the local physical cache.
It restores selected local rows to absolute global compressed rows, then
all-gathers only `(score, row)` candidates. A final top-K over `D * K`
candidates produces identical global-row semantics on every rank. The
attention backend filters those rows back to the local physical shard.

This exchange is `O(D * K)` per query rather than `O(context length)`. For
`D=2`, `K=2048`, FP32 scores, and INT32 row IDs, it transfers 32 KiB of
candidates per query and indexer layer.

All DCP collectives use the dedicated `dcp` process-group role. Every rank in a
DCP group must execute indexer and attention collectives in the same layer
order.
