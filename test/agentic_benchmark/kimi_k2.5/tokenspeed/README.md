# Agentic Benchmark — TokenSpeed

Sweep `ts serve` against an agentic, multi-turn workload (SWE-Smith) at a
fixed set of attention/MoE parallelism layouts and report per-config throughput,
latency, and KV-cache hit rate.

The server listens on port **8000**.

## Layout

```
agentic_bench.sh        # main sweep: prepare data, then launch, wait, benchmark, and stop each config
configs/                # one shell script per parallelism layout (each runs `ts serve` via `exec`)
collect_outputs.py      # parse a sweep into a flat CSV
outputs/<sweep_ts>/<config>/parallel_<P>_number_<N>/  # per-run EvalScope artifacts
```

## Run a sweep

```bash
cd test/agentic_benchmark/tokenspeed
./agentic_bench.sh
```

The script (1) installs EvalScope at the pinned commit, (2) builds the SWE-Smith
multi-turn dataset, and (3) iterates over the configurations in `CONFIGS=()`.
For each configuration, it launches the server, polls `/readiness`, runs
`evalscope perf`, stops the server, and waits for the port to become available.
The script aborts the entire sweep on the first failure (`set -e`).

To narrow the matrix, comment out entries in the `CONFIGS=()` array.

## Configs

Each `configs/*.sh` script runs `ts serve` via `exec` with the full flag set for one
layout. Key flags:

- `--attn-tp-size`, `--moe-tp-size` *or* `--ep-size`
- `--max-num-seqs`, `--max-prefill-tokens`, `--chunked-prefill-size`
- `--quantization nvfp4`, `--kv-cache-dtype fp8`
- `--moe-backend flashinfer_trtllm`, `--attention-backend tokenspeed_mla`
- Eagle3 speculative decoding: `--speculative-algorithm EAGLE3 --speculative-num-steps 3
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`

Naming: `attn_<X>_moe_<Y>` where `X ∈ {tp4,tp8,dp8}` and `Y ∈ {tp4,tp8,ep4,ep8}`.
World size = the number after `attn_(tp|dp)`.

To verify the parallelism actually applied, grep the server log:
```bash
grep -A6 "Parallelism configuration" /tmp/tokenspeed_server_<config>.log
```

## Collect results

```bash
python3 collect_outputs.py outputs/<sweep_ts> -o sweep.csv
```

Emits one row per (config, concurrency) with `Conc.`, `Latency (tps/user)`,
`Throughput (tps/gpu)`, `Approx Cache Hit`, `Decoded Tok/Iter`. `tps/gpu` divides
the system-wide `Total Throughput (tok/s)` by the GPU count inferred from the
config name; the other metrics come straight from `benchmark_summary.json`
(same numbers as EvalScope's `performance_summary.txt` Request Metrics table).
