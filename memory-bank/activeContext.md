# Active Context

## Current State
Project is mature with CPU-only being the optimal path. GPU acceleration implemented but not beneficial due to per-call overhead exceeding GMP's optimized CPU FFT.

## Latest Changes (2026-03-12)
- **Cross-architecture benchmark script** (commit `2ceb42f`):
  - Automated Intel/AMD/Graviton3/Graviton4 comparison via EC2
  - `scripts/bench_architectures.sh` — launches, builds, benchmarks, collects, terminates
  - Running 1B digit benchmark in us-west-2 (results pending)
- **Core utilization improvement** (commit `b7f648d`):
  - 3-tier merge: small=sequential, medium=4-way parallel, large=2-way semi-parallel
  - Parallel subtrees: `compute_sequential` switches to parallel for ranges <10K terms
- **Memory optimization for 5B+ digits** (commit `117af36`):
  - Aggressive early freeing in `merge_parallel()` via `mpz_realloc2(x,0)`
  - Parallel depth capping: >3B digits → depth=2, >700M digits → depth=3
  - Early freeing in `pi_engine.cpp` — BSResult P/Q/R freed as soon as consumed
- Previous: RSS tracking, `--threads` CLI flag (commit `103e3d8`)

## Capacity Planning (50B digits)
- **RAM**: ~415 GB (peak ~8.3 × N bytes) → r7i.16xlarge (512 GB) recommended
- **Disk**: ~200 GB gp3 (checkpoints ~100 GB + output ~50 GB + pi_int ~21 GB + margin)
- **Time estimate**: ~14-20 hours on r7i.16xlarge (64 vCPU)
- **Cost**: ~$97 on-demand, ~$30 spot (r7i.16xlarge)

## Key Findings from Benchmarking

### p4d.24xlarge (96 vCPU + 8x A100), 100M digits:
- CPU (96 threads): **67.9s** — Binary splitting 36%, Final computation 23%, Decimal conversion 41%
- GPU 1x A100: 166s (slower — per-call overhead dominates)
- GPU 8x A100: 90s (multi-GPU helps but still slower than CPU)

### g6e.xlarge (4 vCPU + 1x L40S), 100M digits:
- CPU (4 threads): 82.4s
- GPU L40S: 96.5-114s (FP64-limited)

### Apple Silicon (10 threads), 10M digits:
- CPU: 2.78s (multi-threaded) / 4.27s (single-threaded)

### Cross-Architecture Benchmark (8 vCPU .2xlarge, 100M digits, GMP 6.2.1):
| Architecture | Instance | CPU | Total | BS | Final | String |
|-------------|----------|-----|-------|-----|-------|--------|
| **Graviton4** | c8g.2xlarge | Neoverse-V2 | **42.9s** | 17.9s | 13.0s | 12.0s |
| **AMD** | c7a.2xlarge | EPYC 9R14 | **45.8s** | 19.4s | 14.1s | 12.3s |
| **Intel** | c7i.2xlarge | Xeon 8488C | **49.7s** | 24.3s | 13.3s | 12.0s |
| **Graviton3** | c7g.2xlarge | (Graviton3) | **51.7s** | 21.6s | 15.7s | 14.4s |

**Key findings:**
- **Graviton4 wins overall** — 14% faster than Intel, 6% faster than AMD
- **AMD has fastest binary splitting** after Graviton4 — GMP FFT runs well on EPYC
- **Intel has fastest final computation** (sqrt+divide) — best single-thread perf
- **Graviton3 is slowest** — older ARM core, ~20% behind Graviton4
- **String conversion** is similar across all (~12s), except Graviton3 (~14s)

## Why GPU Doesn't Help
1. GMP's CPU FFT is incredibly optimized (hand-tuned assembly)
2. Per-GPU-call overhead: ~150ms (conversion + transfer + FFT + transfer back)
3. 3,064 GPU calls × 150ms = 460s cumulative overhead
4. Decimal conversion (41% of time) is untouched by GPU

## Remaining Optimization Opportunities
- Integer NTT (custom CUDA kernels, not cuFFT) — would use INT32 throughput
- MPIR instead of GMP — potentially 10-30% faster
- Compute pi as integer (avoid mpf entirely) — eliminates decimal conversion bottleneck
