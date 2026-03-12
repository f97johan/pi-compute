# Active Context

## Current State
Project is mature with CPU-only being the optimal path. GPU acceleration implemented but not beneficial due to per-call overhead exceeding GMP's optimized CPU FFT.

## Latest Changes (2026-03-11)
- Added RSS memory tracking (`get_rss_mb()`) — displays memory usage at each verbose step
- Added `--threads <N>` CLI flag for explicit CPU thread control
- Cleaned up `pi_engine.h` (removed verbose doxygen comments)
- All 46 tests passing, pushed to `origin/main` (commit `103e3d8`)

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

## Why GPU Doesn't Help
1. GMP's CPU FFT is incredibly optimized (hand-tuned assembly)
2. Per-GPU-call overhead: ~150ms (conversion + transfer + FFT + transfer back)
3. 3,064 GPU calls × 150ms = 460s cumulative overhead
4. Decimal conversion (41% of time) is untouched by GPU

## Remaining Optimization Opportunities
- Integer NTT (custom CUDA kernels, not cuFFT) — would use INT32 throughput
- MPIR instead of GMP — potentially 10-30% faster
- Compute pi as integer (avoid mpf entirely) — eliminates decimal conversion bottleneck
