# Progress

## Completed
- [x] Research: algorithms, libraries, GPU approaches, language options
- [x] Research document: `plans/pi-computation-research.md`
- [x] Architecture design: `plans/architecture.md`
- [x] Memory bank initialization
- [x] Phase 1: CPU-only pi engine (46 tests, 10M digits in 4.27s)
- [x] Phase 2: GPU code written and pushed
- [x] RSS memory tracking + --threads CLI flag (commit 103e3d8)
- [x] Memory optimization for 5B+ digit OOM (commit 117af36)
- [x] Core utilization: 3-tier merge + parallel subtrees (commit b7f648d)
- [x] Cross-architecture benchmark script: Intel/AMD/Graviton3/Graviton4 (commit 5ddaff9)
- [x] OOM fix: parallel depth=1 for >1.4B digits (commit 9d2347c)
- [x] Production run script: run_pi.sh with S3 output (commit 66b53f6)
- [x] Checkpoint disk fix: delete children + raise threshold (commit b79c9b3)
- [x] Out-of-core binary splitting: --out-of-core flag (commit db6f529)
- [x] FLINT multi-threaded multiplication: --flint flag (commit 58a5bb1)
- [x] Integer-only math mode: --integer-math flag (commit 7d4d9a3)
  - [x] NttEngine wrapping cuFFT
  - [x] Pointwise multiply CUDA kernel
  - [x] Carry propagation
  - [x] GpuNttMultiplier (GMP ↔ GPU bridge)
  - [x] 12 GPU tests
  - [x] --gpu CLI flag
  - [x] Linux build scripts (setup.sh, benchmark.sh, setup_cloud_gpu.sh)
  - [x] GitHub repo: https://github.com/f97johan/pi-compute

## In Progress
- [ ] Test GPU code on actual NVIDIA hardware (p3.2xlarge or g7e.2xlarge)

## Remaining
- [ ] Phase 3: Optimization (checkpointing, async transfers, profiling)
- [ ] Phase 4: Verification & polish (BBP verifier, benchmark plots)

## Benchmarks (Apple Silicon, CPU-only)
| Digits | Time | Terms |
|--------|------|-------|
| 1,000 | 0.001s | 79 |
| 10,000 | 0.04s | 714 |
| 100,000 | 0.02s | 7,060 |
| 1,000,000 | 0.28s | 70,522 |
| 10,000,000 | 4.27s | 705,145 |

## Bugs Fixed
1. Formula: denominator is R(0,N) not 13591409*Q+R
2. Precision: 100 guard digits + truncation
3. Reference file: trimmed to exact 1002 chars
4. Test string comparison: off-by-one in expected length
