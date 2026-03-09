# Progress

## Completed
- [x] Research: algorithms, libraries, GPU approaches, language options
- [x] Research document: `plans/pi-computation-research.md`
- [x] Architecture design: `plans/architecture.md`
- [x] Memory bank initialization
- [x] Phase 1: CPU-only pi engine
  - [x] CMake project with GMP + Google Test
  - [x] Multiplier interface + GmpMultiplier (10 tests)
  - [x] BinarySplitting with Chudnovsky formula (8 tests)
  - [x] NewtonDivider for division and sqrt (8 tests)
  - [x] BaseConverter for decimal output (5 tests)
  - [x] ChunkedWriter for streaming file output (6 tests)
  - [x] PiEngine orchestrator (5 integration tests)
  - [x] Validation against reference pi digits (4 tests)
  - [x] CLI with --digits, --output, --verbose flags
  - [x] README.md

## Benchmarks (Apple Silicon, CPU-only)
| Digits | Time | Terms |
|--------|------|-------|
| 1,000 | 0.001s | 79 |
| 10,000 | 0.04s | 714 |
| 100,000 | 0.02s | 7,060 |
| 1,000,000 | 0.28s | 70,522 |
| 10,000,000 | 4.27s | 705,145 |

## Remaining
- [ ] Phase 2: GPU acceleration (CUDA + cuFFT NTT)
- [ ] Phase 3: Optimization & scale (checkpointing, async, profiling)
- [ ] Phase 4: Verification & polish (BBP verifier, benchmark plots)

## Known Issues
- None currently. All 46 tests pass.

## Bugs Fixed
1. **Formula bug**: Denominator was `13591409*Q + R` but should be just `R` — the binary splitting R already incorporates the `a(k) = A + Bk` linear terms
2. **Precision bug**: Last digit was wrong due to insufficient guard digits. Fixed by computing with 100 extra guard digits and truncating
3. **Reference file**: Had 1024 digits instead of 1000. Trimmed to exact 1002 chars
4. **Test string comparison**: BaseConverter test compared 16 chars against 15-char expected string
