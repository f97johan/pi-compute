# Lessons Learned

## Implementation Lessons

- **Chudnovsky binary splitting formula**: The final formula is `pi = 426880 * sqrt(10005) * Q(0,N) / R(0,N)`. The denominator is **just R**, not `13591409*Q + R`. This is because R already accumulates the `a(k) = 13591409 + 545140134*k` linear terms during the binary splitting base case. Getting this wrong gives pi/2 or other wrong multiples.

- **Guard digits are essential**: When computing N digits of pi, you must compute with N+100 (or more) guard digits and then truncate. The last few digits of any arbitrary-precision computation are unreliable due to rounding in intermediate operations. Without guard digits, the last 1-2 digits will be wrong.

- **GMP's mpf_get_str returns fewer digits than requested** if the internal precision is insufficient. Always ensure the mpf precision (in bits) is set to at least `digits * 3.3219 + 64` before any operation.

- **Binary splitting base case for k=0 is special**: P(0,1)=1, Q(0,1)=1, R(0,1)=13591409. For k≥1, P includes the sign alternation via negation.

- **LAPACK/BLAS are not applicable** for arbitrary-precision single-number arithmetic. They are linear algebra libraries (matrices, eigenvalues). This is a common misconception when people think "numerical computing = LAPACK."

- **Fortran's numerical heritage doesn't help here** — Fortran excels at array/matrix operations but has a poor arbitrary-precision library ecosystem. GMP (C library) is the standard.

- **cuFFT precision limits**: Double-precision cuFFT is sufficient for numbers up to ~10^15 digits when using base 2^24. Beyond that, you need split-radix or multi-precision FFT techniques. For our 10M–100M target, single double-precision convolution is fine.

- **Apple Silicon has no CUDA**: Apple dropped NVIDIA support. Metal Compute Shaders exist but have limited ecosystem for this use case. Best approach: develop algorithm on Mac, run GPU code on cloud NVIDIA instance.

- **GMP is extremely hard to beat**: 30+ years of hand-tuned assembly. Don't try to write a faster big-integer library unless you're prepared for years of work. Use GMP and focus on the interesting parts (algorithm architecture, GPU integration).

- **GMP on Apple Silicon is surprisingly fast**: 10M digits of pi in 4.27 seconds using only CPU. The wide SIMD units and unified memory architecture of Apple Silicon make GMP very competitive even without GPU acceleration.

## Testing Lessons

- **Reference data must be exact**: The pi_1000.txt file must have exactly 1002 characters (2 for "3." + 1000 digits). Having extra digits causes length mismatch failures.

- **String comparison tests need exact lengths**: When comparing substrings, ensure the expected string length matches the substring length being extracted. Off-by-one in string length causes false failures.

- **FFT convolution base must account for accumulation**: With base B and FFT size N, the maximum convolution value is B² × N. This must fit in double precision (53-bit mantissa): B² × N < 2^53. Base 2^24 overflows for N > 32 (i.e., numbers with more than ~100 digits). Base 2^15 is safe for FFT sizes up to 2^20 (~1M elements, sufficient for 100M+ digit numbers).

- **NVCC is stricter about includes than GCC/Clang**: Always include `<string>`, `<vector>`, etc. explicitly in CUDA headers. Transitive includes that work with GCC may not work with NVCC.

- **CUDA architecture auto-detection**: Use `CUDA_ARCHITECTURES "native"` (CMake 3.24+) instead of hardcoding architecture numbers. CUDA 13.x dropped support for compute_70 (Volta), so hardcoded lists break on newer toolkits.

- **GPU AMI required**: Standard Amazon Linux AMIs don't include NVIDIA drivers even on GPU instances. Use a "Deep Learning AMI" or "GPU AMI" that comes with drivers + CUDA pre-installed.

## Memory Optimization Lessons

- **Binary splitting peak memory**: At the top of the tree, P, Q, R each have ~N digits (where N is the target). For 5B digits, each is ~2GB. The merge needs all 6 inputs + multiplication scratch (~3x input size). With parallel merge (4 concurrent multiplications), peak can reach 40+ GB.

- **`mpz_realloc2(x, 0)` frees GMP internal storage**: This is the correct way to release memory from an mpz_t without clearing it. The variable remains valid (set to 0) but its limb allocation is freed. Much better than `mpz_set_ui(x, 0)` which keeps the allocation.

- **Parallel depth vs memory tradeoff**: Each level of parallel tree traversal doubles the number of concurrent BSResult triples. For 16 threads (depth=4), 16 branches are alive simultaneously. Capping depth to 2-3 for large computations saves enormous memory with minimal performance impact because top-level merges are memory-bound (GMP FFT), not CPU-bound.

- **Free inputs in merge order**: In sequential merge, order multiplications to free each input as soon as its last use completes. This progressively reduces live memory from 6 inputs to 4, then 2, then 0 during the merge.

- **Free BSResult fields in pi_engine immediately**: P is unused in the final formula (pi = 426880 * sqrt(10005) * Q / R). Free it right after binary splitting. Free Q after multiplying by 426880. Free R after converting to mpf. This saves ~3x the top-level number size before the expensive final computation.

- **Depth=2 still OOMs for 5B digits on 64 GB**: With 4 concurrent branches (depth=2), the level-1 merges run in parallel, each with ~1.25B digit numbers. Two concurrent merges × (6 inputs + scratch) exceeded 60 GB. Depth=1 (2 branches) reduces peak to ~21 GB for 5B digits. The key insight: it's not just the top-level merge that matters — the level below also runs concurrently and its memory adds up.

- **GMP FFT scratch is the hidden memory killer**: GMP's FFT multiplication of two N-byte numbers allocates ~8N bytes of scratch internally. This is invisible to our code but dominates peak RSS. Two concurrent multiplications of 2 GB numbers = 32 GB of scratch alone.

- **Checkpoint files accumulate and fill disk**: The time-based checkpointing (every 60s) creates checkpoint files at many tree levels. For 50B digits, this exhausted 200 GB in ~4 hours. Fix: (1) delete child checkpoints when parent is saved (children are redundant), (2) raise minimum checkpoint range from 1K to 100K terms to avoid thousands of tiny files.

- **Disk space for large runs**: Output file = N bytes (1 byte/digit). Top-level checkpoint = ~3 × 0.415 × N bytes. With cleanup, total disk ≈ 2 × N bytes. For 50B digits: ~100 GB minimum, 500 GB recommended.
