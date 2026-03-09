# Computing Millions of Digits of Pi — Research & Architecture Plan

## 1. Algorithm Landscape

### 1.1 The Chudnovsky Algorithm (⭐ Recommended)
- **Convergence**: ~14.18 digits per term — the fastest known series for pi
- **Formula**: Based on Ramanujan-type series discovered by the Chudnovsky brothers (1988)
- **Used by**: y-cruncher (current world record holder), Google Cloud's 100-trillion-digit computation (2022)
- **Complexity**: O(n·log(n)³) with binary splitting + FFT-based multiplication
- **Key insight**: Combined with **binary splitting**, the series can be evaluated without any floating-point division until the very end — you accumulate giant integer numerators and denominators, then do one final division

### 1.2 Bailey–Borwein–Plouffe (BBP) Formula
- **Unique property**: Can compute the *n*-th hexadecimal digit of pi **without computing preceding digits**
- **Use case**: Verification of specific digit positions, not bulk computation
- **Limitation**: Works in base 16 (hex), converting to decimal requires knowing all prior digits anyway
- **Practical value**: Excellent for **verification** — compute digit at position N independently to cross-check your Chudnovsky result

### 1.3 Machin-like Formulas
- **Based on**: Arctangent identities (pi/4 = 4·arctan(1/5) - arctan(1/239), etc.)
- **Historical**: Used for most pi records before Chudnovsky
- **Convergence**: Slower than Chudnovsky but simpler to implement
- **Still relevant**: Good for a first prototype / learning exercise

### 1.4 Gauss–Legendre (AGM) Algorithm
- **Convergence**: Quadratic — doubles correct digits each iteration
- **Downside**: Requires full-precision multiplication at every step (memory-hungry)
- **For millions of digits**: Competitive but Chudnovsky with binary splitting is generally faster in practice due to better memory access patterns

### 1.5 Borwein's Algorithms
- **Quartic convergence** (quadruples digits per iteration)
- **Downside**: Same memory issues as AGM — each iteration needs full-precision arithmetic
- **Rarely used** in modern record-setting computations

### Algorithm Recommendation
**Chudnovsky + Binary Splitting** is the clear winner for millions of digits. It's what every modern record uses.

---

## 2. Core Mathematical Operations Needed

### 2.1 Arbitrary-Precision Integer Arithmetic
At millions of digits, numbers don't fit in any native type. You need:
- **Addition/Subtraction**: O(n) — trivial
- **Multiplication**: This is the **bottleneck**. Naive is O(n²), which is catastrophic at millions of digits
- **Division**: Typically done via Newton's method using multiplication

### 2.2 Fast Multiplication Algorithms (Critical Path)
| Algorithm | Complexity | Practical Range |
|-----------|-----------|-----------------|
| Schoolbook | O(n²) | < 100 digits |
| Karatsuba | O(n^1.585) | 100 – 10,000 digits |
| Toom-Cook 3 | O(n^1.465) | 10,000 – 100,000 digits |
| Schönhage–Strassen (SSA) | O(n·log(n)·log(log(n))) | 100K – 10B digits |
| Harvey–van der Hoeven | O(n·log(n)) | Theoretical; not practical yet |
| **Number Theoretic Transform (NTT)** | O(n·log(n)) | **The practical workhorse** |

**NTT (Number Theoretic Transform)** is the integer analog of FFT. It works in modular arithmetic (no floating-point rounding errors) and is the foundation of all fast arbitrary-precision multiplication at scale.

### 2.3 Binary Splitting
Converts the Chudnovsky series into a tree of integer multiplications:
- Recursively splits the series sum into halves
- Combines using only integer arithmetic (multiply, add)
- Final step: one giant division to get the decimal expansion
- **Memory pattern**: Can be done with bounded working memory if implemented carefully

---

## 3. Libraries & Existing Tools

### 3.1 GMP (GNU Multiple Precision Arithmetic Library) ⭐
- **Language**: C, with bindings for C++, Python, Rust, etc.
- **What it provides**: Arbitrary-precision integers, rationals, and floats
- **Multiplication**: Automatically selects optimal algorithm (schoolbook → Karatsuba → Toom → FFT) based on operand size
- **Performance**: Highly optimized with hand-tuned assembly for x86_64, ARM
- **Limitation**: **CPU only** — no GPU support
- **Python binding**: `gmpy2` — excellent for prototyping

### 3.2 MPFR (Multiple Precision Floating-Point Reliable)
- **Built on**: GMP
- **What it adds**: Correct rounding for floating-point operations
- **Use case**: If you need floating-point arbitrary precision (AGM algorithm)
- **For Chudnovsky+binary splitting**: GMP integers are sufficient; MPFR not strictly needed

### 3.3 y-cruncher ⭐⭐
- **The gold standard** for pi computation
- **Author**: Alexander Yee
- **Features**: Multi-threaded, disk-swapping for out-of-core computation, SIMD-optimized
- **Closed source** but free to use
- **Relevance**: Benchmark to compare against; not something you'd modify

### 3.4 FLINT (Fast Library for Number Theory)
- **Built on**: GMP
- **Adds**: Polynomial arithmetic, NTT, and more
- **Useful for**: If you want to implement binary splitting with optimized polynomial operations

### 3.5 mpmath (Python)
- **Pure Python** arbitrary-precision floating-point
- **Has built-in pi computation** (Chudnovsky + binary splitting)
- **Performance**: 10-100x slower than GMP-based solutions
- **Value**: Excellent for prototyping and verification

### 3.6 LAPACK / BLAS
- **Not applicable** — these are for linear algebra (matrices, eigenvalues, etc.)
- **Fortran heritage** but irrelevant for arbitrary-precision single-number computation
- Pi computation doesn't involve matrices or linear systems

---

## 4. Language Analysis

### 4.1 C / C++ ⭐⭐ (Recommended for production)
- **Pros**: Direct GMP/MPFR access, maximum control, best performance, CUDA interop
- **Cons**: Manual memory management, longer development time
- **Ecosystem**: GMP, FLINT, CUDA toolkit all native
- **Verdict**: Best choice for the final high-performance implementation

### 4.2 Rust ⭐ (Strong alternative)
- **Pros**: Memory safety, good GMP bindings (`rug` crate), excellent CUDA interop via `cudarc`
- **Cons**: Slightly less mature arbitrary-precision ecosystem
- **Crates**: `rug` (GMP wrapper), `num-bigint` (pure Rust, slower)
- **Verdict**: Great if you prefer safety; `rug` wraps GMP so performance is identical

### 4.3 Python (Prototyping only)
- **Pros**: Fastest development, `mpmath` built-in, `gmpy2` for GMP speed
- **Cons**: 10-100x overhead for orchestration; fine if GMP does the heavy lifting via `gmpy2`
- **Verdict**: Excellent for prototyping the algorithm, verifying correctness

### 4.4 Fortran
- **Pros**: Excellent numerical computing heritage
- **Cons**: Poor arbitrary-precision library ecosystem; no equivalent to GMP
- **LAPACK**: Not applicable (linear algebra, not arbitrary-precision arithmetic)
- **Verdict**: Not recommended for this specific problem

### 4.5 Java / JVM
- **Pros**: `BigInteger` / `BigDecimal` built-in
- **Cons**: Java's BigInteger is significantly slower than GMP (no NTT-based multiplication)
- **Verdict**: Possible but suboptimal

### 4.6 Go
- **Pros**: `math/big` package built-in
- **Cons**: Similar to Java — no NTT multiplication, much slower than GMP
- **Verdict**: Not recommended for performance-critical path

---

## 5. GPU Computation — The Big Question 🔥

### 5.1 Can GPUs Help? — It's Complicated

**Short answer**: Yes, but not in the way you might expect. GPUs excel at the **NTT/FFT multiplication step**, which is the bottleneck.

**The challenge**: Arbitrary-precision arithmetic is fundamentally about **carry propagation** — when you add two huge numbers, carries ripple from least-significant to most-significant digit. This is inherently **sequential** and fights against GPU parallelism.

### 5.2 Where GPUs Excel: NTT-Based Multiplication

The Number Theoretic Transform is essentially a **parallel butterfly operation** — exactly what GPUs are designed for:

1. **Forward NTT**: Transform both operands (embarrassingly parallel)
2. **Pointwise multiply**: Multiply corresponding elements (embarrassingly parallel)
3. **Inverse NTT**: Transform back (embarrassingly parallel)
4. **Carry propagation**: Sequential ripple (GPU-unfriendly, but small relative to NTT)

For numbers with millions of digits, the NTT dominates runtime, so GPU acceleration of NTT alone gives massive speedups.

### 5.3 Existing GPU Arbitrary-Precision Libraries

| Library | Language | Status | Notes |
|---------|----------|--------|-------|
| **CGBN** (CUDA GMP-like Big Numbers) | CUDA C++ | Active | From NVlabs; fixed-size big integers, not arbitrary |
| **cuFFT** | CUDA | Mature | NVIDIA's FFT library; can be adapted for NTT |
| **CUMP** | CUDA | Research | Multiple-precision on GPU; academic project |
| **GMP-ECM** | C + CUDA | Active | Uses GPU for specific operations |
| **Custom NTT kernels** | CUDA | DIY | Most practical approach for this project |

### 5.4 Practical GPU Strategy for Pi

The most effective approach is a **hybrid CPU+GPU architecture**:

```
Binary Splitting (CPU)          — tree of integer operations
    └── Large Multiplications   — offloaded to GPU via NTT
         ├── Forward NTT        — GPU kernel
         ├── Pointwise multiply — GPU kernel  
         ├── Inverse NTT        — GPU kernel
         └── Carry propagation  — CPU or GPU with careful sync
    └── Final Division          — Newton's method using GPU-accelerated multiply
```

### 5.5 GPU Frameworks

| Framework | Pros | Cons |
|-----------|------|------|
| **CUDA** (NVIDIA) | Best performance, mature ecosystem | NVIDIA-only |
| **OpenCL** | Cross-vendor (AMD, Intel, NVIDIA) | Harder to optimize, less tooling |
| **Vulkan Compute** | Cross-platform | Very low-level, overkill |
| **Metal** (Apple) | Native on macOS/Apple Silicon | Apple-only, limited for this use case |
| **SYCL / oneAPI** | Cross-platform, C++-like | Newer, less battle-tested |

**Recommendation**: CUDA if you have an NVIDIA GPU; OpenCL for AMD; Metal for Apple Silicon.

### 5.6 Apple Silicon Consideration
If you're on a Mac (which your system info suggests — macOS):
- **No CUDA** — Apple dropped NVIDIA support years ago
- **Metal Compute Shaders** — possible but limited ecosystem for this
- **Apple's Accelerate framework** — has vDSP for FFT, but not arbitrary-precision
- **Best bet on Mac**: Use CPU with GMP (Apple Silicon's unified memory and wide SIMD are quite fast)
- **For GPU experiments**: Consider a cloud GPU instance (AWS P-series, etc.)

---

## 6. Memory Management & Streaming Architecture

### 6.1 The Scale Problem
- 1 million digits ≈ 415 KB (in binary, ~3.32 bits per decimal digit)
- 10 million digits ≈ 4.15 MB
- 100 million digits ≈ 41.5 MB
- 1 billion digits ≈ 415 MB

For millions of digits, everything fits in RAM. The streaming/disk approach becomes critical at **billions** of digits.

### 6.2 Proposed Memory Architecture

```
┌─────────────────────────────────────────┐
│           Computation Engine            │
│  ┌─────────────────────────────────┐    │
│  │   Binary Splitting Tree         │    │
│  │   (bounded working set)         │    │
│  └──────────┬──────────────────────┘    │
│             │                           │
│  ┌──────────▼──────────────────────┐    │
│  │   Big Integer Multiply          │    │
│  │   (NTT on GPU or CPU)           │    │
│  └──────────┬──────────────────────┘    │
│             │                           │
│  ┌──────────▼──────────────────────┐    │
│  │   Final Division                │    │
│  │   (Newton's method)             │    │
│  └──────────┬──────────────────────┘    │
│             │                           │
│  ┌──────────▼──────────────────────┐    │
│  │   Decimal Conversion            │    │
│  │   (binary → base-10 string)     │    │
│  └──────────┬──────────────────────┘    │
└─────────────┼───────────────────────────┘
              │
    ┌─────────▼─────────┐
    │   Output Stream    │
    │  (chunked writes)  │
    │  - File (txt)      │
    │  - SQLite chunks   │
    │  - Verification    │
    └────────────────────┘
```

### 6.3 Output Strategy
- **Chunked file writing**: Write digits in blocks (e.g., 1M digits per chunk)
- **Verification**: Use BBP formula to spot-check random positions
- **Checkpointing**: Save intermediate binary splitting state for crash recovery
- **Format**: Plain text file with optional SQLite index for random access to digit ranges

---

## 7. Verification Strategy

Computing pi is only half the battle — you need to **prove** your digits are correct:

1. **Dual computation**: Run two different algorithms (e.g., Chudnovsky + Machin-like) and compare
2. **BBP spot-checks**: Independently compute hex digits at random positions
3. **Known-good comparison**: Compare against published digit databases (first 10 trillion digits are publicly available)
4. **Checksum verification**: y-cruncher publishes checksums for various digit counts

---

## 8. Proposed Phased Implementation

### Phase 1: Prototype (Python + gmpy2)
- Implement Chudnovsky + binary splitting in Python using `gmpy2`
- Verify correctness against known digits
- Benchmark: target 1M digits
- Learn the algorithm deeply before optimizing

### Phase 2: Production Engine (C/C++ + GMP)
- Port to C/C++ with direct GMP calls
- Implement chunked output and checkpointing
- Multi-threaded binary splitting
- Benchmark: target 10M+ digits

### Phase 3: GPU Acceleration (CUDA or Metal)
- Implement NTT multiplication kernel on GPU
- Hybrid architecture: CPU orchestrates, GPU multiplies
- Compare performance vs CPU-only GMP
- Benchmark: target 100M+ digits

### Phase 4: Verification & Polish
- BBP spot-check verification
- Dual-algorithm verification
- CLI interface with progress reporting
- Documentation and benchmarks

---

## 9. Key Open Questions for User

1. **Hardware**: What GPU do you have? (NVIDIA → CUDA, AMD → OpenCL, Apple Silicon → Metal/CPU)
2. **Target scale**: Millions? Tens of millions? Billions?
3. **Language preference**: Start with Python prototype, or jump straight to C/C++?
4. **Primary goal**: Learning the algorithms? Maximum performance? Both?
5. **GPU priority**: Is GPU acceleration a must-have or a nice-to-have exploration?
