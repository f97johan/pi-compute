# Pi Compute — High-Performance Pi Digit Calculator

A C++17 application that computes millions of decimal digits of pi using the **Chudnovsky algorithm** with **binary splitting**, powered by **GMP** (GNU Multiple Precision Arithmetic Library).

## Performance

| Digits | Time (Apple Silicon M-series) | Terms |
|--------|------------------------------|-------|
| 1,000 | 0.001s | 79 |
| 10,000 | 0.04s | 714 |
| 100,000 | 0.02s | 7,060 |
| 1,000,000 | 0.28s | 70,522 |
| 10,000,000 | 4.27s | 705,145 |

## Quick Start

### Prerequisites

```bash
# macOS
brew install gmp cmake

# Ubuntu/Debian
sudo apt install libgmp-dev cmake build-essential
```

### Build

```bash
cmake -B build -DENABLE_CUDA=OFF
cmake --build build
```

### Run

```bash
# Compute 1 million digits
./build/src/pi_compute --digits 1000000 --output pi_1M.txt --verbose

# Compute 10 million digits
./build/src/pi_compute --digits 10000000 --output pi_10M.txt --verbose
```

### Test

```bash
# Run all 46 tests
cd build && ctest --output-on-failure
```

## Architecture

```
Chudnovsky Series → Binary Splitting Tree → GMP Integer Multiplications
    → Final Division (GMP mpf) → sqrt(10005) → Decimal Conversion → File Output
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `PiEngine` | `src/engine/pi_engine.cpp` | Top-level orchestrator |
| `BinarySplitting` | `src/engine/binary_splitting.cpp` | Chudnovsky series via binary splitting |
| `GmpMultiplier` | `src/arithmetic/gmp_multiplier.cpp` | GMP-based multiplication (CPU) |
| `NewtonDivider` | `src/arithmetic/newton_divider.cpp` | High-precision division and sqrt |
| `BaseConverter` | `src/io/base_converter.cpp` | Binary-to-decimal conversion |
| `ChunkedWriter` | `src/io/chunked_writer.cpp` | Streaming file output |

### Strategy Pattern for GPU Extension

The `Multiplier` interface allows swapping between CPU (GMP) and GPU (cuFFT NTT) multiplication without changing any algorithm code:

```cpp
class Multiplier {
    virtual void multiply(mpz_t result, const mpz_t a, const mpz_t b) = 0;
    virtual void square(mpz_t result, const mpz_t a) = 0;
};
```

## Algorithm

The **Chudnovsky formula** converges at ~14.18 digits per term — the fastest known series for pi:

```
1/π = 12 · Σ ((-1)^k · (6k)! · (13591409 + 545140134k)) / ((3k)! · (k!)³ · 640320^(3k+3/2))
```

**Binary splitting** converts this infinite series into a tree of integer multiplications, computing three sequences P(a,b), Q(a,b), R(a,b) recursively. The final result:

```
π = 426880 · √10005 · Q(0,N) / R(0,N)
```

## Testing

46 tests across 7 test suites:

- **GmpMultiplier** (10 tests): Multiplication correctness, edge cases, commutativity
- **BinarySplitting** (8 tests): Per-term values, merge correctness, term count estimation
- **NewtonDivider** (8 tests): Division, square root, precision verification
- **BaseConverter** (5 tests): Decimal string conversion at various precisions
- **ChunkedWriter** (6 tests): File I/O, chunking, error handling
- **PiEngine** (5 tests): Integration tests at 50, 100, 1000, 10000 digits
- **Validation** (4 tests): Hardcoded digit verification, reference file comparison, cross-scale consistency

## Project Structure

```
pi/
├── CMakeLists.txt              # Build system
├── src/
│   ├── main.cpp                # CLI entry point
│   ├── engine/                 # Pi computation logic
│   ├── arithmetic/             # Multiplication, division interfaces
│   └── io/                     # File output, base conversion
├── tests/
│   ├── test_*.cpp              # 46 unit/integration/validation tests
│   └── data/pi_1000.txt        # Reference digits for validation
├── plans/
│   ├── pi-computation-research.md  # Algorithm research
│   └── architecture.md             # System design & GPU plan
└── memory-bank/                # Project context for continuity
```

## Future: GPU Acceleration (Phase 2)

The architecture is designed for GPU extension via CUDA + cuFFT:

- **NTT multiplication** on GPU for numbers > 100K digits
- **Hybrid CPU+GPU**: CPU orchestrates binary splitting, GPU handles large multiplies
- **cuFFT-based NTT**: Number Theoretic Transform for O(n·log(n)) multiplication
- See `plans/architecture.md` for the full GPU design

## License

Personal hobby project.
