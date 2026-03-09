# System Patterns

## Architecture Pattern: Hybrid CPU+GPU with Strategy Pattern
- `Multiplier` interface abstracts CPU vs GPU multiplication
- `PiEngine` orchestrator selects multiplier based on runtime capabilities and operand size
- Binary splitting runs on CPU; large multiplications dispatched to GPU when available
- Clean separation enables CPU-only builds on macOS and GPU builds on NVIDIA machines

## Key Algorithms
1. **Chudnovsky + Binary Splitting**: Converts infinite series into tree of integer operations
2. **NTT via cuFFT**: Number Theoretic Transform for O(n log n) multiplication
3. **Newton Division**: Division via iterative multiplication (converges quadratically)
4. **BBP Formula**: Independent digit verification at arbitrary positions

## Data Flow
```
Chudnovsky Series → Binary Splitting Tree → Large Integer Multiplications → NTT on GPU
→ Final Division (Newton) → Binary-to-Decimal Conversion → Chunked File Output
```

## Number Representation
- GMP `mpz_t` for all integers on CPU
- Base 2^24 digit arrays on GPU (fits in double-precision for cuFFT)
- Conversion layer between GMP limbs and GPU format

## Build Configuration
- CMake with `ENABLE_CUDA` option
- CPU-only build: macOS Apple Silicon (no CUDA dependency)
- GPU build: Linux with NVIDIA CUDA Toolkit
