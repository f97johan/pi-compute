# Tech Context

## Tech Stack
- **Language**: C++17
- **GPU**: CUDA (cuFFT for NTT)
- **Arbitrary Precision**: GMP (libgmp) >= 6.2
- **Build**: CMake >= 3.18 (CUDA language support)
- **Testing**: Google Test (FetchContent)
- **Benchmarking**: Google Benchmark (FetchContent)
- **Verification scripts**: Python 3 (mpmath, matplotlib)

## Dependencies
| Dependency | Purpose | macOS Install | Linux Install |
|-----------|---------|---------------|---------------|
| GMP >= 6.2 | Big integers | `brew install gmp` | `apt install libgmp-dev` |
| CUDA >= 11.0 | GPU compute | N/A (no CUDA on Mac) | NVIDIA installer |
| Python 3 | Verification scripts | System | System |
| mpmath | Pi verification | `pip install mpmath` | `pip install mpmath` |

## Platform Constraints
- macOS Apple Silicon: No CUDA, no NVIDIA GPU — CPU-only builds
- NVIDIA GPU machine: Full CUDA support — GPU-accelerated builds
- Multiplier interface abstracts this difference at compile time via CMake

## Key Technical Decisions
- **Base 2^24 for GPU**: Products fit in double-precision mantissa (2^48 < 2^53)
- **GMP limbs are 64-bit**: Need conversion layer to/from GPU base-2^24 format
- **cuFFT double-precision**: Sufficient for up to ~10^15 digits; our target is 10^8
- **Pinned memory**: For fast CPU↔GPU transfers of large number arrays
