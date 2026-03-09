# Project Brief: Pi Computation Engine

## Overview
A high-performance C++/CUDA application that computes millions (and potentially billions) of decimal digits of pi using the Chudnovsky algorithm with binary splitting, accelerated by GPU-based NTT multiplication via cuFFT.

## Goals
1. **Primary**: Explore GPU acceleration for arbitrary-precision arithmetic
2. **Secondary**: Compute 10M–100M+ digits of pi accurately and as fast as possible
3. **Tertiary**: Learn the algorithm architecture (Chudnovsky, binary splitting, NTT)

## Key Decisions
- **Algorithm**: Chudnovsky + Binary Splitting (fastest known series, ~14.18 digits/term)
- **Language**: C++ with CUDA
- **CPU Arithmetic**: GMP (GNU Multiple Precision) for arbitrary-precision integers
- **GPU Multiplication**: cuFFT-based NTT for large number multiplication
- **Architecture**: Hybrid CPU+GPU — CPU orchestrates binary splitting, GPU handles large multiplies
- **Multiplier Interface**: Strategy pattern allows CPU-only (Mac) or GPU-accelerated (NVIDIA) execution

## Hardware
- **Local dev**: macOS Apple Silicon (CPU-only, no CUDA)
- **GPU target**: NVIDIA GPU via cloud or separate machine

## Non-Goals
- Custom NTT kernel writing (using cuFFT instead)
- Custom big-integer library (using GMP)
- LAPACK/BLAS (not applicable — those are for linear algebra, not arbitrary-precision arithmetic)
