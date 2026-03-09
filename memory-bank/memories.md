# Memories

## Novel Patterns & Insights
(Updated as implementation progresses)

- **NTT base choice**: Base 2^24 is the sweet spot for cuFFT-based NTT multiplication. Products of two base-2^24 digits (2^48) fit within the 53-bit mantissa of IEEE 754 double-precision floats, avoiding precision loss in the frequency domain.

- **Binary splitting memory pattern**: The Chudnovsky binary splitting tree can be evaluated with O(log N) stack depth, keeping memory bounded even for billions of terms. The large intermediate products are the memory bottleneck, not the recursion.

- **GPU crossover point**: GPU NTT multiplication typically becomes faster than GMP's CPU multiplication at around 100K–500K digits per operand, depending on GPU model and PCIe bandwidth. Below this threshold, CPU is faster due to transfer overhead.
