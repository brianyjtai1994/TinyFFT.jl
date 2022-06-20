# TinyFFT

A small repository for doing one dimensional FFT.

- Perform: `X(k) = ∑(n = 0 → N - 1) x(n) * exp(-2πkn/N)`.
- Output of `fft` and `fft!` is circularly shifted by `fftshift!`.
