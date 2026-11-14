# GPU Inference Latency & Throughput Benchmark

- **Device**: NVIDIA GeForce RTX 3050 Laptop GPU
- **Resolution**: 640 x 640
- **Timing Method**: CUDA Events (`torch.cuda.Event(enable_timing=True)`)
- **Benchmark Protocol**: 100 warm-up runs, 300 timed passes, 5 interleaved repeats per model
- **Reported Metric**: Median ± Standard Deviation across 5 interleaved repeats
- **NMS**: Excluded (raw model forward pass timing only)
- **Train-Only Parameters**: Excluded from inference parameter counts (BoundaryHead discarded at inference)

### Clocks & Thermal State

- **Before**: `210 MHz, 405 MHz, 210 MHz, 4.56 W, 48`
- **After**: `1957 MHz, 6101 MHz, 1957 MHz, 83.37 W, 82`

### Empirical Noise Baseline (E1 vs E4, identical inference architecture)

| Condition | E1 Latency (ms) | E4 Latency (ms) | Absolute Delta (ms) | Relative Delta (%) | FPS Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Batch=1 FP32** | 9.07 | 9.02 | 0.05 | 0.56% | 0.6 |
| **Batch=1 FP16** | 11.08 | 11.07 | 0.01 | 0.06% | 0.1 |
| **Batch=16 FP32** | 60.76 | 60.53 | 0.23 | 0.38% | 1.0 |
| **Batch=16 FP16** | 38.30 | 38.50 | 0.20 | 0.53% | 2.2 |

### Batch = 1 Benchmark

| Stage | Inf Params | GFLOPs | FP32 Latency (ms) | FP32 FPS | FP32 VRAM (MB) | FP16 Latency (ms) | FP16 FPS | FP16 VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | 3,011,433 | 8.1963 | 9.07 ± 0.42 | 110.2 ± 4.7 | 125.2 | 11.08 ± 0.12 | 90.3 ± 1.0 | 78.4 |
| **E2** | 2,926,956 | 12.3626 | 10.58 ± 0.97 | 94.5 ± 7.2 | 145.9 | 13.80 ± 0.41 | 72.5 ± 2.0 | 89.7 |
| **E3** | 2,019,451 | 7.2843 | 9.17 ± 0.93 | 109.1 ± 9.0 | 124.7 | 10.92 ± 0.13 | 91.5 ± 1.1 | 78.4 |
| **E4** | 3,011,433 | 8.1963 | 9.02 ± 1.08 | 110.8 ± 10.2 | 125.2 | 11.07 ± 0.14 | 90.3 ± 1.1 | 78.4 |
| **E7** | 3,073,722 | 8.3391 | 10.99 ± 0.71 | 91.0 ± 5.6 | 124.7 | 12.91 ± 0.22 | 77.5 ± 1.3 | 78.4 |
| **E8** | 3,052,926 | 12.8002 | 13.50 ± 0.28 | 74.1 ± 1.5 | 145.9 | 16.69 ± 0.25 | 59.9 ± 0.9 | 89.7 |
| **E5a** | 3,011,433 | 8.1963 | 9.07 ± 0.42* | 110.2 ± 4.7* | 125.2 | 11.08 ± 0.12* | 90.3 ± 1.0* | 78.4 |
| **E5b** | 3,011,433 | 8.1963 | 9.07 ± 0.42* | 110.2 ± 4.7* | 125.2 | 11.08 ± 0.12* | 90.3 ± 1.0* | 78.4 |
| **E6** | 3,011,433 | 8.1963 | 9.07 ± 0.42* | 110.2 ± 4.7* | 125.2 | 11.08 ± 0.12* | 90.3 ± 1.0* | 78.4 |

### Batch = 16 Benchmark (GPU Compute-Bound)

| Stage | Inf Params | GFLOPs | FP32 Latency (ms) | FP32 FPS | FP32 VRAM (MB) | FP16 Latency (ms) | FP16 FPS | FP16 VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | 3,011,433 | 8.1963 | 60.76 ± 0.51 | 263.3 ± 2.2 | 475.8 | 38.30 ± 0.13 | 417.8 ± 1.4 | 257.2 |
| **E2** | 2,926,956 | 12.3626 | 100.49 ± 0.62 | 159.2 ± 1.0 | 805.6 | 62.63 ± 0.59 | 255.5 ± 2.4 | 421.5 |
| **E3** | 2,019,451 | 7.2843 | 66.21 ± 0.70 | 241.6 ± 2.5 | 475.8 | 38.93 ± 0.54 | 411.0 ± 5.5 | 256.7 |
| **E4** | 3,011,433 | 8.1963 | 60.53 ± 0.57 | 264.3 ± 2.5 | 475.8 | 38.50 ± 0.35 | 415.6 ± 3.7 | 256.2 |
| **E7** | 3,073,722 | 8.3391 | 67.47 ± 0.57 | 237.1 ± 2.0 | 475.8 | 43.21 ± 0.39 | 370.3 ± 3.3 | 256.7 |
| **E8** | 3,052,926 | 12.8002 | 120.00 ± 1.06 | 133.3 ± 1.2 | 805.6 | 73.17 ± 0.59 | 218.7 ± 1.8 | 422.0 |
| **E5a** | 3,011,433 | 8.1963 | 60.76 ± 0.51* | 263.3 ± 2.2* | 475.8 | 38.30 ± 0.13* | 417.8 ± 1.4* | 257.2 |
| **E5b** | 3,011,433 | 8.1963 | 60.76 ± 0.51* | 263.3 ± 2.2* | 475.8 | 38.30 ± 0.13* | 417.8 ± 1.4* | 257.2 |
| **E6** | 3,011,433 | 8.1963 | 60.76 ± 0.51* | 263.3 ± 2.2* | 475.8 | 38.30 ± 0.13* | 417.8 ± 1.4* | 257.2 |

\* E5a/E5b share identical architecture with E1; E6 detector is identical to E1 with post-hoc refinement.
