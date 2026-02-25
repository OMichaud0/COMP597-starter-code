# Run 5 Resource Summary

## Overview
- Iterations: `109,226`
- Total training time: `6,106.38 s` (`1h 41m 46.38s`)
- Carbon intensity used: `40.0 gCO2/kWh`

## Energy and Carbon
- Cumulative GPU energy: `0.243946 kWh` (`243.95 Wh`)
- Cumulative carbon estimate: `9.7578 gCO2e`
- Average step energy: `2.2334021e-06 kWh` (`0.002233 Wh`)
- Average step carbon: `8.9336083e-05 gCO2e`

## Findings
- The run is strongly GPU-driven: average GPU utilization is `95.04%`, with a peak of `97%`. This indicates consistently high GPU occupancy.
- GPU memory pressure remains low relative to capacity: about `4.69 GiB` used out of `32.76 GiB` total, leaving substantial headroom.
- Average GPU power is `180.36 W`, peaking at `185.27 W`, which is consistent with sustained high-throughput execution.
- Total runtime is about `1h 42m`, and average step time is `52.54 ms` with occasional long-tail spikes (peak `458.73 ms`).
- Estimated footprint for this run is modest: `243.95 Wh` energy and `9.76 gCO2e` using the configured carbon intensity (`40 gCO2/kWh`).
- Process CPU usage averages `103.78%` while system-wide CPU remains low (`0.81%`), suggesting no broad CPU bottleneck.
- I/O is negligible (`0 MB` read and very small writes per step), so storage throughput does not appear to limit training.

## Average Metrics
| Metric | Value |
|---|---:|
| GPU utilization | 95.0423 % |
| GPU memory used | 4693.2436 MiB |
| GPU memory total | 32760.0000 MiB |
| GPU power | 180.3599 W |
| GPU energy per step | 8040.2475 mJ |
| Torch CUDA allocated | 2545.8872 MiB |
| Torch CUDA reserved | 3903.9936 MiB |
| System memory used | 14331.5948 MiB |
| System memory total | 515475.1289 MiB |
| System CPU utilization | 0.8063 % |
| Process RSS | 1940.4966 MiB |
| Process VMS | 18177.6532 MiB |
| Process CPU utilization | 103.7841 % |
| Process I/O read | 0.0000 MB |
| Process I/O write | 0.000267 MB |
| Step duration | 52.5446 ms |

## Peak Metrics
| Metric | Value |
|---|---:|
| GPU utilization | 97.0000 % |
| GPU memory used | 4693.2500 MiB |
| GPU power | 185.2660 W |
| Torch CUDA allocated | 2545.8872 MiB |
| Torch CUDA reserved | 3904.0000 MiB |
| System memory used | 14427.4336 MiB |
| Process RSS | 2000.6523 MiB |
| Step duration | 458.7330 ms |

## Source
- Generated from `results/run5_summary.json`
