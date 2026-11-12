# Benchmark System

## Overview

Comprehensive benchmarking framework for evaluating diffusion model serving systems across scheduling strategies, workloads, and SLO configurations.

## Purpose

- Compare scheduling algorithms (data parallel, fixed SP, dynamic SLO-aware)
- Evaluate SLO compliance under varying arrival rates and workload distributions
- Generate publication-quality plots for research

## Architecture

```
benchmark_serving.py       # Main orchestrator (720 lines)
    ↓
├─ Load YAML config
├─ Generate experiment combinations
├─ For each combination:
│  ├─ Create runtime (GPUs, parallelism)
│  ├─ Build workload (requests with SLOs)
│  ├─ Execute async workload
│  ├─ Compute statistics (latency, SLO ratio)
│  └─ Save results
└─ Output: benchmark_results.json

analyze_slo_from_config.py # Post-process with multiple SLO configs
    ↓
└─ Output: benchmark_results_all_slos.json

json_merger.py             # Combine multiple result files
    ↓
└─ Output: merged_results.json

plot_exp/                  # Analysis & visualization
    ├─ main.py            # CLI entry
    ├─ analyzer.py        # Orchestrator
    ├─ data_processor.py  # JSON → DataFrame
    └─ Plotting modules:
       ├─ slo_meet_ratio.py        # Bar charts
       ├─ spider_slo_ratio.py      # Radar charts by resolution
       ├─ performance_over_time.py # Temporal latency/goodput
       └─ academic_plot.py         # Publication-quality figures
```

## Key Components

### 1. benchmark_serving.py
Main entry point for running experiments.

**Key functions:**
- `run_experiment()`: Single experiment execution
- `run_all_experiments()`: Generate all combinations from config
- `run_workload()`: Async request execution
- **Resume mode**: Skip completed experiments

### 2. analyze_slo_from_config.py
Post-processes results with multiple SLO configurations.

**Features:**
- Supports multiplier-based ("1.5x") or fixed SLOs
- Per-shape SLO mappings: (batch_size, height, width) → SLO
- 7% tolerance: latency <= target_slo × 1.07

### 3. json_merger.py
Combines multiple benchmark result files for comparative analysis.

### 4. plot_exp/
Visualization and analysis suite.

**Components:**
- **data_processor.py**: Loads JSON → Pandas DataFrame
- **analyzer.py**: ExperimentAnalyzer orchestrates all plots
- **config.py**: Plot styling (Arial fonts, academic colors)

**Plot types:**
- **SLO meet ratio**: Bar charts showing compliance across configs
- **Spider charts**: Radar plots by resolution (256/512/1024/2048)
- **Performance over time**: Latency and goodput trends (Clockwork Fig 6 style)
- **Academic figures**: Publication-ready with consistent styling

## Configuration

### Benchmark Config (YAML)

Example: `benchmark/serving_benchmark/configs/benchmark_flux_H100_baseline.yaml`

```yaml
# Model and scheduling strategies to compare
models: [flux]                          # "flux" or "sd3"
schedule_logics: [data_parallel, fixed_sp_2, fixed_sp_4, fixed_sp_8, per_resolution_schedule]
gpu_nums: [8]                           # Total GPUs available

# Workload parameters (all combinations are tested)
distributions: [gamma]                  # Arrival process: gamma, poisson, deterministic
rates: [0.2]                            # Arrival rate (requests/sec)
cvs: [1.0]                              # Coefficient of variation (gamma only)
steps: [20]                             # Inference steps per request
shape_dists: [evenly, long_seqlen_high] # Resolution distribution
request_nums: [300]                     # Total requests per experiment
seed: 42

# SLO configuration
slo_configs: [custom_1_1.0x_slo_H100_flux]  # References JSON file in slo_configs/

# Runtime options
output_types: [latent]       # "latent" (fast, no decode) or "pil" (full pipeline)
fake_engine: false           # true = skip real inference, use profiled timing
resume: true                 # Skip already-completed experiments
enable_tracer: true          # Generate Chrome trace files
enable_warmup: false         # Warmup runs before measurement
use_cuda_graph: false
use_torch_compile: false
save_logs: true
no_timestamp: true
save_interval: 1
```

### SLO Config (JSON)
```json
{
  "(1, 256, 256)": 1.5,
  "(1, 512, 512)": 2,
  "(1, 1024, 1024)": 3,
  "(1, 2048, 2048)": 5
}
```

Or multiplier format:
```json
{
  "(1, 256, 256)": "1.5x",
  "(1, 512, 512)": "2.0x"
}
```

## Workflow

### Complete Benchmark Pipeline
```bash
# 1. Run benchmark
uv run --no-sync python -m benchmark.serving_benchmark.benchmark_serving \
  --config configs/benchmark_flux_H100_baseline.yaml \
  --output-dir results/baseline

# 2. Post-process with multiple SLO configs
uv run --no-sync python benchmark/serving_benchmark/analyze_slo_from_config.py \
  results/baseline/benchmark_results.json \
  -o results/baseline/benchmark_results_all_slos.json \
  --slo-configs custom_1_1.0x_slo_H100_flux \
                custom_1_1.1x_slo_H100_flux

# 3. Generate plots
uv run --no-sync python -m benchmark.serving_benchmark.plot_exp.main \
  results/baseline/benchmark_results_all_slos.json \
  --output-dir results/baseline/plots \
  --dpi 600
```

### Comparative Analysis
```bash
# Merge baseline + optimized results
uv run --no-sync python benchmark/serving_benchmark/json_merger.py \
  -i results/baseline/benchmark_results_all_slos.json \
     results/optimized/benchmark_results_all_slos.json \
  -o results/combined_results.json

# Generate comparative plots
uv run --no-sync python -m benchmark.serving_benchmark.plot_exp.main \
  results/combined_results.json \
  --output-dir results/plots
```

## Key Metrics

| Metric | Definition | Scope |
|--------|-----------|-------|
| **throughput** | requests / duration | Aggregate |
| **latency_p90/p95/p99** | Percentile latencies | Aggregate |
| **slo_meet_ratio** | requests_meeting_SLO / total | Per-request |
| **request_rate** | Observed arrival rate | Aggregate |
| **shape_distribution** | Histogram of resolutions | Aggregate |

## Output Structure

```
results/
├── benchmark_results.json              # Raw results
├── benchmark_results_all_slos.json     # With all SLO configs
└── plots/
    ├── slo_meet_ratio_analysis/
    │   ├── *_per_rate_*.png           # Per-rate bar charts
    │   ├── *_merged_avg_*.png         # Summary bar chart
    │   └── *_merged_avg_line_*.png    # Line chart variant
    ├── spider_slo_ratio_analysis/
    │   ├── *_spider_slo_ratio_*.png   # Radar charts
    │   └── *_summary_*.png            # 2×2 layout
    └── performance_over_time/
        ├── latency_over_time/         # Temporal latency
        ├── goodput_sliding_window/    # Windowed goodput
        ├── goodput_cumulative_time/   # Cumulative by time
        └── goodput_cumulative_requests/ # Cumulative by count
```

## Sample Output

### benchmark_results.json (per-experiment entry)

```json
{
  "model": "flux",
  "schedule_logic": "dyn_slo_schedule_global_dp",
  "gpu_num": 8,
  "distribution": "gamma",
  "rate": 0.2,
  "cv": 1.0,
  "shape_dist": "evenly",
  "request_num": 300,
  "slo_config": "custom_1_1.0x_slo_H100_flux",
  "throughput": 0.187,
  "latency_p50": 4.21,
  "latency_p90": 7.83,
  "latency_p95": 9.12,
  "latency_p99": 12.45,
  "slo_meet_ratio": 0.847,
  "total_duration": 1604.2,
  "per_request_data": [
    {"req_id": 0, "latency": 3.45, "slo": 5.0, "met_slo": true, "shape": [1, 1024, 1024]},
    {"req_id": 1, "latency": 6.12, "slo": 5.0, "met_slo": false, "shape": [1, 2048, 2048]}
  ]
}
```

### Generated Plots

The plotting system produces several visualization types:
- **SLO meet ratio bar charts**: Compare `slo_meet_ratio` across scheduling strategies at each arrival rate. The key plot for paper figures.
- **Spider/radar charts**: Per-resolution SLO compliance (256/512/1024/2048) showing which strategies handle large vs. small images better.
- **Performance over time**: Temporal latency traces and cumulative goodput curves for understanding system dynamics.

## Features

- **Resume support**: Long-running benchmarks can be interrupted
- **Flexible SLO definitions**: Per-shape, multiplier-based
- **Rich per-request data**: For detailed temporal analysis
- **Academic styling**: Arial fonts, consistent colors
- **Multiple output formats**: PNG, PDF
