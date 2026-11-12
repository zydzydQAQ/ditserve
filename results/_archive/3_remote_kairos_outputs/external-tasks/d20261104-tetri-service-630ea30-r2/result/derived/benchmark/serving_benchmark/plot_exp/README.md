# TetriServe Benchmark Plotting Module (Simplified)

A streamlined plotting module for analyzing TetriServe benchmark results, focusing on SLO analysis and performance over time.

## Features

This simplified version includes **3 core visualization types**:

1. **SLO Meet Ratio Analysis** - SLO compliance analysis
   - Per-rate plots showing SLO meet ratio at each arrival rate
   - Merged average comparison across all SLO configurations
   - Bar and line chart variants

2. **Spider SLO Ratio Analysis** - Per-resolution SLO compliance patterns
   - Polar/radar charts showing SLO meet ratio by resolution (256x256, 512x512, 1024x1024, 2048x2048)
   - Separate plots for each arrival rate
   - Visualize how resolution complexity affects SLO achievement

3. **Performance Over Time Analysis** - Temporal behavior (Clockwork Fig. 6 style)
   - Latency trends over time for each scheduling method
   - Goodput trends with sliding windows
   - Cumulative goodput analysis
   - Supports both raw data and aggregate calculation approaches

## Installation

The module is designed to be used within the TetriServe project. No additional installation required.

## Usage

### Command Line

```bash
# Basic usage - generates all plots
uv run --no-sync python -m benchmark.serving_benchmark.plot_exp.main results.json

# With custom output directory
uv run --no-sync python -m benchmark.serving_benchmark.plot_exp.main results.json --output-dir my_analysis

# Exclude specific schedulers
uv run --no-sync python -m benchmark.serving_benchmark.plot_exp.main results.json --exclude-schedules fixed_sp_8

# Custom figure settings
uv run --no-sync python -m benchmark.serving_benchmark.plot_exp.main results.json --figsize 12 8 --dpi 300

# Generate only specific plot types
uv run --no-sync python -m benchmark.serving_benchmark.plot_exp.main results.json --plot-types slo_meet spider_slo
uv run --no-sync python -m benchmark.serving_benchmark.plot_exp.main results.json --plot-types performance_time
```

### Python API

```python
from benchmark.serving_benchmark.plot_exp import ExperimentAnalyzer, PlotConfig

# Initialize analyzer
analyzer = ExperimentAnalyzer("results.json")

# Run complete analysis (all 3 plot types)
analyzer.run_analysis("output_dir")

# Or run individual analyses
plot_config = PlotConfig(figsize=(12, 8), dpi=300, use_academic_style=True)
analyzer.plot_slo_meet_ratio_analysis("slo_meet_output", plot_config)
analyzer.plot_spider_slo_ratio_analysis("spider_slo_output", plot_config)
analyzer.plot_performance_over_time_analysis("performance_time_output", plot_config)
```

## Output Structure

```
output_dir/
├── slo_meet_ratio_analysis/
│   ├── *_per_rate_*.png              - Bar charts of SLO meet ratio vs rate for each SLO
│   ├── *_merged_avg_*.png            - Grouped bar chart comparing average SLO meet ratio
│   └── *_merged_avg_line_*.png       - Line chart comparing average SLO meet ratio
│
├── spider_slo_ratio_analysis/
│   ├── *_spider_slo_ratio_by_resolution_*.png  - SLO ratio by resolution for each rate/SLO
│   ├── *_spider_slo_ratio_summary_by_rate_*.png  - Summary plots (2x2 layout)
│   └── *_spider_slo_ratio_summary_by_rate_*.pdf  - Summary plots (PDF)
│
└── performance_over_time/
    ├── raw_data_calculation/           (uses per-request SLOs for temporal variations)
    │   ├── latency_over_time/
    │   │   └── *_latency_combined_*.png              - Combined latency plots
    │   ├── goodput_sliding_window/
    │   │   ├── *_goodput_sliding_window_combined_*.png  - Sliding window goodput
    │   │   └── *_goodput_sliding_window_combined_*.pdf
    │   ├── goodput_cumulative_time/
    │   │   └── *_goodput_cumulative_time_combined_*.png - Cumulative goodput by time
    │   └── goodput_cumulative_requests/
    │       └── *_goodput_cumulative_requests_combined_*.png - Cumulative goodput by requests
    └── aggregate_slo_meet_ratio/       (uses constant aggregate goodput)
        └── (same structure, shows flat-line constant goodput)

Note: Performance over time plots are only generated when real per-request timing data is available.
```

## Data Requirements and Calculation Approaches

### SLO Meet Ratio & Spider SLO Ratio Plots
These plots use **aggregate statistics only** (verified to match per-request calculations):
- Uses `stats.slo_meet_ratio` directly from benchmark results
- No recalculation needed
- Simpler and faster

### Performance Over Time Plots
These plots support **two calculation approaches**:

**1. Raw Data Calculation (shows temporal variations):**
- Recalculates SLO meet ratio for each time window using per-request SLOs
- Shows how goodput varies over time (not constant)
- Requires: `per_request_info`, `request_starts`, `request_finishes`

**2. Aggregate SLO Meet Ratio (shows average behavior):**
- Uses constant aggregate goodput value
- Shows flat line (aggregate performance)
- Useful for comparison with varying approach

**Behavior when per-request data is missing:**
- Performance over time plots are skipped entirely
- Other plots continue using aggregate statistics

## Configuration Options

### PlotConfig Parameters
- `figsize`: Tuple of (width, height) in inches
- `show_reference_line`: Show 95% reference line in SLO plots
- `reference_value`: Value for reference line (default: 0.95)
- `dpi`: Resolution for saved figures (default: 300)
- `color_scheme`: Color scheme - 'academic' for paper style or matplotlib colormap name
- `use_academic_style`: Use academic paper styling (default: True)

### Supported Schedulers
- `dyn_slo_schedule_global` → "FluidSP(ours)"
- `data_parallel` → "xDiT(SP=1)"
- `fixed_sp_2` → "xDiT(SP=2)"
- `fixed_sp_4` → "xDiT(SP=4)"
- `fixed_sp_8` → "xDiT(SP=8)"
- `per_resolution_schedule` → Custom resolution-based scheduler

## Academic Paper Style

The module includes professional styling suitable for academic publications:

### Color Scheme
- **FluidSP**: Orange (#FF8C00) - Highlights the main contribution
- **xDiT(SP=1)**: Light Blue (#87CEEB) - First comparison baseline
- **Other baselines**: Various shades of gray - Additional comparisons

### Visual Elements
- Clean white background with minimal grid lines
- Dotted horizontal grid lines for readability
- No excessive decorations or 3D effects
- Legend placed outside the plot area (right side) for cleaner visualization
- Professional fonts (Arial/DejaVu Sans)

## Required Data in Benchmark Results

For full functionality, your benchmark results should include:

```json
{
  "config": { ... },
  "stats": {
    "num_requests": 300,
    "throughput": 0.15,
    "latency_mean": 8.01,
    "latency_std": 9.65,
    "latency_p90": 24.62,
    "latency_p99": 24.79,
    "latency": [...],                    // Full latency array
    "request_rate": 0.15,
    "slo_meet_ratio": 0.85,

    // For raw data calculation (critical!)
    "per_request_info": [{
      "req_idx": 0,
      "height": 1024,
      "width": 1024,
      "slo": 3.0,
      "latency": 2.5,
      "slo_met": true,
      ...
    }, ...],
    "request_starts": [...],            // Request start times
    "request_finishes": [...]           // Request completion times
  }
}
```

## Troubleshooting

### No SLO Config Data
Ensure your benchmark results include `slo_config` field in the configuration.

### Missing Per-Request Data
If `per_request_info` is missing, plots will fall back to aggregate calculation approach or use mock data for spider plots.

### Empty Plots
Check that your results file contains valid data and isn't filtered out by exclusion rules.

## Example

```bash
# Generate plots for baseline benchmark
uv run --no-sync python -m benchmark.serving_benchmark.plot_exp.main \
    benchmark/serving_benchmark/results/benchmark_flux_H100_baseline/benchmark_results.json \
    --output-dir benchmark/serving_benchmark/results/benchmark_flux_H100_baseline/plots \
    --dpi 300
```

This will create three subdirectories with all plots, ready for inclusion in papers or presentations!
