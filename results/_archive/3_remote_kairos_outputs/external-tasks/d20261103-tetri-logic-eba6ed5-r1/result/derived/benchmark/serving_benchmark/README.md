## Benchmark Serving

Run end-to-end serving benchmarks comparing scheduling strategies under different workloads.
For full documentation of the benchmark system, see [doc/benchmark.md](../../doc/benchmark.md).

### Quick Reference

| Scenario | Config | Description |
|----------|--------|-------------|
| Baseline | `benchmark_flux_H100_baseline.yaml` | data_parallel, fixed_sp_2/4/8, per_resolution |
| TetriServe DP | `benchmark_flux_H100_tetris_dp.yaml` | dyn_slo_schedule_global_dp (recommended) |
| TetriServe Exhaustive | `benchmark_flux_H100_tetris_exhaustive.yaml` | dyn_slo_schedule_exhaustive_optimal |
| Ablation | `benchmark_flux_H100_ablation.yaml` | Ablation study configs |
| SD3 Baseline (A40) | `benchmark_sd3_A40_baseline.yaml` | SD3 baselines on A40 GPUs |
| SD3 TetriServe (A40) | `benchmark_sd3_A40_tetris_dp.yaml` | SD3 TetriServe on A40 GPUs |
| SD3 Baseline (L40) | `benchmark_sd3_L40_baseline.yaml` | SD3 baselines on L40 GPUs |
| FLUX Baseline (L40) | `benchmark_flux_L40_baseline.yaml` | FLUX baselines on L40 GPUs |

Estimated runtimes: Baseline < 15 hours, Exhaustive < 12 hours, DP < 12 hours.

### FLUX Benchmarks (H100)

#### 1. Run baseline strategies

```bash
export GPU_NAME="H100_HBM3"
uv run --no-sync python -m benchmark.serving_benchmark.benchmark_serving \
  --config benchmark/serving_benchmark/configs/benchmark_flux_H100_baseline.yaml \
  --output-dir results/benchmark_flux_H100_baseline
```

#### 2. Run TetriServe (DP variant)

```bash
export GPU_NAME="H100_HBM3"
uv run --no-sync python -m benchmark.serving_benchmark.benchmark_serving \
  --config benchmark/serving_benchmark/configs/benchmark_flux_H100_tetris_dp.yaml \
  --output-dir results/benchmark_flux_H100_tetris_dp
```

#### 3. Post-process with SLO configs

```bash
uv run --no-sync python benchmark/serving_benchmark/analyze_slo_from_config.py \
  results/benchmark_flux_H100_baseline/benchmark_results.json \
  -o results/benchmark_flux_H100_baseline/benchmark_results_all_slos.json \
  --slo-configs custom_1_1.0x_slo_H100_flux \
                custom_1_1.1x_slo_H100_flux \
                custom_1_1.2x_slo_H100_flux \
                custom_1_1.3x_slo_H100_flux \
                custom_1_1.4x_slo_H100_flux \
                custom_1_1.5x_slo_H100_flux
```

#### 4. Merge results and generate plots

```bash
# Merge baseline + TetriServe results
uv run --no-sync python benchmark/serving_benchmark/json_merger.py \
  -i results/benchmark_flux_H100_baseline/benchmark_results_all_slos.json \
     results/benchmark_flux_H100_tetris_dp/benchmark_results.json \
  -o results/combined/benchmark_results_all_slos.json

# Generate publication-quality plots
uv run --no-sync python -m benchmark.serving_benchmark.plot_exp.main \
  results/combined/benchmark_results_all_slos.json \
  --output-dir results/combined/plots \
  --dpi 600 --plot-types academic --figsize 14 10
```

### SD3 Benchmarks (A40)

```bash
export GPU_NAME="A40"

# Baseline
uv run --no-sync python -m benchmark.serving_benchmark.benchmark_serving \
  --config benchmark/serving_benchmark/configs/benchmark_sd3_A40_baseline.yaml \
  --output-dir results/benchmark_sd3_A40_baseline

# TetriServe DP
uv run --no-sync python -m benchmark.serving_benchmark.benchmark_serving \
  --config benchmark/serving_benchmark/configs/benchmark_sd3_A40_tetris_dp.yaml \
  --output-dir results/benchmark_sd3_A40_tetris_dp
```

### Fake Engine Mode

Add `--fake-engine` to skip real GPU inference (uses profiled timing data instead).
Useful for testing scheduling strategies without GPU resources:

```bash
uv run --no-sync python -m benchmark.serving_benchmark.benchmark_serving \
  --config benchmark/serving_benchmark/configs/benchmark_flux_H100_baseline.yaml \
  --output-dir results/fake_engine_test --fake-engine
```
