# TetriServe Workload Package

Clean, simple workload generation for benchmarking - no globals, no caching, just straightforward code.

---

## 🚀 Quick Start

```python
from tetriserve.workload import (
    WorkloadBuilder,
    PoissonProcess,
    ShapeDistribution,
    FixedSLOPolicy,
    plot_all,
    print_workload_summary,
)

# Create workload
workload = (
    WorkloadBuilder()
    .with_model("sd3")
    .with_arrival_process(PoissonProcess(rate=1.0))
    .with_duration(60)
    .with_shape_distribution(ShapeDistribution.evenly())
    .with_slo_policy(FixedSLOPolicy(5.0))
    .build()
)

# Visualize
print_workload_summary(workload)
plot_all(workload, output_path="workload.png", show=True)
```

---

## 📦 What's Included

### Core Classes
- **`Request`**: Single inference request
- **`Workload`**: Collection of requests with arrival times

### Arrival Processes
- **`PoissonProcess(rate)`**: Random arrivals (CV=1.0)
- **`GammaProcess(rate, cv)`**: Bursty (CV>1) or regular (CV<1)
- **`DeterministicProcess(rate)`**: Fixed intervals (CV=0)

### Distributions
- **`ShapeDistribution`**: Image size distributions
  - `.evenly()` - All sizes equal
  - `.short_seqlen_high()` - Favor small images
  - `.long_seqlen_high()` - Favor large images
- **`BatchSizeDistribution`**: Batch size distributions
  - `.single_batch(n)` - Fixed batch size
  - `.uniform([1, 2, 4])` - Uniform over sizes

### SLO Policies
- **`FixedSLOPolicy(slo)`**: Same SLO for all requests
- **`BaselineSLOPolicy(data, multiplier)`**: Based on performance model
- **`load_slo_policy(name, model, steps)`**: Load from config file

### Utilities
- **`WorkloadBuilder`**: Fluent API for creating workloads
- **`compute_statistics(workload)`**: Compute metrics after execution
- **`plot_all(workload)`**: Visualize workload characteristics
- **`print_workload_summary(workload)`**: Text summary

---

## 📊 Visualization

The package includes visualization to understand your workload:

```python
from tetriserve.workload import plot_all, print_workload_summary

# Text summary
print_workload_summary(workload)

# Visual analysis (4 subplots in one figure)
plot_all(workload, output_path="analysis.png", show=True)
```

**What you see:**
- **Top-left**: Cumulative arrivals (overall rate)
- **Top-right**: Instantaneous rate (**peaks = bursts!**)
- **Bottom-left**: Request shapes over time (color-coded)
- **Bottom-right**: Shape distribution (percentages)

**Requirements**: `uv pip install matplotlib`

---

## ⚙️ Duration vs Num Requests

You can specify workload size in two ways:

**Option 1: Duration** (time-based)
```python
.with_duration(60)  # Generate requests for 60 seconds
# Result: ~60 requests at rate=1.0/s (varies due to randomness)
```

**Option 2: Num Requests** (count-based)
```python
.with_num_requests(100)  # Generate exactly 100 requests
# Result: Exactly 100 requests (duration varies based on rate)
```

**When to use which:**
- Use **duration** when testing for specific time period
- Use **num_requests** when you want exact count (e.g., for statistics)
- Cannot use both together!

---

## 🎯 Common Use Cases

### Simple Test Workload
```python
from tetriserve.workload import create_simple_workload

workload = create_simple_workload("sd3", rate=1.0, duration=60)
```

### Exact Number of Requests
```python
# Generate exactly 100 requests (instead of time-based)
workload = (
    WorkloadBuilder()
    .with_model("sd3")
    .with_arrival_process(PoissonProcess(rate=2.0))
    .with_num_requests(100)  # Exactly 100 requests
    .build()
)
```

### Custom Configuration
```python
workload = (
    WorkloadBuilder()
    .with_model("flux")
    .with_arrival_process(GammaProcess(rate=0.8, cv=1.5))  # Bursty
    .with_duration(120)  # Or use .with_num_requests(200)
    .with_shape_distribution(ShapeDistribution.long_seqlen_high())
    .with_num_steps(50)
    .build()
)
```

### Load SLO from Config
```python
from tetriserve.workload import load_slo_policy

slo_policy = load_slo_policy("custom_1_1.5x_slo", "sd3", steps=20)

workload = (
    WorkloadBuilder()
    .with_model("sd3")
    .with_arrival_process(PoissonProcess(1.0))
    .with_duration(60)
    .with_slo_policy(slo_policy)
    .build()
)
```

---

## 🧪 Examples & Tests

```bash
# Run examples (9 examples)
uv run --no-sync python tetriserve/workload/examples.py

# Visualize workload
uv run --no-sync python tetriserve/workload/visualize_example.py

# Interactive demos
uv run --no-sync python tetriserve/workload/quick_viz.py

# Run unit tests
uv run --no-sync python tetriserve/workload/test_examples.py
```

---

## 🎓 Understanding Burstiness

**CV (Coefficient of Variation) = Std Dev / Mean** of inter-arrival times

- **CV < 1.0**: Regular, predictable arrivals
- **CV = 1.0**: Poisson process (random, memoryless)
- **CV > 1.0**: Bursty arrivals (clusters of requests)

The instantaneous rate plot makes this visual:
- Flat line = regular
- Moderate peaks = Poisson
- High peaks = bursty

---

## 📚 Module Structure

```
workload/
├── core.py          - Request & Workload classes
├── arrival.py       - Arrival processes
├── distribution.py  - Shape & batch distributions
├── population.py    - Request population
├── slo.py          - SLO management
├── statistics.py   - Statistics computation
├── builder.py      - Builder pattern
└── visualization.py - Plotting functions
```

---

## 🔄 Legacy API

Old workload.py code is preserved as `tetriserve/workload_legacy.py` for backward compatibility.

```python
# Old code still works
from tetriserve.workload_legacy import (
    Workload, PoissonProcess, fill_slo, populate_requests
)
```

---

## ✨ Design Principles

- ✅ No global state
- ✅ No hidden caching
- ✅ Immutable configurations
- ✅ Type-safe (complete type hints)
- ✅ Easy to test (dependency injection)
- ✅ Clear and explicit

---

## 📞 Quick Help

**Need more examples?** Run `python tetriserve/workload/examples.py`
**Want visualization?** Run `python tetriserve/workload/quick_viz.py`
**Testing?** Check `test_examples.py`

---

That's it! Clean, simple, no mess. 🎉
