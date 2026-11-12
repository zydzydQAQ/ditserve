# Contributing to TetriServe

Thanks for your interest in contributing! Here's everything you need to get started.

## Project Structure

- `tetriserve/` — core library (scheduler, runner, server, globals)
- `benchmark/` — SLO attainment benchmark suite
- `tests/` — unit and integration tests
- `doc/` — architecture and API documentation

See [doc/architecture.md](doc/architecture.md) for a full system overview.

## Development Setup

**Requirements:** Python 3.10+, CUDA 12.4+, 2+ NVIDIA GPUs

```bash
git clone https://github.com/DiT-Serving/TetriServe && cd TetriServe

uv venv .venv --python 3.10
uv pip install -p .venv torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -p .venv -e ".[diffusers]"
uv pip install -p .venv --no-build-isolation "flash-attn==2.6.3"

# Verify
python -c "import tetriserve; print('import OK')"
```

## Code Style

- Python 3.10+, formatted with `black` (line length 88)
- Type hints encouraged for public APIs
- Pre-commit hooks configured — run `pre-commit install` after cloning

## Key Conventions

- **Multi-GPU**: code uses `torch.distributed` with NCCL; GPU group setup is in `tetriserve/globals.py`
- **Scheduler**: scheduling logic lives in `tetriserve/scheduler/`; the primary scheduler is `dyn_slo_schedule_global_dp`
- **Latent transfer**: P2P latent handoff between GPU groups is in `tetriserve/runner.py`; `skip_latent_transfer=True` uses dummy tensors (valid for scheduler benchmarking)
- **Benchmark configs**: YAML files under `benchmark/serving_benchmark/configs/` — one baseline and one TetriServe config per hardware target

## Testing

```bash
# Unit tests (no GPU required)
python -m pytest tests/unit/ -q

# Full tests (requires GPU)
CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/ -q
```

Run tests before submitting a PR.

## Commit Style

- One logical change per commit
- Format: `component: brief description` (e.g. `scheduler: fix SLO window boundary condition`)

## Submitting a PR

1. Fork the repo and create a branch from `main`
2. Make your changes with tests where applicable
3. Ensure `python -c "import tetriserve"` and `pytest` pass
4. Open a PR with a clear description of what changed and why
