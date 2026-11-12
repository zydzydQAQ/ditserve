# TetriServe API Reference

## HTTP Endpoints

### POST `/v1/generate`

Generate images from text prompts.

**Request body** (JSON):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | `string` or `string[]` | (required) | Text prompt(s). String for single image, array for batch. |
| `height` | `int` | 1024 | Image height in pixels. Must be divisible by 8. |
| `width` | `int` | 1024 | Image width in pixels. Must be divisible by 8. |
| `num_inference_steps` | `int` | 20 | Number of denoising steps. More steps = higher quality, slower. |
| `guidance_scale` | `float` | 0.0 | Classifier-free guidance scale. 0.0 for FLUX, 7.0 typical for SD3. |
| `seed` | `int` or `null` | 42 | Random seed for reproducibility. |
| `output_type` | `string` | `"pil"` | `"pil"` for base64 PNG images, `"latent"` for raw latent tensors. |
| `batch_size` | `int` | 1 | Number of images to generate. Must match prompt count if array. |
| `latency_threshold` | `float` or `null` | `inf` | SLO deadline in seconds. Scheduler allocates GPUs to meet this. |

**Response** (JSON):

```json
{
  "images": ["<base64-encoded PNG>", ...]
}
```

**Example**:

```bash
curl -X POST http://localhost:1037/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A sunset over mountains",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 20,
    "latency_threshold": 10.0
  }'
```

### GET `/health`

Health check endpoint.

**Response**: HTTP 200 if server is running.

---

## Server Launch

```bash
uv run --no-sync python -m tetriserve.server.launcher [OPTIONS]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--nnodes` | 1 | Number of nodes (only 1 supported) |
| `--nproc_per_node` | 1 | Number of GPU workers |
| `--master_addr` | 127.0.0.1 | Master address for distributed init |
| `--master_port` | 1037 | API server port and distributed rendezvous port |
| `--schedule_logic` | naive | Scheduling algorithm (see below) |
| `--fake_engine` | false | Run without loading models (for testing) |
| `--output_type` | latent | Default output type (`pil` or `latent`) |
| `--model` | — | HuggingFace model path or name |
| `--perf_model_dir` | log/benchmark/scaling_efficiency/ | Path to profiled scaling data |
| `--enable_tracer` | false | Enable Chrome trace output |

### Scheduling Algorithms (`--schedule_logic`)

**Baselines:**
- `data_parallel` — One GPU per request, round-robin assignment
- `fixed_sp_2` / `fixed_sp_4` / `fixed_sp_8` — Fixed sequence parallelism degree

**Production (SLO-aware):**
- `fcfs_optimal_shape` — FCFS with cost-optimal SP degree per request
- `dyn_slo_schedule_global` — Dynamic SLO-aware GPU allocation
- `dyn_slo_schedule_global_scale_up` — Dynamic with scale-up heuristic
- `dyn_slo_schedule_global_dp` — Dynamic SLO-aware with DP fallback **(recommended)**

**Experimental:**
- `dyn_sp_schedule` — Dynamic SP without SLO awareness
- `per_resolution_schedule` — Fixed SP per resolution tier
- `per_resolution_schedule_1_1_4_8` — 512→SP1, 1024→SP1, 1536→SP4, 2048→SP8
- `per_resolution_schedule_1_1_2_8` — 512→SP1, 1024→SP1, 1536→SP2, 2048→SP8

---

## Troubleshooting

### Common Errors

**`Address already in use` on launch**
Another process is using the port. Either kill it or use a different `--master_port`.

**`NCCL error: unhandled cuda error`**
Check `CUDA_VISIBLE_DEVICES` is set correctly. Ensure all GPUs are accessible.

**`FileNotFoundError: scaling efficiency data not found`**
Required for SLO-aware schedulers. Generate with:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync python -m benchmark.serving_benchmark.profile_scaling_efficiency
```

**`torch.cuda.OutOfMemoryError`**
Reduce `--nproc_per_node`, use smaller images, or use `fixed_sp_2`+ to distribute across GPUs.

**Images look wrong / low quality**
- Check `num_inference_steps` (20+ for SD3, 4+ for FLUX)
- Verify model path is correct
- Compare against diffusers baseline: `uv run --no-sync pytest tests/test_sd3_single_gpu_vs_baseline.py`

**Requests timing out**
- Increase `latency_threshold` in request
- Use `dyn_slo_schedule_global_dp` for best SLO attainment
- Check GPU utilization with `nvidia-smi`
