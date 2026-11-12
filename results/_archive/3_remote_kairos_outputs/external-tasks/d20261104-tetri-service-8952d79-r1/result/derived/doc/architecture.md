# TetriServe Architecture

## Overview

TetriServe is a high-performance serving system for diffusion models (FLUX, SD3) with dynamic distributed scheduling. It optimizes GPU allocation to meet Service Level Objectives (SLOs) while maximizing throughput.

## Core Design Philosophy

- **Decoupled scheduling and execution**: Scheduler decides GPU allocation; workers execute independently
- **Dynamic resource allocation**: GPU configuration varies per request based on SLO and characteristics
- **Multi-step inference support**: Intermediate latent tensors transferred between steps
- **Distributed-first**: Built for multi-GPU execution with flexible parallelism

## System Architecture

```
Client Requests
      ↓
┌─────────────────────────────────┐
│   Engine (Main Process)         │
│  - Request Queue                │
│  - Scheduler                    │
│  - Request Tracker              │
└────────┬────────────────────────┘
         ↓ ZMQ Router-Dealer
┌────────────────────────────────────┐
│  Workers (GPU Processes)           │
│  - Model Loading                   │
│  - Inference Execution             │
│  - Latent Management               │
└────────────────────────────────────┘
```

## Key Modules

### 1. Config (`config/`)
- **ServerArgs**: System configuration (nodes, scheduling logic, tracing)
- **Request types**: UnScheduledRequest → ScheduledRequest → ScheduledRequestsBlock
- **DAG-based dependencies**: NetworkX graph for request scheduling

### 2. Workload (`workload/`)
Pure data structures for request generation:
- **Request**: (model, batch_size, height, width, steps, SLO)
- **Arrival processes**: Poisson, Gamma, Deterministic
- **Distributions**: Shape/batch size patterns
- **SLO policies**: Fixed, baseline-based, custom
- **WorkloadBuilder**: Fluent API for workload construction

### 3. Scheduler (`scheduler/`)
GPU allocation strategies (12 algorithms, defined in `config/args.py --schedule_logic`):

**Baselines:**
- `data_parallel` — One GPU per request (DP only)
- `fixed_sp_2` / `fixed_sp_4` / `fixed_sp_8` — Fixed sequence parallelism degree

**Production (SLO-aware):**
- `fcfs_optimal_shape` — FCFS with cost-optimal SP degree per request
- `dyn_slo_schedule_global` — Dynamic SLO-aware GPU allocation
- `dyn_slo_schedule_global_scale_up` — Dynamic with scale-up heuristic
- `dyn_slo_schedule_global_dp` — Dynamic SLO-aware + data parallel fallback (recommended)

**Experimental:**
- `dyn_sp_schedule` — Dynamic SP without SLO awareness
- `per_resolution_schedule` — Fixed SP per resolution tier
- `per_resolution_schedule_1_1_4_8` — Per-resolution: 512→SP1, 1024→SP1, 1536→SP4, 2048→SP8
- `per_resolution_schedule_1_1_2_8` — Per-resolution: 512→SP1, 1024→SP1, 1536→SP2, 2048→SP8

**Supporting module:**
- **E2ECostEstimator**: Predicts execution time from scaling efficiency data

### 4. Distributed (`distributed/`)
Multi-GPU coordination:
- **ProcessGroupSingleton**: Manages NCCL/Gloo process groups
- **Latent transfer groups**: Dedicated inter-rank communication
- **Yunchang kernels**: Ring Flash Attention for distributed attention

### 5. LightRay (`light_ray/`)
Inter-step tensor management:
- **LatentManager**: Per-GPU latent store with background sender thread
- **LatentFuture**: Lazy tensor retrieval (local or distributed)
- **Optimized paths**: Self-communication uses local memory

### 6. xFuser (`xfuser/`)
Core inference engine:
- **Pipelines**: xFuserFluxPipeline, xFuserStableDiffusion3Pipeline
- **Distributed support**: Ulysses/Ring parallelism
- **Config**: Model, runtime, data parallel settings

### 7. Engine (`engine/`)
- **_CServeEngine**: Main orchestrator (ZMQ router, scheduler invocation)
- **TetriServeRunner**: Worker-side execution (model loading, inference)
- **RequestTracker**: Tracks sub-request progress and merges results

## Request Flow

```
1. add_request(InputConfig)
   ↓
2. scheduler.add_request() → Queue

3. schedule_tasks()
   - scheduler.schedule() → List[ScheduledRequestsBlock]
   - Cost estimation
   - GPU allocation
   - Dependency DAG construction
   ↓
4. handle_single_bunch_requests()
   - Send to workers via ZMQ
   - Workers execute inference
   - Return RunnerOutput
   ↓
5. request_tracker.update_requests()
   - Track progress
   - Merge sub-requests
   ↓
6. Return EngineOutput (PIL.Image)
```

## Multi-Step Inference

TetriServe splits long diffusion processes into multiple scheduling rounds. This enables GPU reallocation between steps — the signature TetriServe optimization.

### How It Works

```
Request: 20-step SD3 at 1024x1024
  Round 1: Scheduler assigns steps 0-9 on GPUs [0,1] (SP=2)
           → produces intermediate latent tensor
  Round 2: Scheduler assigns steps 10-19 on GPUs [2,3,4,5] (SP=4)
           → receives latent from round 1, produces final image
```

### Key Components

1. **Step tracking**: Each `ScheduledRequest` has `start_step` and `end_step` fields. The scheduler increments `start_step` as steps complete, creating new scheduling blocks for remaining steps.

2. **LatentFuture** (`light_ray/latent_future.py`): A `ray.future`-style handle that abstracts latent tensor retrieval. When a round needs the previous round's output:
   - **Same GPU**: Direct local memory access (zero-copy)
   - **Different GPU**: `torch.distributed.send/recv` via dedicated latent transfer process groups

3. **LatentManager** (`light_ray/latent_manager.py`): Per-GPU tensor store with a background sender thread. Stores intermediate latents keyed by `sub_req_id` and sends them on demand to requesting ranks.

4. **Dedicated transfer groups**: Latent transfers use separate NCCL process groups (not the broadcast groups used for attention). This prevents deadlocks between inference collectives and latent transfers.

### Data Flow

```
Runner.pre_process_reqs()
  → For each request with start_step > 0:
    1. Create LatentFuture pointing to previous round's output
    2. LatentFuture.get() blocks until tensor arrives (or retrieves locally)
    3. Feed received latent as initial input to denoising loop
  → For each request with start_step == 0:
    1. Run inference_prologue (encode prompts, prepare initial noise)
    2. Start denoising from random noise
```

## Parallelism Support

1. **Sequence Parallelism (SP)**: Split sequence across GPUs (Ulysses + Ring)
2. **Data Parallelism (DP)**: Independent requests on separate GPUs
3. **Distributed Attention**: Ring Flash Attention with all-to-all communication

## Communication Patterns

- **Main**: ZMQ Router-Dealer (scheduler ↔ workers)
- **Latent transfer**: torch.distributed (inter-step tensors)
- **Attention sync**: NCCL collectives (distributed attention)

## Key Features

- **Resume support**: Skip completed experiments in long-running benchmarks
- **Async execution**: Non-blocking request handling
- **SLO-aware**: Dynamic scheduling based on latency requirements
- **Flexible**: 12 scheduling strategies (4 baselines, 4 production, 4 experimental)
- **Traceable**: Optional Chrome trace format for performance analysis
