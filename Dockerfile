# Lean runtime: C++ controller + resident Python workers on the locked torch 2.10.0+cu126 stack.
# The image is GPU-architecture neutral (no compiled CUDA kernels; torch wheels carry sm_80/sm_86/sm_90).
# Build and run instructions, including an offline wheelhouse build: MIGRATION.md.
ARG PYTHON_BASE=python:3.10.20-slim-bookworm@sha256:019e31cc1f52e4139d338bca44f27b71339a9da261738fd80fab9cf4595152fc
FROM ${PYTHON_BASE} AS runtime-system
# GCC/G++ and /usr/local/include/python3.10 are required by torch.compile.
# CUDA libraries come from the locked PyTorch wheels; no CUDA SDK/nvcc is needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gcc g++ libgomp1 libssl3 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

FROM runtime-system AS dependencies
ARG DITSERVE_WHEELHOUSE_URL=
COPY docker/artifacts-runtime.json /tmp/artifacts-runtime.json
COPY docker/install_dependencies.py /tmp/install_dependencies.py
# Only Triton's static CUPTI archives are pruned; package tests/ stay (numpy.testing imports numpy._core.tests).
RUN python /tmp/install_dependencies.py /opt/venv /tmp/artifacts-runtime.json \
    && rm -f /opt/venv/lib/python3.10/site-packages/triton/backends/nvidia/lib/cupti/*.a \
    && /opt/venv/bin/python -m pip check

FROM runtime-system AS core-builder
RUN apt-get update && apt-get install -y --no-install-recommends cmake make libssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build/source
COPY . .
RUN cmake -S . -B /build/core -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    && cmake --build /build/core --parallel 4 \
    && ctest --test-dir /build/core --output-on-failure \
    && cmake --install /build/core --prefix /opt/ditserve --strip

FROM runtime-system AS runtime
ARG GIT_REVISION=unknown
LABEL org.opencontainers.image.title=ditserve org.opencontainers.image.revision=${GIT_REVISION}
COPY --from=dependencies /opt/venv /opt/venv
COPY --from=core-builder /opt/ditserve /opt/ditserve
# DITSERVE_ROOT holds results/, .state/ and .cache/results-work/; mount it to keep them.
ENV PATH=/opt/ditserve/bin:/opt/venv/bin:$PATH \
    DITSERVE_PYTHON=/opt/venv/bin/python DITSERVE_ROOT=/data \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/data/cache/huggingface TORCH_HOME=/models/torch \
    TORCHINDUCTOR_CACHE_DIR=/data/cache/torchinductor \
    TRITON_CACHE_DIR=/data/cache/triton \
    DITSERVE_CORE=/opt/ditserve/bin/ditserve \
    HOME=/data USER=ditserve LOGNAME=ditserve \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN ditserve --version && ditserve doctor \
    && mkdir -p /data/results /data/cache /models && chmod -R a+rwX /data
WORKDIR /data
CMD ["ditserve", "--help"]
