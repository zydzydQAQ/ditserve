# linux/amd64, Python 3.10.20, Debian bookworm; official multi-platform manifest.
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
COPY configs/artifacts-runtime.json /tmp/artifacts-runtime.json
COPY docker/install_dependencies.py /tmp/install_dependencies.py
RUN python /tmp/install_dependencies.py /opt/venv /tmp/artifacts-runtime.json

FROM runtime-system AS core-builder
RUN apt-get update && apt-get install -y --no-install-recommends cmake make libssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build/source
COPY CMakeLists.txt ./
COPY cmd ./cmd
COPY include ./include
COPY src ./src
COPY tests/core ./tests/core
COPY tests/fixtures ./tests/fixtures
# Upstream v3.11.3 header, verified by CMake against its pinned SHA256.
COPY docker/vendor/json.hpp ./.cache/deps/nlohmann3.11.3/nlohmann/json.hpp
RUN cmake -S . -B /build/core -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    && cmake --build /build/core --parallel 2 \
    && ctest --test-dir /build/core --output-on-failure

FROM runtime-system AS runtime-base
COPY --from=dependencies /opt/venv /opt/venv
COPY --from=core-builder /build/core/ditserve-core /usr/local/bin/ditserve-core
ENV PATH=/opt/venv/bin:$PATH \
    DITSERVE_CORE=/usr/local/bin/ditserve-core \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/data/cache/huggingface TORCH_HOME=/models/torch \
    TORCHINDUCTOR_CACHE_DIR=/data/cache/torchinductor \
    TRITON_CACHE_DIR=/data/cache/triton \
    MPLCONFIGDIR=/data/cache/matplotlib \
    HOME=/data USER=ditserve LOGNAME=ditserve \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /opt/ditserve
COPY . .
RUN python -m pip install --no-deps --no-build-isolation --no-compile -e . \
    && python -m pip check \
    && ditserve --help \
    && mkdir -p /data/results /data/cache /models \
    && rm -rf build python/*.egg-info \
    && chmod -R a+rX /opt/ditserve
CMD ["ditserve", "--help"]

FROM runtime-system AS longlive-dependencies
ARG DITSERVE_WHEELHOUSE_URL=
COPY configs/artifacts-longlive.json /tmp/artifacts-longlive.json
COPY docker/install_dependencies.py /tmp/install_dependencies.py
RUN python /tmp/install_dependencies.py /opt/longlive-venv /tmp/artifacts-longlive.json

FROM runtime-system AS longlive-source
ARG LONGLIVE_GIT_PROXY=
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY docker/checkout_longlive.py /tmp/checkout_longlive.py
RUN python /tmp/checkout_longlive.py /opt/longlive-source --proxy "${LONGLIVE_GIT_PROXY:-${HTTPS_PROXY:-}}" \
    && rm /tmp/checkout_longlive.py

# Optional target: a separate venv, never appended to the main interpreter's path.
FROM runtime-base AS longlive
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=longlive-dependencies /opt/longlive-venv /opt/longlive-venv
COPY --from=longlive-source /opt/longlive-source /opt/longlive-source
RUN git config --system --add safe.directory /opt/longlive-source \
    && /opt/venv/bin/python -m pip check \
    && /opt/longlive-venv/bin/python -m pip check \
    && /opt/longlive-venv/bin/python -c "import torch, flash_attn, cv2; assert torch.__version__ == '2.8.0+cu126'"

# Keep the last target as the minimal default; LongLive is opt-in via --target longlive.
FROM runtime-base AS runtime
