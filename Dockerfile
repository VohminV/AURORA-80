# CPU image by default. For CUDA, pass:
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
# and run the container with --gpus all.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY aurora ./aurora

RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install ".[tokenizer]"

COPY configs ./configs
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 aurora \
    && mkdir -p /workspace/data /workspace/runs /workspace/reports \
    && chown -R aurora:aurora /workspace

USER aurora

ENTRYPOINT ["aurora-train"]
CMD ["--help"]
