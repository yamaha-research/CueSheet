# CueSheet demo — containerised standalone.
#
#   docker build -t cuesheet -f demo/Dockerfile .
#   docker run --rm -p 8001:8001 cuesheet
#   open http://localhost:8001/cuesheet
#
# The CueSheet demo (preloaded gallery, file upload, live-mic streaming)
# runs fully in the container. The EfficientAT encoder weights download
# on the first inference, so the container needs network access on first
# use.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv: fast, lockfile-pinned dependency installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY . .

# Vendored EfficientAT encoder (cloned, not committed to the repo).
RUN if [ ! -d vendor/EfficientAT ]; then \
        mkdir -p vendor && \
        git clone --depth 1 https://github.com/fschmid56/EfficientAT.git \
            vendor/EfficientAT; \
    fi

RUN uv sync --frozen --no-dev

EXPOSE 8001
CMD ["uv", "run", "uvicorn", "demo.server:app", \
     "--host", "0.0.0.0", "--port", "8001"]
