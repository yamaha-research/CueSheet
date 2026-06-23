#!/usr/bin/env bash
# CueSheet demo — one-command local launch.
#
#   bash demo/run.sh           # serves http://localhost:8001/cuesheet
#   PORT=9000 bash demo/run.sh # pick another port
#
# Requirements: uv (https://docs.astral.sh/uv/), ffmpeg, git.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8001}"

command -v uv >/dev/null 2>&1 || {
  echo "error: uv not found. Install from https://docs.astral.sh/uv/"; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || {
  echo "warning: ffmpeg not on PATH — audio/video upload decoding will fail."; }

# Vendored EfficientAT encoder (cloned on first run, not committed to the repo).
if [ ! -d vendor/EfficientAT ]; then
  echo "fetching the EfficientAT encoder into vendor/ ..."
  mkdir -p vendor
  git clone --depth 1 https://github.com/fschmid56/EfficientAT.git vendor/EfficientAT
fi

echo "syncing dependencies with uv ..."
uv sync --quiet

echo ""
echo "  CueSheet demo  ->  http://localhost:${PORT}/cuesheet"
echo ""
exec uv run uvicorn demo.server:app --host 0.0.0.0 --port "${PORT}"
