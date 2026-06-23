#!/usr/bin/env bash
# Fetch every audio recording listed in data/CONCERT-10.tsv into data/raw/<stem>.mp3.
# Idempotent — files that already exist are skipped. Requires yt-dlp + ffmpeg.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v yt-dlp &>/dev/null; then
  echo "error: yt-dlp is not installed."
  echo "install with:  brew install yt-dlp   (macOS)"
  echo "             | sudo apt install yt-dlp   (Linux)"
  exit 1
fi

mkdir -p data/raw

count_new=0
count_skipped=0
while IFS=$'\t' read -r stem url notes; do
  [[ -z "${stem:-}" || "${stem}" == \#* || "${stem}" == "show_id" ]] && continue
  out="data/raw/${stem}.mp3"
  if [[ -f "$out" ]]; then
    echo "skip:     $stem  (already in data/raw/)"
    count_skipped=$((count_skipped + 1))
    continue
  fi
  echo "download: $stem  ←  $url"
  yt-dlp -x --audio-format mp3 --audio-quality 0 --no-playlist \
    -o "data/raw/${stem}.%(ext)s" "$url" \
    --quiet --progress
  count_new=$((count_new + 1))
done < data/CONCERT-10.tsv

echo ""
echo "==========================================================================="
echo "  Downloaded $count_new new file(s); skipped $count_skipped already present."
echo "  Audio files now in data/raw/:"
ls -lh data/raw/*.mp3 2>/dev/null | awk '{print "    " $9 "  (" $5 ")"}'
echo "==========================================================================="
