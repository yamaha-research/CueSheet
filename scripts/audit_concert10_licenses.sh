#!/usr/bin/env bash
# Re-run the YouTube license audit for every CONCERT-10 v1.0 row.
#
# yt-dlp returns the uploader-declared license field (either "Standard
# YouTube License" — uploader retains all rights — or "Creative Commons
# Attribution license (reuse allowed)" when the uploader has opted in).
# We capture license + uploader + channel + duration so the output is
# stable enough to commit verbatim and diff over time.
#
# Output:
#   data/license_audit/concert10_licenses.tsv
#
# Re-run when the manifest changes, or annually as a sanity audit
# (uploaders occasionally relicense individual videos).
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/license_audit
OUT=data/license_audit/concert10_licenses.tsv

echo -e "show_id\turl\tlicense\tuploader\tchannel\tdurations_sec" > "$OUT"
awk -F'\t' 'NR>1 && NF>=2 {print $1"\t"$2}' data/CONCERT-10.tsv \
  | while IFS=$'\t' read -r show url; do
      meta=$(yt-dlp --skip-download --no-warnings -j "$url" 2>/dev/null \
        | uv run python -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('license') or 'Standard YouTube License',
      d.get('uploader', ''),
      d.get('channel',  ''),
      d.get('duration', ''),
      sep='\t')
" 2>/dev/null) || meta=$'ERROR\t\t\t'
      echo -e "$show\t$url\t$meta" | tee -a "$OUT"
    done

echo
echo "License counts:"
awk -F'\t' 'NR>1 {print $3}' "$OUT" | sort | uniq -c | sort -rn
