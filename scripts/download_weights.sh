#!/usr/bin/env bash
# Download model weights from Google Drive into ./weights/
# Skips the download entirely if all required files are already present.
# Retries up to 3 times on network failure.
#
# Manual fallback: download https://drive.google.com/drive/folders/1joO7w1Am7B418SIqGBq90YipQl81FMzh
# and mount the extracted folder: docker run -v /path/to/weights:/app/weights ...
set -uo pipefail

FOLDER_ID="1joO7w1Am7B418SIqGBq90YipQl81FMzh"

REQUIRED_FILES=(
    "weights/players_detection/yolov8m.pt"
    "weights/ball_detection/TrackNet_best.pt"
    "weights/ball_detection/InpaintNet_best.pt"
    "weights/players_keypoints_detection/best.pt"
    "weights/court_keypoints_detection/best.pt"
)

mkdir -p weights/players_detection \
         weights/ball_detection \
         weights/players_keypoints_detection \
         weights/court_keypoints_detection

# Allow completely bypassing the download (e.g. weights mounted via a volume)
if [ "${SKIP_WEIGHTS_DOWNLOAD:-0}" = "1" ]; then
    echo "SKIP_WEIGHTS_DOWNLOAD=1 — skipping download, trusting mounted weights."
    find weights/ -name "*.pt" -exec ls -lh {} \; 2>/dev/null || true
    exit 0
fi

# Skip download if all weights already present (e.g. mounted volume or container restart)
all_present=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -s "$f" ]; then
        all_present=false
        break
    fi
done

if $all_present; then
    echo "All weights already present — skipping download."
    find weights/ -name "*.pt" -exec ls -lh {} \;
    exit 0
fi

pip install --quiet gdown

# Retry folder download up to 3 times with exponential backoff
max_attempts=3
delay=10
success=false

for attempt in $(seq 1 $max_attempts); do
    echo "Downloading weights from Google Drive (attempt $attempt/$max_attempts)..."
    if gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" \
             -O . --remaining-ok; then
        success=true
        break
    fi
    echo "Attempt $attempt failed. Waiting ${delay}s before retry..."
    sleep "$delay"
    delay=$((delay * 2))
done

if ! $success; then
    echo "ERROR: Failed to download weights after $max_attempts attempts." >&2
    echo "Manual fix: download the folder in your browser and mount it:" >&2
    echo "  https://drive.google.com/drive/folders/${FOLDER_ID}" >&2
    echo "  docker run -v /path/to/weights:/app/weights ..." >&2
    exit 1
fi

# gdown --folder places subfolders at the repo root (e.g. ./ball_detection/).
# Move the *contents* of each subfolder into the pre-created weights/ subdirectory.
# Using `mv src/* dst/` avoids the double-nesting that occurs when the target
# directory already exists and `mv src dst` moves src *inside* dst instead.
for folder in ball_detection players_detection players_keypoints_detection court_keypoints_detection; do
    if [ -d "$folder" ]; then
        echo "Moving $folder/ into weights/"
        mv "$folder"/* "weights/$folder/"
        rm -rf "$folder"
    fi
done

# Verify all required files are actually present and non-empty
all_present=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -s "$f" ]; then
        echo "ERROR: Expected weight file missing after download: $f" >&2
        all_present=false
    fi
done

if ! $all_present; then
    echo "" >&2
    echo "Files found on disk after download:" >&2
    find . -name "*.pt" -exec ls -lh {} \; 2>/dev/null || echo "  (none)" >&2
    echo "" >&2
    echo "Manual fix: download the folder in your browser and mount it:" >&2
    echo "  https://drive.google.com/drive/folders/${FOLDER_ID}" >&2
    echo "  docker run -v /path/to/weights:/app/weights ..." >&2
    exit 1
fi

echo "Weights downloaded successfully."
find weights/ -name "*.pt" -exec ls -lh {} \;
