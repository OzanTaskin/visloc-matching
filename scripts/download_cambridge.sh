#!/usr/bin/env bash
# Download the ShopFacade scene of Cambridge Landmarks and the reference
# COLMAP (SIFT) models provided with hloc. Usage: bash scripts/download_cambridge.sh
set -euo pipefail

DATASET="${DATASET:-datasets/cambridge}"
SCENE="${SCENE:-ShopFacade}"
SCENE_ID="${SCENE_ID:-251336}"
GDRIVE_ID="1esqzZ1zEQlzZVic-H32V6kkZvc4NeS15"
REF_DIR="$DATASET/CambridgeLandmarks_Colmap_Retriangulated_1024px"

mkdir -p "$DATASET"

if [ ! -d "$DATASET/$SCENE" ]; then
  echo "Downloading $SCENE (~1.4 GB)..."
  curl -L --fail --retry 3 -o "$DATASET/$SCENE.zip" \
    "https://www.repository.cam.ac.uk/bitstream/handle/1810/$SCENE_ID/$SCENE.zip"
  unzip -q "$DATASET/$SCENE.zip" -d "$DATASET"
  rm "$DATASET/$SCENE.zip"
else
  echo "$SCENE already present, skipping."
fi

if [ ! -d "$REF_DIR" ]; then
  echo "Downloading reference SfM models (Google Drive)..."
  gdown "$GDRIVE_ID" \
    -O "$DATASET/ref_models.zip"
  unzip -q "$DATASET/ref_models.zip" -d "$DATASET"
  rm "$DATASET/ref_models.zip"
else
  echo "Reference models already present, skipping."
fi

echo "Done. Scene: $DATASET/$SCENE  Reference: $REF_DIR/$SCENE"
