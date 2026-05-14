#!/usr/bin/env bash
# download_data.sh — fetch Spider and BIRD dev sets.
# Spider: https://yale-lily.github.io/spider
# BIRD : https://bird-bench.github.io/

set -euo pipefail

mkdir -p data
cd data

# --- Spider ---
if [ ! -d "spider" ]; then
  echo "Downloading Spider dataset..."
  # Official mirror (Yale). If link rots, replace with a current one.
  # Spider zip is ~95MB.
  curl -L -o spider.zip "https://drive.usercontent.google.com/download?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m&export=download&confirm=t"
  unzip -q spider.zip
  rm spider.zip
  # The unzip may produce a spider/ folder directly, or a parent.
  if [ ! -f "spider/dev.json" ]; then
    echo "ERROR: spider/dev.json not found after extraction."
    echo "Manual fallback: download from https://yale-lily.github.io/spider and unzip to data/spider/"
    exit 1
  fi
else
  echo "Spider already present."
fi

# --- BIRD dev ---
if [ ! -d "bird" ]; then
  echo "Downloading BIRD dev set (~2GB)..."
  mkdir -p bird
  cd bird
  curl -L -O "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip" || {
    echo "BIRD direct download failed."
    echo "Manual fallback: https://bird-bench.github.io/ → Download dev set → unzip to data/bird/"
    cd ..
    exit 1
  }
  unzip -q dev.zip
  rm dev.zip
  # Inside the zip, the layout is typically dev_20240627/ or similar.
  # We want: data/bird/dev.json and data/bird/dev_databases/
  # Flatten if needed:
  if [ -d "dev_20240627" ]; then
    mv dev_20240627/* .
    rmdir dev_20240627
  fi
  # Some releases ship dev_databases as another zip:
  if [ -f "dev_databases.zip" ]; then
    unzip -q dev_databases.zip
    rm dev_databases.zip
  fi
  cd ..
else
  echo "BIRD already present."
fi

echo
echo "=== Datasets ready ==="
echo "Spider: data/spider/dev.json + data/spider/database/"
echo "BIRD  : data/bird/dev.json + data/bird/dev_databases/"
