#!/usr/bin/env bash
# Builds the self-contained executable dist/lfm3-1.7b-mi50.
#
#   ./build.sh /path/to/LFM3-1.7B
#
# The directory needs model.safetensors, config.json and tokenizer.json (see the README
# for how to download them). Requires only the Go toolchain: the GPU kernels are committed
# as compiled code objects, and kernels/build.sh rebuilds them after a .hip change.
set -euo pipefail
if [ $# -ne 1 ]; then
  echo "usage: $0 <LFM3-1.7B snapshot directory>" >&2
  exit 2
fi
model=$(cd "$1" && pwd)
cd "$(dirname "$0")"
export CGO_ENABLED=0
mkdir -p build dist
go build -trimpath -ldflags='-s -w' -o build/lfm3-1.7b-mi50 ./cmd/lfm3-1.7b-mi50
go build -trimpath -ldflags='-s -w' -o build/pack ./cmd/pack
if [ ! -s build/payload.bin ]; then
  ./build/pack -model "$model" -out build/payload.bin
fi
./build/pack -elf build/lfm3-1.7b-mi50 -payload build/payload.bin -out dist/lfm3-1.7b-mi50
ls -l dist/lfm3-1.7b-mi50
