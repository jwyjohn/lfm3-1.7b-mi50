#!/usr/bin/env bash
# Compiles the GPU kernels to gfx906 code objects with the ROCm toolchain image.
# Only needed after changing a .hip file: the .co files are committed and embedded.
set -euo pipefail
cd "$(dirname "$0")"
image=${HIP_IMAGE:-rocm/dev-ubuntu-24.04:7.2.4}
for src in *.hip; do
  name=${src%.hip}
  docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/src" -w /src --entrypoint /opt/rocm/bin/hipcc "$image" \
    -O3 -w --offload-arch=gfx906 --genco --no-gpu-bundle-output -o "$name.co" "$src"
  echo "built $name.co"
done
