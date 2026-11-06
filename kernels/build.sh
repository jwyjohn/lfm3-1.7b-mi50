#!/usr/bin/env bash
# Compiles the GPU kernels to gfx906 code objects with the ROCm toolchain image.
# Only needed after changing a .hip file: the .co files are committed and embedded.
set -euo pipefail
cd "$(dirname "$0")"
image=${HIP_IMAGE:-rocm/dev-ubuntu-24.04:7.2.4}
for src in *.hip; do
  name=${src%.hip}
  flags=()
  # Hoisting would keep values across the DSpark phases and spill registers to scratch,
  # which this driver does not provide.
  [ "$name" = dspark ] && flags=(-mllvm -disable-machine-licm)
  docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/src" -w /src --entrypoint /opt/rocm/bin/hipcc "$image" \
    -O3 -w --offload-arch=gfx906 --genco --no-gpu-bundle-output "${flags[@]}" -o "$name.co" "$src"
  # The driver provides no scratch memory: a VGPR spill or a private segment is a build error
  # (SGPR spills go to VGPR lanes, not memory).
  notes=$(docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/src" -w /src --entrypoint /opt/rocm/llvm/bin/llvm-readelf "$image" \
    --notes "$name.co")
  if grep -E "(vgpr_spill_count|private_segment_fixed_size): +[1-9]" <<<"$notes"; then
    echo "$name.co spills registers or uses scratch memory" >&2
    exit 1
  fi
  echo "built $name.co"
done
