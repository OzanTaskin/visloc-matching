#!/usr/bin/env bash
# On macOS, the PyTorch and pycolmap wheels each ship a libomp.dylib. Loading
# both in one process aborts (OMP Error #15/#179). Point pycolmap at PyTorch's
# copy so a single OpenMP runtime is loaded. No-op on other platforms.
set -euo pipefail
[ "$(uname)" = "Darwin" ] || exit 0

SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
LIB="$SITE/pycolmap/.dylibs/libomp.dylib"
if [ -f "$LIB" ] && [ ! -L "$LIB" ]; then
  mv "$LIB" "$LIB.orig"
  ln -s ../../torch/lib/libomp.dylib "$LIB"
  echo "pycolmap now uses PyTorch's OpenMP runtime."
fi
