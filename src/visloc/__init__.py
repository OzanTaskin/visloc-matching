"""Visual localization benchmark: classical vs learned feature matching."""

import os

# PyTorch and pycolmap each ship their own OpenMP runtime on macOS, which makes
# the process abort when both are loaded. This flag lets them coexist; it must be
# set before either library is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
