"""Visual localization benchmark: classical vs learned feature matching.

macOS note: PyTorch and pycolmap wheels each bundle their own OpenMP runtime,
which crashes the process when both are loaded. `make setup` runs
scripts/fix_macos_openmp.sh so that pycolmap uses PyTorch's copy.
"""
