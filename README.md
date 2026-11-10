# Where was this photo taken?

**Classical vs. learned feature matching for visual localization under image degradations**

> **Status: work in progress.** The pipeline runs end to end and has been validated on a
> subset of queries; the full benchmark, figures and report are being finalised.

Visual localization estimates the position and orientation (6-DoF pose) of a camera from a
single photo, given a 3D map of the scene. It is how a robot or an AR headset knows where it
is. This project measures how much **learned feature matching** (SuperPoint, ALIKED +
LightGlue) improves over **classical SIFT**, and whether the gap grows when query photos are
taken in harder conditions: **low light, blur and low resolution**.

## Research question

On a small outdoor scene, with the same 3D map, retrieval and pose solver for every method:

1. How accurate is each feature/matcher pair on clean query images?
2. How much does each one degrade when the query images are dark, blurred or low-resolution?
3. What does it cost in runtime on a laptop CPU?

## Method

The pipeline follows the structure-based approach of
[hloc](https://github.com/cvg/Hierarchical-Localization) (Sarlin et al., CVPR 2019):

```
database images ──► local features ──► matching (co-visible pairs) ──► triangulation ──► 3D map
                                                                                          │
query image ──► NetVLAD retrieval (top-10) ──► local features ──► 2D–3D matches ──► PnP + RANSAC ──► pose
```

| Method | Features | Matcher |
|---|---|---|
| `sift-nn` | SIFT (DoG + RootSIFT, COLMAP) | Mutual nearest neighbour + ratio test |
| `superpoint-lightglue` | SuperPoint | LightGlue |
| `aliked-lightglue` | ALIKED | LightGlue |

All methods use at most 2048 keypoints and images resized to 1024 px. Each method gets its own
map, triangulated from its own matches with the ground-truth camera poses, so only the
features and the matcher change between methods.

**Query degradations** (database images are never modified, seeds are fixed):

| Condition | Transformation |
|---|---|
| `clean` | original image |
| `dark` | gamma 1.8, exposure ×0.35, Gaussian sensor noise (σ = 4 grey levels) |
| `blur` | Gaussian blur, σ = 3 px |
| `lowres` | downsample ×4, then upsample back |

**Metrics:** median position and rotation error, and the percentage of queries localized within
(25 cm, 2°), (50 cm, 5°) and (5 m, 10°). Queries where no pose is found count as failures.

**Data:** the ShopFacade scene of [Cambridge Landmarks](https://www.repository.cam.ac.uk/handle/1810/251336)
(Kendall et al., ICCV 2015): 231 database and 103 query images.

## Preliminary sanity check

On the first 10 query images (clean condition), both pipelines reproduce the accuracy reported by
hloc on this scene (≈ 4 cm median error):

| Method | Median error | Localized |
|---|---|---|
| SIFT + NN ratio | 4.6 cm / 0.15° | 10 / 10 |
| SuperPoint + LightGlue | 4.9 cm / 0.15° | 10 / 10 |

These numbers only validate the pipeline; they are not the benchmark result.

## Reproduce

Tested on macOS (Apple M2, CPU only) with Python 3.11.

```bash
make setup   # virtual environment, PyTorch, hloc (+ macOS OpenMP fix)
make data    # ShopFacade (~1.4 GB) and reference COLMAP models
make test    # unit tests (no data needed)
make smoke   # 10 queries, one method, ~5 min once the map is built
make run     # full benchmark: 3 methods × 4 conditions (a few hours on CPU)
make report  # compile report/report.pdf
```

Outputs are cached in `outputs/`, so an interrupted run resumes where it stopped. Results are
written to `results/` (`results.csv`, `table.tex`, figures).

## Repository layout

```
run.py                  # entry point: build maps, localize, evaluate, report
src/visloc/
  config.py             # paths, methods, degradations, hyper-parameters
  pipeline.py           # thin wrappers around hloc (map building, localization)
  degrade.py            # query degradations
  evaluate.py           # pose errors and recall metrics
  report.py             # results table and plots
  visualize.py          # qualitative figures
tests/                  # unit tests
report/                 # LaTeX report
scripts/                # data download, macOS OpenMP fix
```

## Implementation notes

- **macOS / OpenMP.** The PyTorch and pycolmap wheels each bundle `libomp`, which crashes the
  process when both are loaded. `scripts/fix_macos_openmp.sh` makes pycolmap use PyTorch's copy.
- **pycolmap ≥ 4.** hloc stores geometrically verified pairs with an `UNDEFINED` configuration,
  which pycolmap 4 ignores during triangulation (the map ends up empty). Since pairs are verified
  against ground-truth relative poses, `pipeline.triangulate` marks them as `CALIBRATED`.

## References

- Sarlin et al., *From Coarse to Fine: Robust Hierarchical Localization at Large Scale*, CVPR 2019.
- Lindenberger, Sarlin, Pollefeys, *LightGlue: Local Feature Matching at Light Speed*, ICCV 2023.
- DeTone et al., *SuperPoint: Self-Supervised Interest Point Detection and Description*, CVPRW 2018.
- Zhao et al., *ALIKED*, IEEE TIM 2023.
- Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, IJCV 2004.
- Arandjelović et al., *NetVLAD*, CVPR 2016.
- Kendall et al., *PoseNet* (Cambridge Landmarks), ICCV 2015.
- Schönberger and Frahm, *Structure-from-Motion Revisited*, CVPR 2016.

## License

MIT for this code. Cambridge Landmarks and the pretrained models keep their own licenses.
