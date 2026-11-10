"""Paths, methods, image degradations and hyper-parameters (single source of truth)."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "datasets" / "cambridge"
REF_MODELS_DIR = DATASET_DIR / "CambridgeLandmarks_Colmap_Retriangulated_1024px"
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DEFAULT_SCENE = "ShopFacade"

# Pairs per database image used to triangulate the map, and database images
# retrieved per query for localization (kept small: everything runs on CPU).
NUM_COVIS = 10
NUM_LOC = 10
RESIZE_MAX = 1024  # same input resolution for every method, for a fair comparison
MAX_KEYPOINTS = 2048

# Localization success thresholds (translation in metres, rotation in degrees).
THRESHOLDS = [(0.25, 2.0), (0.5, 5.0), (5.0, 10.0)]

RETRIEVAL_CONF = {
    "output": "global-feats-netvlad",
    "model": {"name": "netvlad"},
    "preprocessing": {"resize_max": RESIZE_MAX},
}


@dataclass(frozen=True)
class Method:
    name: str  # short id used in file names and tables
    label: str  # human-readable name used in figures
    features: dict  # hloc feature-extraction config
    matcher: dict  # hloc matching config


METHODS = {
    m.name: m
    for m in [
        Method(
            name="sift-nn",
            label="SIFT + NN ratio",
            features={
                "output": "feats-sift",
                "model": {"name": "dog", "max_keypoints": MAX_KEYPOINTS},
                "preprocessing": {"grayscale": True, "resize_max": RESIZE_MAX},
            },
            matcher={
                "output": "matches-nn-ratio",
                "model": {
                    "name": "nearest_neighbor",
                    "do_mutual_check": True,
                    "ratio_threshold": 0.8,
                },
            },
        ),
        Method(
            name="superpoint-lightglue",
            label="SuperPoint + LightGlue",
            features={
                "output": "feats-superpoint",
                "model": {
                    "name": "superpoint",
                    "nms_radius": 3,
                    "max_keypoints": MAX_KEYPOINTS,
                },
                "preprocessing": {"grayscale": True, "resize_max": RESIZE_MAX},
            },
            matcher={
                "output": "matches-lightglue",
                "model": {"name": "lightglue", "features": "superpoint"},
            },
        ),
        Method(
            name="aliked-lightglue",
            label="ALIKED + LightGlue",
            features={
                "output": "feats-aliked",
                "model": {
                    "name": "aliked",
                    "model_name": "aliked-n16",
                    "max_num_keypoints": MAX_KEYPOINTS,
                },
                "preprocessing": {"grayscale": False, "resize_max": RESIZE_MAX},
            },
            matcher={
                "output": "matches-lightglue",
                "model": {"name": "lightglue", "features": "aliked"},
            },
        ),
    ]
}

# Query-image degradations (the database images are never modified).
CONDITIONS = {
    "clean": {},
    "dark": {"gamma": 1.8, "gain": 0.35, "noise_std": 4.0},
    "blur": {"sigma": 3.0},
    "lowres": {"scale": 0.25},
}


def scene_paths(scene: str, tag: str = "") -> dict:
    """All input/output locations for one scene. `tag` separates e.g. smoke runs."""
    ref = REF_MODELS_DIR / scene
    base = OUTPUTS_DIR / scene  # query-independent: maps and database features
    return {
        "images": DATASET_DIR / scene,
        "ref_sfm": ref / "model_train",  # reference SIFT model (database poses)
        "gt": ref / "empty_all",  # ground-truth poses of all images (text model)
        "list_query": ref / "list_query.txt",
        "list_db": ref / "list_db.txt",
        "maps": base / "maps",
        "out": base / f"queries{tag}",  # everything that depends on the query set
        "ref_sfm_scaled": base / "sfm_sift_scaled",
        "sfm_pairs": base / f"pairs-db-covis{NUM_COVIS}.txt",
        "retrieval_db": base / "netvlad_db.h5",
    }
