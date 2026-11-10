"""Thin wrappers around hloc: build one 3D map per method, then localize queries.

Mirrors hloc's Cambridge Landmarks pipeline (hloc/pipelines/Cambridge), with two
changes: query features live in their own file (so degraded queries can be
swapped in without touching the map), and every step is cached on disk.
"""

import pickle
import time
from pathlib import Path

import cv2
from hloc import (extract_features, localize_sfm, match_features,
                  pairs_from_covisibility, pairs_from_retrieval, triangulation)
from hloc.pipelines.Cambridge.utils import create_query_list_with_intrinsics, scale_sfm_images

from . import degrade
from .config import CONDITIONS, NUM_COVIS, NUM_LOC, RETRIEVAL_CONF, Method


def read_list(path: Path) -> list:
    return [ln for ln in Path(path).read_text().split() if ln]


def prepare_scene(p: dict, max_queries: int = 0) -> list:
    """Shared, method-independent inputs. Returns the query names used."""
    p["out"].mkdir(parents=True, exist_ok=True)
    queries = read_list(p["list_query"])
    if max_queries:
        queries = queries[:max_queries]
    (p["out"] / "queries.txt").write_text("\n".join(queries))

    query_list = p["out"] / "query_list_with_intrinsics.txt"
    create_query_list_with_intrinsics(p["gt"], query_list, p["out"] / "queries.txt",
                                      ext=".txt", image_dir=p["images"])
    if not (p["ref_sfm_scaled"] / "images.bin").exists():
        scale_sfm_images(p["ref_sfm"], p["ref_sfm_scaled"], p["images"])
    if not p["sfm_pairs"].exists():
        pairs_from_covisibility.main(p["ref_sfm"], p["sfm_pairs"], num_matched=NUM_COVIS)
    extract_features.main(RETRIEVAL_CONF, p["images"], image_list=read_list(p["list_db"]),
                          feature_path=p["retrieval_db"])
    return queries


def condition_dir(p: dict, condition: str, queries: list) -> Path:
    """Folder holding the query images for one condition (degraded copies)."""
    if condition == "clean":
        return p["images"]
    out = p["out"] / "images" / condition
    for name in queries:
        dst = out / name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(p["images"] / name))
            cv2.imwrite(str(dst), degrade.apply(img, condition, CONDITIONS[condition], name))
    return out


def build_map(p: dict, method: Method) -> dict:
    """Database features + triangulated 3D map for one method (cached)."""
    out = p["out"] / method.name
    paths = {
        "db_features": out / "feats_db.h5",
        "db_matches": out / "matches_db.h5",
        "sfm": out / "sfm",
    }
    extract_features.main(method.features, p["images"], image_list=read_list(p["list_db"]),
                          feature_path=paths["db_features"])
    if not paths["db_matches"].exists():
        match_features.main(method.matcher, p["sfm_pairs"], features=paths["db_features"],
                            matches=paths["db_matches"])
    if not (paths["sfm"] / "images.bin").exists():
        triangulation.main(paths["sfm"], p["ref_sfm_scaled"], p["images"], p["sfm_pairs"],
                           paths["db_features"], paths["db_matches"])
    return paths


def localize(p: dict, method: Method, map_paths: dict, condition: str, queries: list) -> dict:
    """Localize the (possibly degraded) queries against the method's map."""
    images = condition_dir(p, condition, queries)
    shared = p["out"] / "conditions" / condition
    out = p["out"] / method.name / condition
    out.mkdir(parents=True, exist_ok=True)

    # Image retrieval (NetVLAD) is shared by all methods for a given condition.
    retrieval_q = extract_features.main(RETRIEVAL_CONF, images, image_list=queries,
                                        feature_path=shared / "netvlad_q.h5")
    loc_pairs = shared / f"pairs-query-netvlad{NUM_LOC}.txt"
    if not loc_pairs.exists():
        pairs_from_retrieval.main(retrieval_q, loc_pairs, NUM_LOC, db_model=p["ref_sfm"],
                                  db_descriptors=p["retrieval_db"])

    results = out / "poses.txt"
    timing = out / "seconds_per_query.txt"
    if not results.exists():
        start = time.perf_counter()
        q_features = extract_features.main(method.features, images, image_list=queries,
                                           feature_path=out / "feats_q.h5")
        q_matches = match_features.main(method.matcher, loc_pairs, features=q_features,
                                        matches=out / "matches_q.h5",
                                        features_ref=map_paths["db_features"])
        localize_sfm.main(map_paths["sfm"], p["out"] / "query_list_with_intrinsics.txt",
                          loc_pairs, q_features, q_matches, results,
                          covisibility_clustering=False, prepend_camera_name=True)
        timing.write_text(f"{(time.perf_counter() - start) / len(queries):.3f}")
    return {"poses": results, "images": images,
            "seconds_per_query": float(timing.read_text()), "failed": failed_queries(results)}


def failed_queries(results: Path) -> set:
    """Queries where PnP+RANSAC found no pose (hloc then writes the closest DB pose)."""
    with open(f"{results}_logs.pkl", "rb") as f:
        logs = pickle.load(f)
    return {q for q, log in logs["loc"].items() if log.get("PnP_ret") is None}
