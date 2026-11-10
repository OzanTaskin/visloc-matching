"""Thin wrappers around hloc: build one 3D map per method, then localize queries.

Mirrors hloc's Cambridge Landmarks pipeline (hloc/pipelines/Cambridge), with two
changes: query features live in their own file (so degraded queries can be
swapped in without touching the map), and every step is cached on disk.
"""

import pickle
import sqlite3
import time
from pathlib import Path

import cv2
import pycolmap
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
    p["maps"].mkdir(parents=True, exist_ok=True)
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
    out = p["maps"] / method.name
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
    if not (paths["sfm"] / "points3D.bin").exists() or _num_points(paths["sfm"]) == 0:
        triangulate(paths["sfm"], p["ref_sfm_scaled"], p["images"], p["sfm_pairs"],
                    paths["db_features"], paths["db_matches"])
    return paths


def _num_points(sfm_dir: Path) -> int:
    return pycolmap.Reconstruction(sfm_dir).num_points3D()


def triangulate(sfm_dir, reference_model, image_dir, pairs, features, matches):
    """hloc's `triangulation.main`, adapted to pycolmap >= 4.

    hloc stores geometrically verified pairs with configuration UNDEFINED, which
    pycolmap 4 skips when building the correspondence graph, giving an empty map.
    The pairs are verified against the known (ground-truth) relative poses, so we
    mark them as CALIBRATED before triangulating.
    """
    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"
    database.unlink(missing_ok=True)
    reference = pycolmap.Reconstruction(reference_model)
    image_ids = triangulation.create_db_from_model(reference, database)
    with pycolmap.Database.open(database) as db:
        triangulation.import_features(image_ids, db, features)
        triangulation.import_matches(image_ids, db, pairs, matches)
        triangulation.geometric_verification(image_ids, reference, db, features, pairs, matches)
    with sqlite3.connect(database) as con:
        con.execute("UPDATE two_view_geometries SET config = ? WHERE rows > 0",
                    (int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED),))
    rec = triangulation.run_triangulation(sfm_dir, database, image_dir, reference)
    print(f"Triangulated map: {rec.num_points3D()} points, "
          f"mean reprojection error {rec.compute_mean_reprojection_error():.2f} px")
    return rec


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
