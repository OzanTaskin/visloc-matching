"""Pose-error metrics. Pure NumPy so it can be unit-tested without hloc."""

from pathlib import Path

import numpy as np

from .config import THRESHOLDS


def qvec2rotmat(q) -> np.ndarray:
    """Rotation matrix from a COLMAP quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def read_poses(path: Path) -> dict:
    """Read `name qw qx qy qz tx ty tz` lines (hloc results) -> {name: (R, t)}."""
    poses = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        name, *vals = line.split()
        vals = np.array(vals, dtype=float)
        poses[name] = (qvec2rotmat(vals[:4]), vals[4:7])
    return poses


def read_colmap_images_txt(path: Path) -> dict:
    """Ground-truth poses from a COLMAP `images.txt` -> {name: (R, t)}."""
    poses = {}
    lines = [ln for ln in Path(path).read_text().splitlines() if not ln.startswith("#")]
    # COLMAP alternates one pose line and one (possibly empty) 2D-points line.
    for line in lines[::2]:
        parts = line.split()
        if len(parts) < 10:
            continue
        q, t, name = parts[1:5], parts[5:8], parts[9]
        poses[name] = (qvec2rotmat(np.array(q, float)), np.array(t, float))
    return poses


def pose_error(R_gt, t_gt, R, t):
    """Camera-centre distance (same unit as t) and rotation angle (degrees)."""
    c_gt, c = -R_gt.T @ t_gt, -R.T @ t
    cos = np.clip((np.trace(R_gt.T @ R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.linalg.norm(c_gt - c)), float(np.degrees(np.arccos(cos)))


def summarize(err_t, err_R, thresholds=THRESHOLDS) -> dict:
    """Median errors and % of queries under each (metres, degrees) threshold."""
    err_t, err_R = np.asarray(err_t, float), np.asarray(err_R, float)
    out = {
        "n_queries": int(len(err_t)),
        "median_t_cm": float(np.median(err_t) * 100),
        "median_R_deg": float(np.median(err_R)),
    }
    for th_t, th_R in thresholds:
        key = f"recall_{th_t * 100:g}cm_{th_R:g}deg"
        out[key] = float(np.mean((err_t < th_t) & (err_R < th_R)) * 100)
    return out


def evaluate(results: Path, gt_images_txt: Path, query_names, failed=()) -> tuple:
    """Per-query errors and summary. Failed or missing queries count as infinite error."""
    preds, gt = read_poses(results), read_colmap_images_txt(gt_images_txt)
    rows = []
    for name in query_names:
        if name in preds and name not in failed:
            e_t, e_R = pose_error(*gt[name], *preds[name])
        else:
            e_t, e_R = np.inf, 180.0
        rows.append({"query": name, "err_t_m": e_t, "err_R_deg": e_R})
    summary = summarize([r["err_t_m"] for r in rows], [r["err_R_deg"] for r in rows])
    summary["localized_pct"] = 100.0 * (1 - len(set(failed) & set(query_names)) / len(query_names))
    return rows, summary
