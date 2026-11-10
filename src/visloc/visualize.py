"""Qualitative figures: degraded query examples, top-down map, 3D viewer."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pycolmap  # noqa: E402

from . import degrade  # noqa: E402
from .config import CONDITIONS  # noqa: E402
from .evaluate import read_colmap_images_txt, read_poses  # noqa: E402


def plot_conditions(image_path: Path, name: str, out: Path) -> None:
    """One query image under every condition, side by side."""
    img = cv2.imread(str(image_path))
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(13, 2.4))
    for ax, (cond, params) in zip(axes, CONDITIONS.items()):
        ax.imshow(cv2.cvtColor(degrade.apply(img, cond, params, name), cv2.COLOR_BGR2RGB))
        ax.set_title(cond)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _centre(R, t):
    return -R.T @ t


def plot_topdown(sfm_dir: Path, gt_txt: Path, poses: dict, queries: list, out: Path,
                 failed: dict, title: str) -> None:
    """Map points and query camera centres projected on the dominant ground plane.

    `poses` maps a method label to its results file; lines join ground truth to
    estimate, and failed queries are drawn as crosses at their ground truth.
    """
    rec = pycolmap.Reconstruction(sfm_dir)
    pts = np.array([p.xyz for p in rec.points3D.values()])
    gt = read_colmap_images_txt(gt_txt)
    centres = np.array([_centre(*gt[q]) for q in queries])
    # Principal axes of the query trajectory give a stable top-down view.
    mean = centres.mean(0)
    _, _, vt = np.linalg.svd(centres - mean)
    proj = vt[:2].T
    keep = np.linalg.norm((pts - mean) @ proj, axis=1) < 3 * np.abs((centres - mean) @ proj).max()

    fig, axes = plt.subplots(1, len(poses), figsize=(5.5 * len(poses), 4.5), squeeze=False)
    for ax, (label, path) in zip(axes[0], poses.items()):
        est = read_poses(path)
        p2 = (pts[keep] - mean) @ proj
        ax.scatter(p2[:, 0], p2[:, 1], s=0.2, c="0.7", rasterized=True)
        for q, c in zip(queries, centres):
            g = (c - mean) @ proj
            if q in failed.get(label, set()):
                ax.plot(*g, "rx", ms=5)
                continue
            e = (_centre(*est[q]) - mean) @ proj
            ax.plot([g[0], e[0]], [g[1], e[1]], "r-", lw=0.8)
            ax.plot(*g, "o", ms=2.5, c="tab:green")
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def export_3d(sfm_dir: Path, out: Path) -> None:
    """Interactive 3D view of the map (Plotly HTML, opened in a browser)."""
    from hloc.utils import viz_3d

    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, pycolmap.Reconstruction(sfm_dir), color="rgba(50,90,200,0.6)",
                               name="map", points_rgb=True)
    fig.write_html(out)
