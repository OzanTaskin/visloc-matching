"""Unit tests for the parts that do not need hloc or the dataset."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visloc import degrade  # noqa: E402
from visloc.config import CONDITIONS  # noqa: E402
from visloc.evaluate import (evaluate, pose_error, qvec2rotmat, read_colmap_images_txt,  # noqa: E402
                             summarize)


def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_qvec2rotmat_identity_and_z_rotation():
    assert np.allclose(qvec2rotmat([1, 0, 0, 0]), np.eye(3))
    a = np.radians(90)
    q = [np.cos(a / 2), 0, 0, np.sin(a / 2)]
    assert np.allclose(qvec2rotmat(q), rot_z(90))


def test_pose_error_translation_and_rotation():
    R, t = np.eye(3), np.zeros(3)
    # Same orientation, camera centre moved by 0.3 m (t = -R c).
    e_t, e_R = pose_error(R, t, R, -np.array([0.3, 0, 0]))
    assert e_t == pytest.approx(0.3) and e_R == pytest.approx(0.0, abs=1e-6)
    # Rotation only.
    e_t, e_R = pose_error(R, t, rot_z(10), t)
    assert e_t == pytest.approx(0.0) and e_R == pytest.approx(10.0)


def test_summarize_thresholds_and_failures():
    s = summarize([0.1, 0.4, 3.0, np.inf], [1.0, 4.0, 9.0, 180.0])
    assert s["recall_25cm_2deg"] == 25.0
    assert s["recall_50cm_5deg"] == 50.0
    assert s["recall_500cm_10deg"] == 75.0
    assert s["median_t_cm"] == pytest.approx(170.0)


def test_evaluate_end_to_end(tmp_path):
    gt = tmp_path / "images.txt"
    gt.write_text("# header\n1 1 0 0 0 0 0 0 1 a.png\n\n2 1 0 0 0 1 0 0 1 b.png\n\n")
    assert set(read_colmap_images_txt(gt)) == {"a.png", "b.png"}
    pred = tmp_path / "poses.txt"
    pred.write_text("a.png 1 0 0 0 0.1 0 0\nb.png 1 0 0 0 1 0 0\n")
    rows, s = evaluate(pred, gt, ["a.png", "b.png"], failed={"b.png"})
    assert rows[0]["err_t_m"] == pytest.approx(0.1)
    assert np.isinf(rows[1]["err_t_m"])
    assert s["localized_pct"] == 50.0


@pytest.mark.parametrize("cond", list(CONDITIONS))
def test_degradations_keep_shape_and_are_deterministic(cond):
    img = np.random.default_rng(0).integers(0, 256, (60, 80, 3), dtype=np.uint8)
    a = degrade.apply(img, cond, CONDITIONS[cond], "seq1/x.png")
    b = degrade.apply(img, cond, CONDITIONS[cond], "seq1/x.png")
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)


def test_dark_is_darker_and_blur_is_smoother():
    img = np.random.default_rng(1).integers(0, 256, (60, 80, 3), dtype=np.uint8)
    assert degrade.apply(img, "dark", CONDITIONS["dark"], "x").mean() < img.mean() / 2
    blurred = degrade.apply(img, "blur", CONDITIONS["blur"])
    assert np.abs(np.diff(blurred.astype(float), axis=1)).mean() < \
        np.abs(np.diff(img.astype(float), axis=1)).mean() / 3
