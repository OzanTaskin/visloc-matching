"""Controlled degradations applied to query images (pure NumPy / OpenCV)."""

import zlib

import cv2
import numpy as np


def seed_for(name: str) -> int:
    """Deterministic per-image seed, so every run produces identical images."""
    return zlib.crc32(name.encode())


def darken(img: np.ndarray, gamma: float, gain: float, noise_std: float, seed: int):
    """Simulate a low-light capture: gamma curve, lower exposure, sensor noise."""
    rng = np.random.default_rng(seed)
    x = (img.astype(np.float32) / 255.0) ** gamma * gain * 255.0
    x += rng.normal(0.0, noise_std, size=img.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def blur(img: np.ndarray, sigma: float):
    """Simulate defocus or motion blur with an isotropic Gaussian kernel."""
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def lowres(img: np.ndarray, scale: float):
    """Simulate a low-resolution camera: downsample, then upsample back."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply(img: np.ndarray, condition: str, params: dict, name: str = "") -> np.ndarray:
    """Return a degraded copy of `img` (same shape and dtype)."""
    if condition == "clean":
        return img.copy()
    if condition == "dark":
        return darken(img, seed=seed_for(name), **params)
    if condition == "blur":
        return blur(img, **params)
    if condition == "lowres":
        return lowres(img, **params)
    raise ValueError(f"Unknown condition: {condition}")
