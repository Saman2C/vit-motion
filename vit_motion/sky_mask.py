"""Per-frame semantic region masks (sky / ground / other) for ViT-Motion.

Why this module exists
----------------------
Stage 2/3 of the interpretability arc measured and then penalized "the fraction
of saliency mass that falls on the sky".  Until now the sky was defined by a
crude heuristic: *the top half of the frame* (``horizon_frac = 0.5``).  That is
wrong in three separate ways and every number derived from it inherits the error:

1. It labels the horizon, the apartment towers, the tents and the tree line as
   "sky", so a model looking at man-made structure is scored as looking at sky.
2. It labels genuinely-sky pixels below the mid-line (the frame is not level;
   the vehicle pitches) as "ground".
3. The sky area share is *forced* to be exactly 0.50, so the headline
   "saliency-to-area ratio" is normalized by a constant instead of by the real
   per-frame sky area (measured at 0.44 +- 0.02 on the 2026-08-19 data).

This module replaces the heuristic with a real, per-frame mask, following the
practice of the papers the method comes from: RobustViT (Chefer et al.,
NeurIPS 2022) supervises relevance maps with *segmentation* masks, and
Right-for-the-Right-Reasons (Ross et al., IJCAI 2017) needs an explicit
"irrelevant region" annotation A per input.

Two independent mask sources are provided, plus the tools to cross-check them:

``segmentation``
    A pretrained semantic segmentation network (SegFormer fine-tuned on
    ADE20K by default) run on the RGB frame.  ADE20K contains an explicit
    ``sky`` class and a family of ground classes (grass / earth / road / path /
    field / sand / dirt track / water), so one forward pass yields the full
    three-way decomposition sky / ground / other-structure.  Works on *any* RGB
    frame, including the v0.2.1 training set, which has no depth.

``depth``
    A geometric mask from the RealSense aligned-to-color 16-bit depth frame:
    a pixel is sky when the depth return is invalid (0) or beyond
    ``far_m`` metres.  This needs no learned model and no labels, but exists
    only for the 2026-08-19 collection.

The depth mask is therefore used as an *independent reference* to validate the
segmentation mask (:func:`mask_agreement`), and the segmentation mask is what
the training / quantification pipeline consumes everywhere.

Everything here is plain numpy + PIL; ``torch`` and ``transformers`` are
imported lazily and only by :class:`SegformerSkyMasker`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

__all__ = [
    "DepthRegions",
    "EnergySkyMasker",
    "RegionMasks",
    "depth_regions",
    "SegformerSkyMasker",
    "build_masker",
    "cache_provides_ground",
    "enhance_for_segmentation",
    "load_mask_quality",
    "mask_is_plausible",
    "depth_sky_mask",
    "half_frame_mask",
    "load_depth_png",
    "mask_agreement",
    "read_mask_png",
    "region_masks_from_labels",
    "resize_mask",
    "write_mask_png",
]

# --------------------------------------------------------------------------- #
# Region definition
# --------------------------------------------------------------------------- #
# Matched against the segmentation model's own ``id2label`` (case-insensitive,
# comma-separated synonyms are split), so the code does not hard-code ADE20K
# index numbers and keeps working if the checkpoint is swapped.
SKY_LABELS = frozenset({"sky"})
GROUND_LABELS = frozenset(
    {
        "grass", "earth", "ground", "road", "route", "path", "field", "sand",
        "dirt track", "land", "soil", "floor", "flooring", "sidewalk",
        "pavement", "water",
    }
)
#: The drivable surface the vehicle is actually moving over — mud, dry dirt,
#: grass, puddles.  ``water`` is in the list on purpose: the wet runs have
#: standing water ON the ground, and those pixels move with the vehicle.
#: ``hill``, ``mountain`` and ``sea`` are deliberately NOT here — distant
#: terrain is as useless for egomotion as a building is.

#: Everything that is neither sky nor ground: trees, buildings, cars, tents,
#: people, fences. Together with sky this is the "distractor" region the
#: attention-guided fine-tune penalizes.

REGION_SKY, REGION_GROUND, REGION_OTHER = 0, 1, 2
REGION_NAMES = ("sky", "ground", "other")


@dataclass
class RegionMasks:
    """Boolean masks for one frame, all shaped ``[H, W]``.

    ``provides_ground`` is not decoration.  Only a semantic segmenter can say
    "this is drivable terrain"; a sky detector can only say "this is not sky",
    which lumps trees, buildings and cars in with the grass.  Any consumer that
    penalizes the NON-ground region must refuse to run on masks where this is
    False, or it will train the model to look away from the very surface it is
    supposed to look at.
    """

    sky: np.ndarray
    ground: np.ndarray
    other: np.ndarray
    source: str = "segmentation"
    provides_ground: bool = True
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.sky.shape)  # type: ignore[return-value]

    def area_fractions(self) -> dict[str, float]:
        n = float(self.sky.size)
        return {
            "sky": float(self.sky.sum()) / n,
            "ground": float(self.ground.sum()) / n,
            "other": float(self.other.sum()) / n,
        }

    def label_image(self) -> np.ndarray:
        """``[H, W]`` uint8 with 0 = sky, 1 = ground, 2 = other."""
        out = np.full(self.sky.shape, REGION_OTHER, dtype=np.uint8)
        out[self.ground] = REGION_GROUND
        out[self.sky] = REGION_SKY
        return out

    def resized(self, size: tuple[int, int]) -> "RegionMasks":
        labels = np.asarray(
            Image.fromarray(self.label_image()).resize((size[1], size[0]), Image.NEAREST)
        )
        return RegionMasks(
            sky=labels == REGION_SKY,
            ground=labels == REGION_GROUND,
            other=labels == REGION_OTHER,
            source=self.source,
            provides_ground=self.provides_ground,
            meta=dict(self.meta),
        )


def region_masks_from_labels(
    labels: np.ndarray,
    id2label: dict[int, str],
    sky_labels: Iterable[str] = SKY_LABELS,
    ground_labels: Iterable[str] = GROUND_LABELS,
    source: str = "segmentation",
) -> RegionMasks:
    """Collapse a dense semantic label map into sky / ground / other."""
    sky_set = {s.lower() for s in sky_labels}
    ground_set = {s.lower() for s in ground_labels}
    sky_ids, ground_ids = [], []
    for idx, name in id2label.items():
        parts = {p.strip().lower() for p in str(name).split(",")}
        if parts & sky_set:
            sky_ids.append(int(idx))
        elif parts & ground_set:
            ground_ids.append(int(idx))
    sky = np.isin(labels, sky_ids)
    ground = np.isin(labels, ground_ids)
    other = ~(sky | ground)
    return RegionMasks(
        sky=sky, ground=ground, other=other, source=source, provides_ground=True,
        meta={"sky_ids": sorted(sky_ids), "ground_ids": sorted(ground_ids),
              "sky_names": sorted(id2label[i] for i in sky_ids),
              "ground_names": sorted(id2label[i] for i in ground_ids)},
    )


# --------------------------------------------------------------------------- #
# 1) Segmentation-based masker
# --------------------------------------------------------------------------- #
class SegformerSkyMasker:
    """Sky / ground / other masks from a pretrained ADE20K segmentation model.

    Defaults to ``nvidia/segformer-b0-finetuned-ade-512-512`` (3.8M parameters,
    fast enough to pre-compute masks for a 37k-image dataset in minutes on one
    GPU).  Any HuggingFace ``SemanticSegmentation`` checkpoint whose label set
    contains ``sky`` works; the region mapping is resolved from ``id2label``.

    The masks are *pre-computed once and cached to disk* (see
    ``precompute_sky_masks.py``); they are never produced inside the training
    loop, so the RRR fine-tune costs the same as before.
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: str | None = None,
        batch_size: int = 8,
        sky_labels: Iterable[str] = SKY_LABELS,
        ground_labels: Iterable[str] = GROUND_LABELS,
    ) -> None:
        import torch  # local import: only this class needs torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        self.torch = torch
        self.model_name = model_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.batch_size = int(batch_size)
        self.sky_labels = frozenset(sky_labels)
        self.ground_labels = frozenset(ground_labels)

    # -- core ------------------------------------------------------------- #
    def label_maps(self, images: Sequence[Image.Image], size: tuple[int, int] | None = None):
        """Dense label maps for a batch of PIL images, at ``size`` (H, W)."""
        torch = self.torch
        out: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            chunk = [im.convert("RGB") for im in images[start : start + self.batch_size]]
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits          # [B, C, h, w]
            target = size or (chunk[0].height, chunk[0].width)
            logits = torch.nn.functional.interpolate(
                logits, size=target, mode="bilinear", align_corners=False
            )
            out.extend(logits.argmax(dim=1).cpu().numpy().astype(np.int32))
        return out

    def __call__(self, image: Image.Image, size: tuple[int, int] | None = None) -> RegionMasks:
        return self.masks([image], size=size)[0]

    def masks(self, images: Sequence[Image.Image], size: tuple[int, int] | None = None) -> list[RegionMasks]:
        return [
            region_masks_from_labels(
                lm, self.id2label, self.sky_labels, self.ground_labels,
                source=f"segmentation:{self.model_name}",
            )
            for lm in self.label_maps(images, size=size)
        ]


# --------------------------------------------------------------------------- #
# 1b) Classical, download-free sky detector
# --------------------------------------------------------------------------- #
class EnergySkyMasker:
    """Sky detection by energy-function optimization (Shen & Wang, 2013).

    A learned segmenter is the better mask when its weights can be fetched, but
    it needs a model download, which is not available in every environment (and
    a mask that silently fails to load is worse than one that is a little
    coarser).  This is the classical alternative and it needs nothing but the
    image itself:

    1. A candidate sky/ground boundary ``b_t(x)`` is read off the vertical
       gradient image for a gradient threshold ``t``: for each column, the first
       row whose gradient exceeds ``t``.
    2. Each candidate partition is scored by the energy
       ``J = 1 / (gamma*|Sigma_s| + |Sigma_g| + gamma*lambda_s + lambda_g)``,
       the covariance determinants and principal eigenvalues of the RGB
       distributions of the two regions.  A good boundary makes both regions
       internally uniform, i.e. maximizes ``J``.
    3. The threshold that maximizes ``J`` is chosen by a coarse-to-fine sweep.

    Because a straight per-column boundary cannot follow towers, tents and tree
    crowns, the boundary is only used to *seed* two RGB Gaussians; every pixel
    is then assigned to the closer one (Mahalanobis), and the result is cleaned
    morphologically and restricted to components hanging from the top of the
    frame.  ``ground`` is everything else — this masker gives a two-way split,
    so ``other`` is empty (use the segmentation masker for the three-way one).

    Reference: Y. Shen and Q. Wang, "Sky Region Detection in a Single Image for
    Autonomous Ground Robot Navigation", Int. J. Advanced Robotic Systems, 2013.
    """

    def __init__(
        self,
        n_thresholds: int = 24,
        gamma: float = 2.0,
        refine: bool = True,
        morph_radius: int = 3,
        min_sky_frac: float = 0.02,
        max_sky_frac: float = 0.95,
        search_width: int = 160,
    ) -> None:
        self.n_thresholds = int(n_thresholds)
        self.gamma = float(gamma)
        self.refine = bool(refine)
        self.morph_radius = int(morph_radius)
        self.min_sky_frac = float(min_sky_frac)
        self.max_sky_frac = float(max_sky_frac)
        #: The threshold search only needs to locate a boundary, not resolve it,
        #: so it runs on a downscaled copy (~15x faster on a 640x480 frame); the
        #: colour models it produces are then applied at full resolution.
        self.search_width = int(search_width)

    # -- internals ---------------------------------------------------------- #
    @staticmethod
    def _gradient(rgb: np.ndarray) -> np.ndarray:
        from scipy import ndimage

        gray = rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
        gx = ndimage.sobel(gray, axis=1, mode="nearest")
        gy = ndimage.sobel(gray, axis=0, mode="nearest")
        return np.hypot(gx, gy)

    @staticmethod
    def _border_mask(grad: np.ndarray, threshold: float) -> np.ndarray:
        """Sky = rows above the first strong gradient in each column."""
        h, w = grad.shape
        strong = grad > threshold
        first = np.where(strong.any(axis=0), strong.argmax(axis=0), h)
        rows = np.arange(h)[:, None]
        return rows < first[None, :]

    def _energy(self, rgb: np.ndarray, sky: np.ndarray) -> float:
        n_sky = int(sky.sum())
        n_ground = int(sky.size - n_sky)
        if n_sky < 20 or n_ground < 20:
            return -np.inf
        total = 0.0
        for region, weight in ((sky, self.gamma), (~sky, 1.0)):
            values = rgb[region].astype(np.float64)
            cov = np.cov(values, rowvar=False) + np.eye(3) * 1e-6
            eig = np.linalg.eigvalsh(cov)
            total += weight * (float(np.linalg.det(cov)) + float(eig.max()))
        return 1.0 / total if total > 0 else -np.inf

    @staticmethod
    def _gaussian_models(sample_rgb: np.ndarray, seed: np.ndarray):
        """Fit an RGB Gaussian to each side of a seed partition."""
        values = sample_rgb.reshape(-1, 3).astype(np.float64)
        flat = seed.reshape(-1)
        if flat.sum() < 50 or (~flat).sum() < 50:
            return None
        models = []
        for region in (flat, ~flat):
            block = values[region]
            mean = block.mean(axis=0)
            cov = np.cov(block, rowvar=False) + np.eye(3) * 1e-3
            models.append((mean, np.linalg.inv(cov)))
        return models

    @staticmethod
    def _apply_models(rgb: np.ndarray, models) -> np.ndarray:
        values = rgb.reshape(-1, 3).astype(np.float64)
        distances = []
        for mean, inv in models:
            delta = values - mean
            distances.append(np.einsum("ij,jk,ik->i", delta, inv, delta))
        return (distances[0] < distances[1]).reshape(rgb.shape[:2])

    # -- public ------------------------------------------------------------- #
    def __call__(self, image: Image.Image, size: tuple[int, int] | None = None) -> RegionMasks:
        image = image.convert("RGB")
        rgb = np.asarray(image)

        if self.search_width and image.width > self.search_width:
            scale = self.search_width / image.width
            small = image.resize(
                (self.search_width, max(8, int(round(image.height * scale)))), Image.BILINEAR
            )
        else:
            small = image
        search_rgb = np.asarray(small)

        grad = self._gradient(search_rgb)
        lo, hi = float(np.percentile(grad, 5)), float(np.percentile(grad, 99.5))
        best_score, best_mask = -np.inf, None
        for t in np.linspace(lo, hi, self.n_thresholds):
            mask = self._border_mask(grad, t)
            fraction = mask.mean()
            if fraction < self.min_sky_frac or fraction > self.max_sky_frac:
                continue
            score = self._energy(search_rgb, mask)
            if score > best_score:
                best_score, best_mask = score, mask

        if best_mask is None:
            sky = np.zeros(rgb.shape[:2], dtype=bool)
        elif self.refine:
            models = self._gaussian_models(search_rgb, best_mask)
            sky = (self._apply_models(rgb, models) if models
                   else np.asarray(Image.fromarray(best_mask.astype(np.uint8) * 255)
                                   .resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 127)
        else:
            sky = np.asarray(Image.fromarray(best_mask.astype(np.uint8) * 255)
                             .resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 127

        sky = _binary_open_close(sky, self.morph_radius)
        sky = _keep_top_connected(sky, self.min_sky_frac)
        # Non-sky is NOT ground: this detector cannot separate grass from a
        # tree or a building, so everything below the skyline goes to `other`
        # and `provides_ground` stays False.
        masks = RegionMasks(
            sky=sky, ground=np.zeros_like(sky), other=~sky,
            source="energy_optimization", provides_ground=False,
            meta={"energy": float(best_score)},
        )
        return masks.resized(size) if size and tuple(size) != rgb.shape[:2] else masks

    def masks(self, images: Sequence[Image.Image], size: tuple[int, int] | None = None) -> list[RegionMasks]:
        return [self(im, size=size) for im in images]


def build_masker(kind: str = "auto", **kwargs):
    """Return a masker.  ``auto`` prefers segmentation and falls back offline.

    ``kind`` is one of ``segmentation`` / ``energy`` / ``auto``.  With ``auto``
    the segmentation model is attempted first and any failure (no torch, no
    network, no such checkpoint) falls back to :class:`EnergySkyMasker` with a
    printed warning, so a pipeline never silently runs without a mask.
    """
    seg_keys = {"model_name", "device", "batch_size", "sky_labels", "ground_labels"}
    energy_keys = {"n_thresholds", "gamma", "refine", "morph_radius",
                   "min_sky_frac", "max_sky_frac"}
    if kind == "energy":
        return EnergySkyMasker(**{k: v for k, v in kwargs.items() if k in energy_keys})
    if kind == "segmentation":
        return SegformerSkyMasker(**{k: v for k, v in kwargs.items() if k in seg_keys})
    if kind != "auto":
        raise ValueError("kind must be one of {'segmentation','energy','auto'}")
    try:
        return SegformerSkyMasker(**{k: v for k, v in kwargs.items() if k in seg_keys})
    except Exception as exc:  # noqa: BLE001 - any failure means "no segmenter here"
        print(f"[sky_mask] segmentation masker unavailable ({exc.__class__.__name__}: {exc}); "
              f"falling back to the energy-optimization masker.")
        return EnergySkyMasker(**{k: v for k, v in kwargs.items() if k in energy_keys})


# --------------------------------------------------------------------------- #
# 2) Depth-based geometric masker (independent reference)
# --------------------------------------------------------------------------- #
def load_depth_png(path: str | Path) -> np.ndarray:
    """Read a 16-bit aligned-to-color depth PNG as ``[H, W]`` uint16 millimetres."""
    with Image.open(path) as img:
        depth = np.asarray(img)
    if depth.ndim != 2:
        raise ValueError(f"expected a single-channel depth image, got shape {depth.shape}")
    return depth.astype(np.uint16)


def _binary_open_close(mask: np.ndarray, radius: int) -> np.ndarray:
    """Morphological open-then-close with a square structuring element.

    The array is edge-padded first: without it the erosion step treats the image
    border as background and shaves the top rows off the sky region, which then
    stops being "connected to the top of the frame".
    """
    if radius <= 0:
        return mask
    size = 2 * radius + 1
    try:  # OpenCV is an order of magnitude faster and is already a dependency
        import cv2

        k = np.ones((size, size), np.uint8)
        padded = np.pad(mask.astype(np.uint8), radius, mode="edge")
        padded = cv2.morphologyEx(padded, cv2.MORPH_OPEN, k)
        padded = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, k)
        return padded[radius:-radius, radius:-radius].astype(bool)
    except Exception:
        pass
    try:
        from scipy import ndimage
    except Exception:  # pragma: no cover - neither backend available
        return mask
    k = np.ones((size, size), dtype=bool)
    padded = np.pad(mask, radius, mode="edge")
    padded = ndimage.binary_opening(padded, structure=k)
    padded = ndimage.binary_closing(padded, structure=k)
    return padded[radius:-radius, radius:-radius]


def _keep_top_connected(mask: np.ndarray, min_area_frac: float) -> np.ndarray:
    """Keep only components that touch the top border and are big enough.

    Invalid depth (0) also occurs on wet/specular ground, dark mud and at object
    edges.  Real sky is a single large region hanging from the top of the frame,
    so this filter removes the ground-side false positives that a plain
    "depth invalid or far" test would otherwise pick up.
    """
    try:
        from scipy import ndimage
    except Exception:  # pragma: no cover
        return mask
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    areas = np.bincount(labels.reshape(-1), minlength=n + 1).astype(np.float64)
    touches_top = np.zeros(n + 1, dtype=bool)
    touches_top[np.unique(labels[0, :])] = True
    keep = touches_top & (areas >= min_area_frac * mask.size)
    keep[0] = False
    return keep[labels]


@dataclass
class DepthRegions:
    """Geometric regions from one aligned depth frame, all ``[H, W]`` boolean.

    IMPORTANT — depth cannot tell sky from a distant building.  A stereo/IR
    depth camera returns nothing for the sky, and equally nothing for an
    apartment tower 150 m away or a tree line past its range.  So the depth
    frame answers a *different*, and for this task arguably more useful,
    question than "is this sky":

    ``near``  valid return closer than ``near_m`` — the terrain whose apparent
              motion actually carries the vehicle's egomotion signal.
    ``mid``   valid return between ``near_m`` and ``far_m``.
    ``far``   invalid return or farther than ``far_m`` — sky *and* distant
              structure together; nothing here can support a metric motion cue.

    Use ``far`` as an independent check on an RGB sky mask (real sky must be a
    subset of it), and ``near`` as the geometric "relevant region" for a
    Right-for-the-Right-Reasons penalty.
    """

    near: np.ndarray
    mid: np.ndarray
    far: np.ndarray
    meta: dict = field(default_factory=dict)

    def area_fractions(self) -> dict[str, float]:
        n = float(self.near.size)
        return {
            "near": float(self.near.sum()) / n,
            "mid": float(self.mid.sum()) / n,
            "far": float(self.far.sum()) / n,
        }


def depth_regions(
    depth_mm: np.ndarray,
    near_m: float = 8.0,
    far_m: float = 20.0,
    morph_radius: int = 3,
) -> DepthRegions:
    """Split a 16-bit depth frame into near / mid / far (see :class:`DepthRegions`)."""
    depth = np.asarray(depth_mm)
    near_mm, far_mm = float(near_m) * 1000.0, float(far_m) * 1000.0
    invalid = depth == 0
    saturated = depth == np.iinfo(np.uint16).max
    valid = ~invalid
    near = valid & (depth <= near_mm)
    far = invalid | (depth >= far_mm)
    mid = ~(near | far)
    near = _binary_open_close(near, morph_radius)
    far = _binary_open_close(far, morph_radius)
    mid = ~(near | far)
    return DepthRegions(
        near=near, mid=mid, far=far,
        meta={
            "near_m": float(near_m),
            "far_m": float(far_m),
            "invalid_frac": float(invalid.mean()),
            "saturated_frac": float(saturated.mean()),
            "valid_frac": float(valid.mean()),
        },
    )


def depth_sky_mask(
    depth_mm: np.ndarray,
    far_m: float = 20.0,
    treat_invalid_as_sky: bool = True,
    morph_radius: int = 3,
    require_top_connected: bool = True,
    min_area_frac: float = 0.01,
) -> RegionMasks:
    """"Sky-or-farther" mask from a depth frame, as a :class:`RegionMasks`.

    This is the ``far`` region of :func:`depth_regions`, optionally restricted to
    components that hang from the top of the frame.  It is an *upper bound* on
    the sky, not the sky itself — distant buildings and tree lines fall inside
    it.  Kept because "real sky must be contained in this" is the cheapest
    independent test of an RGB sky mask.
    """
    depth = np.asarray(depth_mm)
    far_mm = float(far_m) * 1000.0
    invalid = depth == 0
    far = depth >= far_mm
    raw = (invalid & bool(treat_invalid_as_sky)) | far
    mask = _binary_open_close(raw, morph_radius)
    if require_top_connected:
        mask = _keep_top_connected(mask, min_area_frac)
    return RegionMasks(
        sky=mask,
        ground=np.zeros_like(mask),
        other=~mask,
        source="depth_far",
        provides_ground=False,
        meta={
            "far_m": float(far_m),
            "invalid_frac": float(invalid.mean()),
            "far_frac": float(far.mean()),
            "raw_far_frac": float(raw.mean()),
            "clean_far_frac": float(mask.mean()),
        },
    )


# --------------------------------------------------------------------------- #
# 3) Legacy heuristic, kept for the ablation
# --------------------------------------------------------------------------- #
def half_frame_mask(height: int, width: int, horizon_frac: float = 0.5) -> RegionMasks:
    """The v2 heuristic: everything above ``horizon_frac * H`` is called sky."""
    cut = max(1, int(round(height * horizon_frac)))
    sky = np.zeros((height, width), dtype=bool)
    sky[:cut, :] = True
    return RegionMasks(
        sky=sky, ground=np.zeros_like(sky), other=~sky,
        source="half_frame", provides_ground=False,
        meta={"horizon_frac": float(horizon_frac)},
    )


# --------------------------------------------------------------------------- #
# Agreement metrics
# --------------------------------------------------------------------------- #
def mask_agreement(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Overlap statistics between a predicted and a reference boolean mask.

    ``pred`` is scored against ``ref``: precision is "of the pixels called sky
    by ``pred``, how many are sky in ``ref``", recall the converse.
    """
    pred = np.asarray(pred, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {ref.shape}")
    inter = float((pred & ref).sum())
    union = float((pred | ref).sum())
    p_sum, r_sum = float(pred.sum()), float(ref.sum())
    n = float(pred.size)
    return {
        "iou": inter / union if union else float("nan"),
        "dice": 2 * inter / (p_sum + r_sum) if (p_sum + r_sum) else float("nan"),
        "precision": inter / p_sum if p_sum else float("nan"),
        "recall": inter / r_sum if r_sum else float("nan"),
        "pixel_accuracy": float((pred == ref).sum()) / n,
        "pred_area_frac": p_sum / n,
        "ref_area_frac": r_sum / n,
    }


# --------------------------------------------------------------------------- #
# Cache I/O — masks live on disk as a single-channel PNG of region codes
# --------------------------------------------------------------------------- #
def write_mask_png(path: str | Path, masks: RegionMasks) -> Path:
    """Store a :class:`RegionMasks` as an 8-bit PNG (0 sky / 1 ground / 2 other)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(masks.label_image(), mode="L").save(path, optimize=True)
    return path


def read_mask_png(path: str | Path, size: tuple[int, int] | None = None,
                  provides_ground: bool = True) -> RegionMasks:
    """Read a cached region-code PNG, optionally nearest-resized to ``size`` (H, W).

    ``provides_ground`` comes from the cache's ``index.json`` — a PNG of region
    codes cannot carry it, and guessing from "are there any ground pixels?"
    would silently be wrong on a frame that simply has no visible ground.
    """
    with Image.open(path) as img:
        if size is not None:
            img = img.resize((size[1], size[0]), Image.NEAREST)
        labels = np.asarray(img.convert("L"))
    return RegionMasks(
        sky=labels == REGION_SKY,
        ground=labels == REGION_GROUND,
        other=labels == REGION_OTHER,
        source="cache",
        provides_ground=bool(provides_ground),
    )


def cache_provides_ground(cache_dir: str | Path) -> bool:
    """Read ``provides_ground`` out of a mask cache's index.json (default False)."""
    index = Path(cache_dir) / "index.json"
    if not index.is_file():
        return False
    try:
        return bool(json.loads(index.read_text(encoding="utf-8")).get("provides_ground", False))
    except Exception:
        return False


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a boolean mask to ``size`` (H, W)."""
    img = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    return np.asarray(img.resize((size[1], size[0]), Image.NEAREST)) > 127


def mask_cache_path(cache_dir: str | Path, experiment_id: str, rgb_path: str | Path) -> Path:
    """Deterministic on-disk location of the cached mask for one RGB frame."""
    return Path(cache_dir) / str(experiment_id) / (Path(rgb_path).stem + ".png")


def summarize_area_fractions(all_masks: Iterable[RegionMasks]) -> dict[str, float]:
    rows = [m.area_fractions() for m in all_masks]
    if not rows:
        return {}
    out: dict[str, float] = {}
    for key in REGION_NAMES:
        values = np.asarray([r[key] for r in rows], dtype=float)
        out[f"{key}_area_mean"] = float(values.mean())
        out[f"{key}_area_std"] = float(values.std())
    out["n_frames"] = float(len(rows))
    return out


def dump_json(path: str | Path, payload: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

# --------------------------------------------------------------------------- #
# Image repair, and trusting a mask only when it is plausible
# --------------------------------------------------------------------------- #
#: Where ``precompute_sky_masks.py`` records the per-frame area of every region and
#: whether the mask is trusted.  Kept beside the PNGs because a region-code image
#: cannot carry it.
MASK_QUALITY_FILE = "quality.csv"


def enhance_for_segmentation(image: Image.Image) -> Image.Image:
    """Grey-world white balance + CLAHE, for frames a segmenter cannot read.

    Part of the collection is washed out and colour-cast: the terrain is pale and
    low-contrast, and the segmenter labels it anything but ground — on those frames
    it finds 4-8% ground where half the image is grass.  Since it was trained on
    ordinarily-exposed photographs, the cheapest fix is to hand it an ordinarily
    exposed photograph.

    This touches ONLY the segmenter's input.  The model keeps seeing the original
    frame, so nothing about the task changes; only the annotation improves.
    """
    rgb = np.asarray(image.convert("RGB"))
    try:
        import cv2
    except Exception:  # pragma: no cover - fall back to a plain grey-world balance
        values = rgb.astype(np.float32)
        means = values.reshape(-1, 3).mean(axis=0) + 1e-6
        values = values * (means.mean() / means)
        return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))

    values = rgb.astype(np.float32)
    means = values.reshape(-1, 3).mean(axis=0) + 1e-6
    values = np.clip(values * (means.mean() / means), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(values, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def mask_is_plausible(masks: "RegionMasks", min_ground: float = 0.15) -> bool:
    """Would a forward-driving ground vehicle really see this little terrain?

    On the depth-validated part of the collection the ground is 47% +- 15% of the
    frame and the near-field alone never drops below 41%, so a mask claiming under
    ``min_ground`` is far more likely to be a segmentation failure than a real view.
    Such frames are still trained on — only their *guidance* is switched off, since
    penalizing "everything but this 4% sliver" is worse than not guiding at all.
    """
    return bool(masks.provides_ground and masks.ground.mean() >= float(min_ground))


def load_mask_quality(cache_dir: str | Path) -> dict[tuple[str, str], bool]:
    """``{(experiment_id, rgb stem): trusted}`` from a cache's quality.csv."""
    path = Path(cache_dir) / MASK_QUALITY_FILE
    if not path.is_file():
        return {}
    import csv

    out: dict[tuple[str, str], bool] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out[(row["experiment_id"], row["stem"])] = row["trusted"].strip().lower() in (
                "1", "true", "yes")
    return out
