from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .manifest import NUMERIC_INPUT_COLUMNS, TARGET_COLUMNS, validate_normalization_schema
from .sky_mask import load_mask_quality, mask_cache_path, read_mask_png


def sort_experiment_rows(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = ["experiment_id"]
    for name in ("source_row_index", "sample_index", "time_sec", "target_time_ns"):
        if name in frame:
            sort_columns.append(name)
            break
    return frame.sort_values(sort_columns).reset_index(drop=True)


def sequence_windows(
    frame: pd.DataFrame, sequence_length: int, image_update_interval: int = 1
) -> list[tuple[int, int, int]]:
    """Return (window end, RGB row, image age) without crossing data gaps.

    The RGB row is refreshed at the first row of every contiguous run and then
    every M rows.  It is therefore never newer than the prediction target.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if image_update_interval < 1:
        raise ValueError("image_update_interval must be at least 1")
    windows: list[tuple[int, int, int]] = []
    for _, group in frame.groupby("experiment_id", sort=False):
        positions = group.index.to_numpy()
        if "source_row_index" in frame:
            sample_ids = frame.loc[positions, "source_row_index"].to_numpy(dtype=np.int64)
            breaks = np.flatnonzero(np.diff(sample_ids) != 1) + 1
            runs = np.split(positions, breaks)
        elif "sample_index" in frame:
            sample_ids = frame.loc[positions, "sample_index"].to_numpy(dtype=np.int64)
            breaks = np.flatnonzero(np.diff(sample_ids) != 1) + 1
            runs = np.split(positions, breaks)
        else:
            runs = [positions]
        for run in runs:
            for offset in range(sequence_length - 1, len(run)):
                image_age = offset % image_update_interval
                image_index = int(run[offset - image_age])
                windows.append((int(run[offset]), image_index, int(image_age)))
    return windows


def sequence_end_indices(frame: pd.DataFrame, sequence_length: int) -> list[int]:
    """Backward-compatible endpoint-only view used by older callers."""
    return [end for end, _, _ in sequence_windows(frame, sequence_length, 1)]


def load_region_tensors(
    sky_mask_dir: str | Path | None,
    experiment_id: str,
    rgb_path: str | Path,
    image_size: tuple[int, int],
    quality: dict[tuple[str, str], bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a cached region mask as ``(sky, ground, has_mask)`` tensors.

    ``sky`` and ``ground`` are ``[1, H, W]`` float masks aligned with the model
    input; ``has_mask`` is a 0/1 scalar so a batch can mix masked and unmasked
    frames without silently penalizing an all-zero region.

    ``quality`` is the cache's per-frame trusted flag.  A frame the segmenter
    clearly failed on returns ``has_mask = 0``: it still trains normally, it just
    contributes nothing to the attention guidance, because pushing saliency out of
    "everything except this 4% sliver" is worse than not guiding it at all.
    """
    # When no cache is configured the placeholders are 1x1x1, not HxW: plain
    # training would otherwise carry ~19 MB of zeros per batch for nothing.
    # Every consumer of these tensors first checks that a cache exists.
    if not sky_mask_dir:
        tiny = torch.zeros(1, 1, 1, dtype=torch.float32)
        return tiny, tiny.clone(), torch.tensor(0.0)
    empty = torch.zeros(1, *image_size, dtype=torch.float32)
    path = mask_cache_path(sky_mask_dir, experiment_id, rgb_path)
    if not Path(path).is_file():
        return empty, empty.clone(), torch.tensor(0.0)
    if quality and not quality.get((str(experiment_id), Path(rgb_path).stem), True):
        return empty, empty.clone(), torch.tensor(0.0)
    regions = read_mask_png(path, size=image_size)
    sky = torch.from_numpy(regions.sky.astype(np.float32)).unsqueeze(0)
    ground = torch.from_numpy(regions.ground.astype(np.float32)).unsqueeze(0)
    return sky, ground, torch.tensor(1.0)


class MotionDataset(Dataset):
    """Windows of K numeric steps + one cached RGB frame -> next-step motion.

    ``sky_mask_dir`` optionally attaches the pre-computed per-frame region mask
    (see ``precompute_sky_masks.py``) to every item, which is what the
    attention-guided fine-tune penalizes.  The augmentations are photometric
    only (colour jitter, blur) — no crop, flip or rotation — so a cached mask
    stays pixel-aligned with the augmented image.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        normalization_json: str | Path,
        split: str,
        image_size: tuple[int, int] = (224, 224),
        augment: bool = False,
        sequence_length: int = 6,
        image_update_interval: int = 1,
        sky_mask_dir: str | Path | None = None,
    ) -> None:
        frame = pd.read_csv(manifest_csv)
        self.frame = sort_experiment_rows(frame[frame["split"] == split].copy())
        self.sequence_length = int(sequence_length)
        self.image_update_interval = int(image_update_interval)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.sky_mask_dir = Path(sky_mask_dir) if sky_mask_dir else None
        self.mask_quality = load_mask_quality(self.sky_mask_dir) if self.sky_mask_dir else {}
        self.windows = sequence_windows(
            self.frame, self.sequence_length, self.image_update_interval
        )
        if not self.windows:
            raise ValueError(
                f"No valid K={self.sequence_length} windows found for split={split!r}"
            )
        norm = json.loads(Path(normalization_json).read_text(encoding="utf-8"))
        validate_normalization_schema(norm)
        self.input_mean = np.asarray(norm["numeric_input"]["mean"], dtype=np.float32)
        self.input_std = np.asarray(norm["numeric_input"]["std"], dtype=np.float32)
        self.target_mean = np.asarray(norm["target"]["mean"], dtype=np.float32)
        self.target_std = np.asarray(norm["target"]["std"], dtype=np.float32)

        ops: list = [transforms.Resize(self.image_size)]
        if augment:
            ops.extend([
                transforms.ColorJitter(0.15, 0.15, 0.10, 0.03),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.10),
            ])
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        end, image_index, image_age = self.windows[index]
        start = end - self.sequence_length + 1
        row = self.frame.iloc[end]
        image_row = self.frame.iloc[image_index]
        with Image.open(image_row["rgb_abs_path"]) as image:
            image_tensor = self.transform(image.convert("RGB"))
        numeric = self.frame.iloc[start : end + 1][NUMERIC_INPUT_COLUMNS].to_numpy(dtype=np.float32)
        target = row[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        numeric = (numeric - self.input_mean) / self.input_std
        target = (target - self.target_mean) / self.target_std
        sky, ground, has_mask = load_region_tensors(
            self.sky_mask_dir, str(image_row["experiment_id"]),
            image_row["rgb_abs_path"], self.image_size, self.mask_quality,
        )
        return {
            "image": image_tensor,
            "numeric": torch.from_numpy(numeric),
            "image_age": torch.tensor(
                image_age / max(self.image_update_interval - 1, 1), dtype=torch.float32
            ),
            "target": torch.from_numpy(target),
            "sky_mask": sky,
            "ground_mask": ground,
            "has_mask": has_mask,
            "experiment_id": str(row["experiment_id"]),
            "source_row_index": int(row["source_row_index"]) if "source_row_index" in row else end,
            "image_source_row_index": int(image_row["source_row_index"]) if "source_row_index" in image_row else image_index,
        }

    def denormalize_target(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.target_mean, device=value.device)
        std = torch.as_tensor(self.target_std, device=value.device)
        return value * std + mean
