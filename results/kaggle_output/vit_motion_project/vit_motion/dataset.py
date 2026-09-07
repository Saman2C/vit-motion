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


class MotionDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        normalization_json: str | Path,
        split: str,
        image_size: tuple[int, int] = (224, 224),
        augment: bool = False,
        sequence_length: int = 6,
        image_update_interval: int = 1,
    ) -> None:
        frame = pd.read_csv(manifest_csv)
        self.frame = sort_experiment_rows(frame[frame["split"] == split].copy())
        self.sequence_length = int(sequence_length)
        self.image_update_interval = int(image_update_interval)
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

        ops: list = [transforms.Resize(image_size)]
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
        return {
            "image": image_tensor,
            "numeric": torch.from_numpy(numeric),
            "image_age": torch.tensor(
                image_age / max(self.image_update_interval - 1, 1), dtype=torch.float32
            ),
            "target": torch.from_numpy(target),
            "experiment_id": str(row["experiment_id"]),
            "source_row_index": int(row["source_row_index"]) if "source_row_index" in row else end,
            "image_source_row_index": int(image_row["source_row_index"]) if "source_row_index" in image_row else image_index,
        }

    def denormalize_target(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.target_mean, device=value.device)
        std = torch.as_tensor(self.target_std, device=value.device)
        return value * std + mean
