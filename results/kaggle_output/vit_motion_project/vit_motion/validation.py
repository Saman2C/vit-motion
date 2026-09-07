from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .dataset import sequence_windows, sort_experiment_rows
from .manifest import NUMERIC_INPUT_COLUMNS, TARGET_COLUMNS, validate_normalization_schema
from .model import ViTMotionModel


DISPLAY_INPUTS = [
    ("current_body_dx", "Current body dx", "m"),
    ("current_body_dy", "Current body dy", "m"),
    ("current_delta_yaw", "Current delta yaw", "rad"),
    ("left_mps", "Left track speed", "m/s"),
    ("right_mps", "Right track speed", "m/s"),
]
DISPLAY_OUTPUTS = [
    ("next_body_dx", "Next body dx", "m"),
    ("next_body_dy", "Next body dy", "m"),
    ("next_delta_yaw", "Next delta yaw", "rad"),
]


def load_normalization(path: str | Path) -> dict:
    normalization = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_normalization_schema(normalization)
    return normalization


def load_experiment_frame(manifest_path: str | Path, experiment_id: str | None) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(manifest_path)
    ids = sorted(frame["experiment_id"].astype(str).unique())
    if not ids:
        raise ValueError("The manifest contains no experiments.")
    if experiment_id is None:
        print("Available experiments:")
        for i, item in enumerate(ids, 1):
            split = frame.loc[frame["experiment_id"].astype(str) == item, "split"].iloc[0]
            count = int((frame["experiment_id"].astype(str) == item).sum())
            print(f"  {i:2d}. {item}  ({split}, {count} rows)")
        choice = input("Select experiment number or ID: ").strip()
        experiment_id = ids[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(ids) else choice
    experiment_id = str(experiment_id)
    selected = frame[frame["experiment_id"].astype(str) == experiment_id].copy()
    if selected.empty:
        raise ValueError(f"Experiment {experiment_id!r} not found. Choices: {', '.join(ids)}")
    sort_columns = [x for x in ("source_row_index", "sample_index", "time_sec", "target_time_ns") if x in selected]
    if sort_columns:
        selected = selected.sort_values(sort_columns[0])
    return selected.reset_index(drop=True), experiment_id


def time_axis(frame: pd.DataFrame, sample_period_sec: float) -> np.ndarray:
    if "source_row_index" in frame:
        values = frame["source_row_index"].to_numpy(dtype=np.float64)
        return (values - values[0]) * sample_period_sec
    return np.arange(len(frame), dtype=np.float64) * sample_period_sec


class ExperimentDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, normalization: dict, image_size: tuple[int, int],
                 sequence_length: int, image_update_interval: int, sample_period_sec: float):
        self.frame = sort_experiment_rows(frame.copy())
        self.sequence_length = int(sequence_length)
        self.image_update_interval = int(image_update_interval)
        self.sample_period_sec = float(sample_period_sec)
        self.windows = sequence_windows(
            self.frame, self.sequence_length, self.image_update_interval
        )
        if not self.windows:
            raise ValueError(f"Experiment has no valid K={self.sequence_length} sequence windows.")
        self.input_mean = np.asarray(normalization["numeric_input"]["mean"], dtype=np.float32)
        self.input_std = np.asarray(normalization["numeric_input"]["std"], dtype=np.float32)
        self.target_mean = np.asarray(normalization["target"]["mean"], dtype=np.float32)
        self.target_std = np.asarray(normalization["target"]["std"], dtype=np.float32)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        end, image_index, image_age = self.windows[index]
        start = end - self.sequence_length + 1
        row = self.frame.iloc[end]
        image_row = self.frame.iloc[image_index]
        with Image.open(image_row["rgb_abs_path"]) as image:
            image_tensor = self.transform(image.convert("RGB"))
        numeric = self.frame.iloc[start : end + 1][NUMERIC_INPUT_COLUMNS].to_numpy(dtype=np.float32)
        return {
            "image": image_tensor,
            "numeric": torch.from_numpy((numeric - self.input_mean) / self.input_std),
            "image_age": torch.tensor(
                image_age / max(self.image_update_interval - 1, 1), dtype=torch.float32
            ),
        }

    def result_frame(self) -> pd.DataFrame:
        result = self.frame.iloc[[x[0] for x in self.windows]].copy().reset_index(drop=True)
        result["model_rgb_abs_path"] = [
            self.frame.iloc[x[1]]["rgb_abs_path"] for x in self.windows
        ]
        result["image_age_steps"] = [x[2] for x in self.windows]
        return result

    def result_time_axis(self) -> np.ndarray:
        full_time = time_axis(self.frame, self.sample_period_sec)
        return full_time[np.asarray([x[0] for x in self.windows], dtype=np.int64)]


def load_model(checkpoint_path: str | Path, config: dict, device: torch.device) -> ViTMotionModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get("config", {}).get("model", config["model"])
    # The checkpoint already contains encoder weights; avoid an unnecessary download.
    model_cfg = dict(model_cfg)
    if "sequence_length" not in model_cfg:
        raise ValueError(
            "This is a legacy single-step checkpoint. Train a new checkpoint with the "
            "temporal model before running sequence validation."
        )
    model_cfg["pretrained"] = False
    model = ViTMotionModel(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.inference_mode()
def predict_frame(frame: pd.DataFrame, normalization: dict, config: dict, checkpoint_path: str | Path,
                  batch_size: int = 16, num_workers: int = 0) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = tuple(int(x) for x in config["data"]["image_size"])
    model = load_model(checkpoint_path, config, device)
    sequence_length = model.sequence_length
    image_update_interval = model.image_update_interval
    sample_period_sec = float(config["data"].get("sample_period_sec", 0.1))
    dataset = ExperimentDataset(
        frame, normalization, image_size, sequence_length, image_update_interval,
        sample_period_sec,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    predicted = []
    for batch in tqdm(loader, desc=f"Inference ({device})"):
        output = model(
            batch["image"].to(device, non_blocking=True),
            batch["numeric"].to(device, non_blocking=True),
            batch["image_age"].to(device, non_blocking=True),
        )
        predicted.append(output.cpu().numpy())
    values = np.concatenate(predicted, axis=0)
    target_mean = np.asarray(normalization["target"]["mean"], dtype=np.float32)
    target_std = np.asarray(normalization["target"]["std"], dtype=np.float32)
    values = values * target_std + target_mean
    result = dataset.result_frame()
    result["time_from_start_sec"] = dataset.result_time_axis()
    result.attrs["sequence_length"] = sequence_length
    result.attrs["image_update_interval"] = image_update_interval
    for i, (column, _, _) in enumerate(DISPLAY_OUTPUTS):
        result[f"predicted_{column}"] = values[:, i]
        result[f"error_{column}"] = values[:, i] - result[column].to_numpy(dtype=np.float32)
    return result


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = None if denominator < 1e-15 else float(1.0 - np.sum(error ** 2) / denominator)
    fit = None if denominator < 1e-15 else float(100.0 * (1.0 - np.linalg.norm(error) / np.linalg.norm(actual - actual.mean())))
    return {"mae": mae, "rmse": rmse, "r2": r2, "fit_percent": fit}
