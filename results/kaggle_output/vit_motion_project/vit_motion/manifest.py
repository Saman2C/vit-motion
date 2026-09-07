from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

SOURCE_COLUMNS = [
    "rgb_path",
    "current_body_dx",
    "current_body_dy",
    "current_delta_yaw",
    "left_rpm",
    "right_rpm",
    "next_body_dx",
    "next_body_dy",
    "next_delta_yaw",
]
REQUIRED_COLUMNS = set(SOURCE_COLUMNS)

NUMERIC_INPUT_COLUMNS = [
    "current_body_dx",
    "current_body_dy",
    "current_delta_yaw",
    "left_mps",
    "right_mps",
]
TARGET_COLUMNS = ["next_body_dx", "next_body_dy", "next_delta_yaw"]
DATA_SCHEMA_VERSION = "clean_9col_rpm_to_mps_v0.2.1"


@dataclass
class ExperimentResult:
    experiment_id: str
    csv_path: str
    total_rows: int
    valid_rows: int
    rejection_counts: dict[str, int]
    error: str | None = None


def discover_experiments(root: Path) -> list[Path]:
    return sorted(root.rglob("samples.csv"))


def _stable_unit_interval(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_experiment_splits(
    experiment_ids: Iterable[str], ratios: tuple[float, float, float], seed: int
) -> dict[str, str]:
    ids = sorted(set(experiment_ids))
    if not ids:
        return {}
    if len(ids) == 1:
        return {ids[0]: "train"}
    ranked = sorted(ids, key=lambda x: _stable_unit_interval(x, seed))
    n = len(ranked)
    n_test = max(1, round(n * ratios[2])) if n >= 3 else 1
    n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
    while n_test + n_val >= n:
        if n_val > 0:
            n_val -= 1
        else:
            n_test -= 1
    result = {x: "train" for x in ranked}
    for x in ranked[:n_test]:
        result[x] = "test"
    for x in ranked[n_test : n_test + n_val]:
        result[x] = "val"
    return result


def inspect_rgb(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing_rgb"
    try:
        with Image.open(path) as img:
            img.verify()
        return True, ""
    except Exception:
        return False, "corrupt_rgb"


def process_experiment(csv_path: Path, data_cfg: dict) -> tuple[pd.DataFrame, ExperimentResult]:
    exp_dir = csv_path.parent
    exp_id = exp_dir.name
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return pd.DataFrame(), ExperimentResult(exp_id, str(csv_path), 0, 0, {}, str(exc))

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        msg = f"missing columns: {', '.join(missing)}"
        return pd.DataFrame(), ExperimentResult(exp_id, str(csv_path), len(df), 0, {}, msg)

    reasons: list[list[str]] = [[] for _ in range(len(df))]

    def reject(mask: pd.Series | np.ndarray, reason: str) -> None:
        for i in np.flatnonzero(np.asarray(mask)):
            reasons[i].append(reason)

    # The source CSV stays compact and human-auditable in RPM.  Conversion to
    # model-facing track speed happens here, before normalization/windowing.
    numeric_needed = [
        "current_body_dx",
        "current_body_dy",
        "current_delta_yaw",
        "left_rpm",
        "right_rpm",
        *TARGET_COLUMNS,
    ]
    finite = np.isfinite(df[numeric_needed].to_numpy(dtype=np.float64)).all(axis=1)
    reject(~finite, "nonfinite_numeric")

    rgb_abs = [(exp_dir / str(p)).resolve() for p in df[data_cfg["rgb_column"]]]
    for i, path in enumerate(rgb_abs):
        ok, reason = inspect_rgb(path)
        if not ok:
            reasons[i].append(reason)

    # Keep only the clean model-facing schema. Extra legacy metadata, if it
    # happens to be present in a CSV, is deliberately not propagated.
    out = df[SOURCE_COLUMNS].copy()
    rpm_to_mps_gain = float(data_cfg["rpm_to_mps_gain"])
    if not math.isfinite(rpm_to_mps_gain) or rpm_to_mps_gain <= 0:
        raise ValueError("data.rpm_to_mps_gain must be a positive finite value")
    out["left_mps"] = out["left_rpm"].astype(float) / rpm_to_mps_gain
    out["right_mps"] = out["right_rpm"].astype(float) / rpm_to_mps_gain
    # The clean nine-column CSV intentionally has no old timing/sample metadata.
    # Preserve its row position internally so sequence windows never bridge a
    # row rejected for a bad image or non-finite numeric value.
    out["source_row_index"] = np.arange(len(out), dtype=np.int64)
    out["experiment_id"] = exp_id
    out["rgb_abs_path"] = [str(p) for p in rgb_abs]
    # Current RGB filenames are nanosecond timestamps. Preserve that timestamp
    # as manifest metadata when available; it is useful for synchronization and
    # debugging but is deliberately NOT a model input or a substitute for the
    # control-loop timestamp.
    out["rgb_timestamp_ns"] = [
        int(Path(str(p)).stem) if Path(str(p)).stem.isdigit() else np.nan
        for p in df[data_cfg["rgb_column"]]
    ]
    out["rejection_reason"] = [";".join(x) for x in reasons]
    valid = out["rejection_reason"].eq("")
    counts: dict[str, int] = {}
    for row_reasons in reasons:
        for reason in row_reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return out.loc[valid].copy(), ExperimentResult(
        exp_id, str(csv_path), len(out), int(valid.sum()), counts
    )


def calculate_normalization(train_df: pd.DataFrame) -> dict:
    if train_df.empty:
        raise ValueError("No valid training rows available for normalization.")

    def stats(columns: list[str]) -> dict:
        values = train_df[columns].to_numpy(dtype=np.float64)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return {"columns": columns, "mean": mean.tolist(), "std": std.tolist()}

    return {
        "schema_version": DATA_SCHEMA_VERSION,
        "numeric_input": stats(NUMERIC_INPUT_COLUMNS),
        "target": stats(TARGET_COLUMNS),
    }


def validate_normalization_schema(normalization: dict) -> None:
    version = normalization.get("schema_version")
    input_columns = normalization.get("numeric_input", {}).get("columns")
    target_columns = normalization.get("target", {}).get("columns")
    if (
        version != DATA_SCHEMA_VERSION
        or input_columns != NUMERIC_INPUT_COLUMNS
        or target_columns != TARGET_COLUMNS
    ):
        raise ValueError(
            "Normalization schema does not match v0.2.1 RPM-to-m/s data. "
            "Delete/recreate artifacts/manifest with inspect_dataset.py before training."
        )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
