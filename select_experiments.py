#!/usr/bin/env python3
"""Choose a subset of experiments that keeps the motion diversity, not just the count.

If the wet side has to shrink to meet the dry side, dropping runs at random throws
away whatever driving pattern happened to live in the runs it removed.  This picks
the subset by *coverage* instead: describe every run by what the vehicle actually
did in it, then greedily take the run furthest from everything already taken
(farthest-point sampling).  The result spans the same space with fewer runs.

Features per run, all from ``samples.csv`` — the driving, not the scenery:

    dx_mean, dx_std          forward step size and how much it varies
    yaw_std, yaw_abs_p95     how much turning, and how hard at the extreme
    speed_p50, speed_p95     |L+R| command magnitude
    reverse_frac             share of rows commanding backwards
    counter_rotate_frac      share with L and R in opposite directions — the
                             regime the collection is thinnest in, so a run that
                             has some of it is worth keeping
    saturated_frac           share at the +-4000 RPM clip

    python select_experiments.py --data-root data/processed --keep 20 \
        --only wet --screening analysis/screening/experiment_screening.csv \
        --output-dir analysis/selection

Prints the chosen ids ready to paste, and what the subset gives up against the
full set on each feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#6b7280"
FEATURES = ["dx_mean", "dx_std", "yaw_std", "yaw_abs_p95", "speed_p50", "speed_p95",
            "reverse_frac", "counter_rotate_frac", "saturated_frac"]


def describe(csv_path: Path) -> dict[str, float] | None:
    frame = pd.read_csv(csv_path)
    needed = {"next_body_dx", "next_delta_yaw", "left_rpm", "right_rpm"}
    if not needed.issubset(frame.columns) or frame.empty:
        return None
    left = frame["left_rpm"].to_numpy(float)
    right = frame["right_rpm"].to_numpy(float)
    total = np.abs(left + right)
    return {
        "experiment_id": csv_path.parent.name,
        "rows": int(len(frame)),
        "dx_mean": float(frame["next_body_dx"].mean()),
        "dx_std": float(frame["next_body_dx"].std()),
        "yaw_std": float(frame["next_delta_yaw"].std()),
        "yaw_abs_p95": float(np.percentile(np.abs(frame["next_delta_yaw"]), 95)),
        "speed_p50": float(np.percentile(total, 50)),
        "speed_p95": float(np.percentile(total, 95)),
        "reverse_frac": float(((left + right) < 0).mean()),
        "counter_rotate_frac": float((left * right < 0).mean()),
        "saturated_frac": float(((np.abs(left) >= 3900) | (np.abs(right) >= 3900)).mean()),
    }


def farthest_point(matrix: np.ndarray, keep: int, seeded: list[int]) -> list[int]:
    """Greedy max-min selection: each pick is the point furthest from the chosen set."""
    chosen = list(seeded)
    if not chosen:
        chosen = [int(np.argmax(np.linalg.norm(matrix - matrix.mean(axis=0), axis=1)))]
    while len(chosen) < min(keep, len(matrix)):
        distances = np.min(
            np.linalg.norm(matrix[:, None, :] - matrix[None, chosen, :], axis=2), axis=1)
        distances[chosen] = -1.0
        chosen.append(int(np.argmax(distances)))
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", nargs="+", required=True)
    ap.add_argument("--keep", type=int, required=True, help="How many runs to keep.")
    ap.add_argument("--only", choices=["wet", "dry", "all"], default="all",
                    help="Restrict the candidate pool to one condition.")
    ap.add_argument("--exclude", default="", help="Comma list of ids never to pick (held-out runs).")
    ap.add_argument("--require", default="", help="Comma list of ids always to pick.")
    ap.add_argument("--screening", help="experiment_screening.csv; flagged runs are dropped first.")
    ap.add_argument("--output-dir", default="analysis/selection")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for root in args.data_root:
        for csv_path in sorted(Path(root).expanduser().resolve().glob("*/samples.csv")):
            row = describe(csv_path)
            if row:
                records.append(row)
    if not records:
        raise SystemExit("No usable samples.csv found.")
    df = pd.DataFrame(records)
    df["condition"] = np.where(df["experiment_id"].str.contains("wet"), "wet", "dry")
    df["day"] = df["experiment_id"].str.split("_").str[0]
    print(f"{len(df)} candidate runs")

    if args.screening and Path(args.screening).is_file():
        screen = pd.read_csv(args.screening)[["experiment_id", "usable", "flags"]]
        df = df.merge(screen, on="experiment_id", how="left")
        dropped = df.loc[df["usable"] == False, "experiment_id"].tolist()   # noqa: E712
        if dropped:
            print(f"  {len(dropped)} dropped as image-quality failures: {', '.join(dropped[:8])}"
                  + ("..." if len(dropped) > 8 else ""))
        df = df[df["usable"] != False].reset_index(drop=True)               # noqa: E712

    excluded = {x.strip() for x in args.exclude.split(",") if x.strip()}
    required = [x.strip() for x in args.require.split(",") if x.strip()]
    pool = df[~df["experiment_id"].isin(excluded)].reset_index(drop=True)
    if args.only != "all":
        held = pool[pool["condition"] != args.only]
        pool = pool[pool["condition"] == args.only].reset_index(drop=True)
        print(f"  candidate pool restricted to {args.only}: {len(pool)} runs "
              f"({len(held)} of the other condition are kept untouched)")
    if pool.empty:
        raise SystemExit("Nothing left to choose from.")

    values = pool[FEATURES].to_numpy(float)
    values = (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-9)
    seeded = [int(pool.index[pool["experiment_id"] == r][0])
              for r in required if (pool["experiment_id"] == r).any()]
    chosen_idx = farthest_point(values, args.keep, seeded)
    chosen = pool.loc[chosen_idx, "experiment_id"].tolist()

    comparison = pd.DataFrame({
        "full_pool": pool[FEATURES].mean(),
        "chosen": pool.loc[chosen_idx, FEATURES].mean(),
        "full_range": pool[FEATURES].max() - pool[FEATURES].min(),
        "chosen_range": pool.loc[chosen_idx, FEATURES].max() - pool.loc[chosen_idx, FEATURES].min(),
    })
    comparison["range_kept"] = (comparison["chosen_range"] /
                                comparison["full_range"].replace(0, np.nan))
    print("\ncoverage kept by the subset:")
    print(comparison.round(3).to_string())

    summary = {
        "keep": args.keep, "only": args.only,
        "n_candidates": int(len(pool)),
        "chosen": sorted(chosen),
        "not_chosen": sorted(set(pool["experiment_id"]) - set(chosen)),
        "range_kept": {k: (None if np.isnan(v) else round(float(v), 3))
                       for k, v in comparison["range_kept"].items()},
        "rows_chosen": int(pool.loc[chosen_idx, "rows"].sum()),
        "rows_pool": int(pool["rows"].sum()),
    }
    (out_dir / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pool.assign(chosen=pool["experiment_id"].isin(chosen)).to_csv(
        out_dir / "candidates.csv", index=False)

    print(f"\nkeeping {len(chosen)} of {len(pool)} runs "
          f"({summary['rows_chosen']}/{summary['rows_pool']} rows)")
    print("\n" + ",".join(sorted(chosen)))

    # ------------------------------------------------------------------- figure
    centred = values - values.mean(axis=0)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ components[:2].T
    mask = pool["experiment_id"].isin(chosen).to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    ax.scatter(projected[~mask, 0], projected[~mask, 1], s=48, c=GREY, alpha=0.5,
               edgecolor="white", label=f"dropped ({int((~mask).sum())})")
    ax.scatter(projected[mask, 0], projected[mask, 1], s=70, c=BLUE,
               edgecolor="white", label=f"kept ({int(mask.sum())})")
    ax.set_xlabel("motion-profile component 1")
    ax.set_ylabel("motion-profile component 2")
    ax.set_title("The kept runs still span the driving space")
    ax.legend(fontsize=9)
    fig.savefig(out_dir / "f_selection.png", dpi=170)
    plt.close(fig)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
