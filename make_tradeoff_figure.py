#!/usr/bin/env python3
"""The one figure the argument rests on: focus versus spread, and accuracy flat.

Every arm in this study lowers ``distractor_fraction``.  That is not the finding —
a penalty on a region always lowers attention in that region.  The finding is what
the penalty COSTS: three of the four arms buy their focus by collapsing the
attention onto a handful of ground pixels, and only the arm with the spread term
buys it for nothing.  A table hides that because the two numbers sit in different
columns; on one pair of axes it is a single picture.

    python make_tradeoff_figure.py --summary artifacts/v03_summary.csv \
        --output artifacts/f_tradeoff.png

Left panel  x = distractor_fraction (log; it spans 40x), y = ground_coverage.
            Up and to the LEFT is better, and the shaded corner says so.
Right panel yaw R2 per arm, so "at no accuracy cost" is shown rather than asserted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Three roles, not six: identity is carried by the direct labels, colour carries
# the ARGUMENT.  (Validated all-pairs on a light surface: worst CVD dE 9.2.)
GUIDED_C, REFERENCE_C, PARTIAL_C = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1c1b19", "#6b7280", "#e3e2dd"

ROLE = {
    "v03_base":   ("no guidance", REFERENCE_C),
    "control":    ("no guidance", REFERENCE_C),
    "v1_tophalf": ("guidance, no spread term", PARTIAL_C),
    "sky_only":   ("guidance, no spread term", PARTIAL_C),
    "nonground":  ("guidance, no spread term", PARTIAL_C),
    "guided":     ("guidance + spread term", GUIDED_C),
}
# Where to put each label so it does not sit on its own dot.
OFFSET = {
    "v03_base": (18, 14), "control": (16, -20), "v1_tophalf": (2, -22),
    "sky_only": (0, 17), "nonground": (0, 17), "guided": (0, 17),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="artifacts/v03_summary.csv")
    ap.add_argument("--output", default="artifacts/f_tradeoff.png")
    ap.add_argument("--accuracy-column", default="yaw_R2")
    ap.add_argument("--dpi", type=int, default=190)
    args = ap.parse_args()

    frame = pd.read_csv(args.summary)
    need = {"run", "distractor_fraction", "ground_coverage", args.accuracy_column}
    missing = need - set(frame.columns)
    if missing:
        raise SystemExit(f"{args.summary} is missing {sorted(missing)}")
    frame = frame.dropna(subset=["distractor_fraction", "ground_coverage"])
    order = [r for r in ROLE if r in set(frame["run"])] + \
            [r for r in frame["run"] if r not in ROLE]
    frame = frame.set_index("run").loc[order].reset_index()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.4, 5.6),
                                 gridspec_kw={"width_ratios": [1.55, 1]},
                                 constrained_layout=True)

    # ---------------------------------------------------------------- left
    # The x axis is inverted below, so "fewer distractors" is to the RIGHT and the
    # corner worth being in is top-right.  xmin/xmax are axes fractions, i.e. they
    # are read in screen order and do not flip with the axis.
    base = frame[frame["run"] == "v03_base"]
    floor = float(base["ground_coverage"].iloc[0]) * 0.9 if len(base) else 0.8
    ax.axhspan(floor, 1.06, xmin=0.52, xmax=1.0, color=GUIDED_C, alpha=0.06, zorder=0)

    seen = set()
    for _, row in frame.iterrows():
        label, colour = ROLE.get(row["run"], ("other", MUTED))
        ax.scatter(row["distractor_fraction"], row["ground_coverage"],
                   s=190, color=colour, edgecolor="white", linewidth=2.0,
                   zorder=3, label=label if label not in seen else None)
        seen.add(label)
        dx, dy = OFFSET.get(row["run"], (0, 14))
        ax.annotate(row["run"], (row["distractor_fraction"], row["ground_coverage"]),
                    textcoords="offset points", xytext=(dx, dy), ha="center",
                    fontsize=10.5, color=INK, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("attention on sky + distractor objects   (lower is better →)")
    ax.set_ylabel("spread of attention over the ground   (higher is better ↑)")
    ax.set_ylim(0.10, 1.06)
    ax.invert_xaxis()
    lo = float(frame["distractor_fraction"].min()) / 1.9
    hi = float(frame["distractor_fraction"].max()) * 1.5
    ax.set_xlim(hi, lo)          # inverted: large distractor share on the left

    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=9.5, loc="lower left", frameon=False)
    ax.text(0.985, 0.965, "the corner worth being in", transform=ax.transAxes,
            fontsize=9.5, color=GUIDED_C, style="italic", ha="right", va="top")
    ax.set_title("Every penalty lowers the distractor share.\n"
                 "Only one keeps the attention spread out.",
                 fontsize=12, loc="left")

    # --------------------------------------------------------------- right
    values = frame[args.accuracy_column].astype(float)
    positions = range(len(frame))
    bx.hlines(list(positions), values.min() - 0.004, values,
              color=GRID, lw=1.5, zorder=1)
    for k, (_, row) in enumerate(frame.iterrows()):
        _, colour = ROLE.get(row["run"], ("other", MUTED))
        bx.scatter(row[args.accuracy_column], k, s=150, color=colour,
                   edgecolor="white", linewidth=2.0, zorder=3)
        bx.text(row[args.accuracy_column] + 0.0012, k, f"{row[args.accuracy_column]:.3f}",
                va="center", fontsize=9.5, color=INK)
    bx.set_yticks(list(positions))
    bx.set_yticklabels(frame["run"], fontsize=10.5)
    bx.invert_yaxis()
    span = values.max() - values.min()
    bx.set_xlim(values.min() - 0.006, values.max() + 0.008)
    bx.set_xlabel("held-out yaw R²")
    bx.grid(True, axis="x", color=GRID, lw=0.8)
    bx.set_axisbelow(True)
    for side in ("top", "right", "left"):
        bx.spines[side].set_visible(False)
    bx.tick_params(axis="y", length=0)
    bx.set_title(f"Accuracy does not move — whole spread {span:.3f} R²\n"
                 "(control = same extra epochs, no penalty)",
                 fontsize=12, loc="left")

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
