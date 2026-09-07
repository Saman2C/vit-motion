"""How much of the saliency is on the ground, and how spread out it is there.

Two separate questions, and the v2 pipeline only ever asked the first one:

**Placement** — what share of the saliency mass falls in a region, and is that
more or less than the region's own area share?  ``region_stats`` returns the
share, the per-frame area, and their ratio (*concentration*): 1.0 means the
model attends to the region exactly in proportion to its size.

**Spread** — *within* the useful region, is the attention distributed over the
terrain or piled onto two or three pixels?  This is the failure mode the first
attention-guided fine-tune produced: sky saliency went 45% -> 0%, which the
penalty rewarded, but what was left collapsed into a couple of hot spots on the
grass.  Nothing in ``task_loss + lambda * (mass outside the ground)`` objects to
that, because the objective only says *don't be outside*; it never says *be
spread out inside*.

``spread_stats`` measures it with the Shannon entropy of the saliency treated as
a distribution over the ground pixels.  The headline number is

    coverage = exp(H) / N_ground

the *effective* fraction of the ground the attention actually covers: 1.0 for a
perfectly even map, k/N for a map concentrated on k pixels.  ``exp(H)`` is the
perplexity of the distribution, i.e. the number of pixels a uniform map would
need to be equally spread — so coverage is readable straight out loud as
"the model is effectively looking at 6% of the ground".

The torch half of this module is the same arithmetic, differentiable, for the
fine-tune's second loss term.  Note the deliberate split there: the *loss* is
``1 - H/log(N)``, whose gradient with respect to H is a constant ``1/log(N)``,
while the *metric* is ``exp(H)/N``.  Coverage is the readable number but a bad
objective — its gradient is proportional to itself, so it vanishes exactly in
the collapsed-to-a-few-hot-spots regime the term exists to escape.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "coverage_from_entropy",
    "ground_focus_report",
    "region_stats",
    "spread_stats",
]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# numpy — analysis and reporting
# --------------------------------------------------------------------------- #
def region_stats(saliency: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Placement of the saliency mass with respect to one region."""
    saliency = np.clip(np.asarray(saliency, dtype=np.float64), 0.0, None)
    mask = np.asarray(mask, dtype=bool)
    total = float(saliency.sum())
    area = float(mask.mean())
    if total <= 0 or area <= 0:
        return {"fraction": float("nan"), "area": area, "concentration": float("nan")}
    fraction = float(saliency[mask].sum()) / total
    return {"fraction": fraction, "area": area, "concentration": fraction / area}


def coverage_from_entropy(entropy: float, n_pixels: int) -> float:
    """``exp(H) / N`` — the effective share of the region the attention covers."""
    if n_pixels <= 0:
        return float("nan")
    return float(np.exp(entropy) / n_pixels)


def spread_stats(saliency: np.ndarray, mask: np.ndarray,
                 top_share: float = 0.10) -> dict[str, float]:
    """How evenly the saliency is distributed *inside* ``mask``.

    Returns
    -------
    entropy            Shannon entropy of the in-region saliency, in nats.
    normalized_entropy ``H / log(N)`` in [0, 1]; 1.0 = perfectly even.
    effective_pixels   ``exp(H)`` — pixels a uniform map would need to match it.
    coverage           ``exp(H) / N`` — the number to quote.
    top_decile_share   share of the in-region mass held by the brightest
                       ``top_share`` of in-region pixels (0.10 -> "the hottest
                       10% of the ground holds this much of the attention").
                       A perfectly even map gives 0.10; a peaky one gives ~1.0.
    """
    saliency = np.clip(np.asarray(saliency, dtype=np.float64), 0.0, None)
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    nan = {"entropy": float("nan"), "normalized_entropy": float("nan"),
           "effective_pixels": float("nan"), "coverage": float("nan"),
           "top_decile_share": float("nan"), "n_pixels": float(n)}
    if n < 2:
        return nan
    values = saliency[mask]
    total = float(values.sum())
    if total <= 0:
        return nan
    p = values / total
    entropy = float(-(p * np.log(p + _EPS)).sum())
    k = max(1, int(round(top_share * n)))
    top = float(np.sort(p)[::-1][:k].sum())
    return {
        "entropy": entropy,
        "normalized_entropy": entropy / float(np.log(n)),
        "effective_pixels": float(np.exp(entropy)),
        "coverage": coverage_from_entropy(entropy, n),
        "top_decile_share": top,
        "n_pixels": float(n),
    }


def ground_focus_report(saliency: np.ndarray, sky: np.ndarray, ground: np.ndarray,
                        other: np.ndarray) -> dict[str, float]:
    """Everything worth reporting for one frame, flattened for a DataFrame row."""
    out: dict[str, float] = {}
    for name, mask in (("sky", sky), ("ground", ground), ("other", other)):
        for key, value in region_stats(saliency, mask).items():
            out[f"{name}_{key}"] = value
    distractor = np.asarray(sky, dtype=bool) | np.asarray(other, dtype=bool)
    for key, value in region_stats(saliency, distractor).items():
        out[f"distractor_{key}"] = value
    for key, value in spread_stats(saliency, ground).items():
        out[f"ground_{key}"] = value
    return out


# --------------------------------------------------------------------------- #
# torch — the differentiable loss terms
# --------------------------------------------------------------------------- #
def masked_fraction(saliency, mask, eps: float = 1e-8):
    """``[B]`` share of each sample's saliency mass that falls inside ``mask``.

    ``saliency`` and ``mask`` are ``[B, 1, H, W]``.  This is the RRR / RobustViT
    term: drive it to zero for the distractor region.
    """
    region = (saliency * mask).sum(dim=(1, 2, 3))
    total = saliency.sum(dim=(1, 2, 3)) + eps
    return region / total


def normalized_entropy(saliency, mask, eps: float = 1e-8):
    """``[B]`` entropy of the in-mask saliency, divided by ``log(N_mask)``.

    1.0 when the attention is perfectly even over the region, ~0 when it sits on
    a single pixel.  Samples whose mask holds fewer than two pixels return 0 and
    should be excluded by the caller's validity mask.
    """
    import torch

    weights = (saliency.clamp_min(0.0) * mask).flatten(1)          # [B, H*W]
    counts = mask.flatten(1).sum(dim=1)                            # [B]
    totals = weights.sum(dim=1, keepdim=True) + eps
    p = weights / totals
    entropy = -(p * torch.log(p + eps)).sum(dim=1)                 # [B]
    ceiling = torch.log(counts.clamp_min(2.0))
    return (entropy / ceiling).clamp(0.0, 1.0)


def spread_loss(saliency, mask, eps: float = 1e-8):
    """``1 - normalized_entropy`` — minimize to spread attention over ``mask``.

    Bounded in [0, 1], so it composes with the placement penalty on a comparable
    scale and one lambda per term is enough.
    """
    return 1.0 - normalized_entropy(saliency, mask, eps=eps)
