"""
ERASER-style evidence-pointer faithfulness (DeYoung et al. 2020).

CLBench's locked design makes evidence traceability first-class: every
KG node carries :class:`EvidencePointer`s back to the source windows that
activated the concept.  A faithful evidence set is one that actually
*drives* the model's belief that the concept is present — not a post-hoc
decoration.

We adapt the two ERASER faithfulness metrics to this setting.  Treat the
patient-level "concept presence" as the model prediction:

    presence(probs) = mean of the top-k window activations
                      (k = number of evidence windows for the concept)

and the rationale as the set of windows flagged as evidence (the windows
inside the concept's above-threshold activation runs).

* **Comprehensiveness** = presence(all) - presence(all \\ rationale).
  High ⇒ removing the evidence windows collapses the concept signal, i.e.
  the rationale is necessary.

* **Sufficiency** = presence(all) - presence(rationale only).
  Low (near 0) ⇒ the rationale alone reproduces the concept signal, i.e.
  the rationale is sufficient.

* **Insertion / deletion AUC** (Petsiuk et al. 2018, used in ERASER):
  rank windows by activation and build a presence curve as windows are
  progressively inserted (from empty) or deleted (from full).  A faithful
  importance ordering gives a high insertion AUC and a low deletion AUC.
  We report ``deletion_auc`` and ``insertion_auc`` plus their gap.

All functions are pure-numpy and operate on a single concept's
per-window probability sequence plus a boolean rationale mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _trapz(y: list[float] | np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of ``y`` over ``x``.

    Implemented directly to be independent of the numpy 2.0 ``trapz`` ->
    ``trapezoid`` rename.
    """
    ya = np.asarray(y, dtype=np.float64)
    xa = np.asarray(x, dtype=np.float64)
    return float(np.sum((xa[1:] - xa[:-1]) * (ya[1:] + ya[:-1]) / 2.0))


@dataclass(frozen=True)
class FaithfulnessResult:
    """Per-(patient, concept) evidence-faithfulness scores."""

    comprehensiveness: float
    sufficiency: float
    deletion_auc: float
    insertion_auc: float
    n_evidence: int
    n_windows: int

    @property
    def insertion_minus_deletion(self) -> float:
        return self.insertion_auc - self.deletion_auc


def _presence_topk(probs: np.ndarray, k: int) -> float:
    """Mean of the k highest activations (0.0 if k == 0 or empty)."""
    if k <= 0 or probs.size == 0:
        return 0.0
    k = min(k, probs.size)
    return float(np.sort(probs)[::-1][:k].mean())


def faithfulness(
    probs: np.ndarray,
    rationale_mask: np.ndarray,
    n_steps: int = 20,
) -> FaithfulnessResult:
    """
    Compute comprehensiveness, sufficiency, and insertion/deletion AUC for one
    concept's activation sequence given a boolean ``rationale_mask`` over the
    same windows.

    Parameters
    ----------
    probs : (N,) float ndarray of per-window CBM activations in [0, 1].
    rationale_mask : (N,) bool ndarray; True where the window is evidence.
    n_steps : number of points on the insertion/deletion curves.
    """
    probs = np.nan_to_num(np.asarray(probs, dtype=np.float64))
    mask = np.asarray(rationale_mask, dtype=bool)
    n = probs.size
    k = int(mask.sum())

    if n == 0:
        return FaithfulnessResult(0.0, 0.0, 0.0, 0.0, 0, 0)

    presence_full = _presence_topk(probs, k if k > 0 else 1)

    # Comprehensiveness: drop the rationale windows, recompute on the rest.
    rest = probs[~mask]
    presence_without = _presence_topk(rest, k if k > 0 else 1)
    comprehensiveness = float(presence_full - presence_without)

    # Sufficiency: keep only the rationale windows.
    only = probs[mask]
    presence_only = _presence_topk(only, k if k > 0 else 1)
    sufficiency = float(presence_full - presence_only)

    # Insertion / deletion curves over windows ranked by activation desc.
    order = np.argsort(probs)[::-1]
    ranked = probs[order]
    fractions = np.linspace(0.0, 1.0, n_steps + 1)
    ins, dele = [], []
    for f in fractions:
        m = int(round(f * n))
        # Insertion: presence using the top-m most-active windows.
        ins.append(float(ranked[:m].mean()) if m > 0 else 0.0)
        # Deletion: presence of what remains after removing top-m.
        dele.append(float(ranked[m:].mean()) if m < n else 0.0)
    insertion_auc = float(_trapz(ins, fractions))
    deletion_auc = float(_trapz(dele, fractions))

    return FaithfulnessResult(
        comprehensiveness=comprehensiveness,
        sufficiency=sufficiency,
        deletion_auc=deletion_auc,
        insertion_auc=insertion_auc,
        n_evidence=k,
        n_windows=n,
    )


def rationale_mask_from_runs(
    probs: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Reconstruct the evidence (rationale) mask the KG builder would use: the
    windows whose activation is at/above the concept's activation threshold
    (i.e. the windows inside the above-threshold runs that mint nodes).
    """
    probs = np.nan_to_num(np.asarray(probs, dtype=np.float64), nan=-np.inf)
    return probs >= threshold
