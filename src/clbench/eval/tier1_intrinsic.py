"""
Tier 1 — Intrinsic KG quality metrics.

Implements the Tier-1 evaluation defined in the V1 plan, every metric tied
to prior work:

1. **Concept fidelity** (Koh et al. 2020, *Concept Bottleneck Models*):
   per-concept precision / recall / F1 / accuracy / AUROC / AUPRC of the CBM
   activations against the rule-derived ground-truth labels.

2. **Edge verification** — held-out replication of each edge type.  The KG
   edges were built on each patient's full window series; to verify they are
   not artefacts we recompute the underlying relation on the first vs second
   half of the series and report how often an edge that fires on the first
   half also fires on the second half.

   * Temporal (precedes/follows): cross-correlation (standard signal proc.)
   * Co-evidence (co_occurs): Jaccard overlap (DeYoung et al. 2020)
   * Granger (granger_precedes): F-test re-run on held-out windows
     (Seth 2010; Barnett & Seth 2014)

Evidence-pointer faithfulness lives in :mod:`clbench.eval.eraser`;
bootstrap CIs live in :mod:`clbench.eval.stats`.  This module is
I/O-free and operates on numpy arrays so it is unit-testable on synthetic
activations; the orchestration script handles loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from clbench.edges import (
    co_evidence_relation,
    granger_precedes_score,
    is_co_evidence_edge,
    is_directed_temporal_edge,
    is_granger_edge,
    temporal_relation,
)
from clbench.eval.stats import auprc, auroc
from clbench.persistence import WINDOW_STRIDE_SEC_DEFAULT

# ---------------------------------------------------------------------------
# 1. Concept fidelity
# ---------------------------------------------------------------------------


@dataclass
class ConceptFidelity:
    """Per-concept fidelity of CBM activations vs rule labels."""

    concept: str
    modality: str
    n: int
    n_positive: int
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    auroc: float
    auprc: float


def concept_fidelity(
    concept: str,
    modality: str,
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> ConceptFidelity:
    """
    Koh-2020-style concept fidelity: how well the CBM probability reproduces
    the rule-derived label for one concept, pooled across windows.

    ``threshold`` is the concept's calibrated activation threshold (the same
    one used at KG-build time), so precision/recall reflect the operating
    point the graph actually uses.
    """
    probs = np.nan_to_num(np.asarray(probs, dtype=np.float64))
    labels = np.asarray(labels).astype(np.int64)
    n = len(labels)
    pred = (probs >= threshold).astype(np.int64)

    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and np.isfinite(precision) and np.isfinite(recall)
        else 0.0
    )
    accuracy = (tp + tn) / n if n else float("nan")

    return ConceptFidelity(
        concept=concept,
        modality=modality,
        n=n,
        n_positive=int(labels.sum()),
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=float(accuracy),
        auroc=auroc(labels, probs),
        auprc=auprc(labels, probs),
    )


# ---------------------------------------------------------------------------
# 2. Edge verification (held-out replication on series halves)
# ---------------------------------------------------------------------------


@dataclass
class EdgeReplication:
    """Held-out replication counts for one edge type, accumulated over pairs."""

    edge_type: str
    n_pairs: int = 0  # concept pairs tested
    n_fired_h1: int = 0  # edges present on first half
    n_replicated: int = 0  # of those, also present on second half
    weights_h1: list[float] = field(default_factory=list)  # full-series-style weight on h1
    weights_h2: list[float] = field(default_factory=list)

    @property
    def replication_rate(self) -> float:
        return self.n_replicated / self.n_fired_h1 if self.n_fired_h1 else float("nan")


def _halves(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = len(arr) // 2
    return arr[:mid], arr[mid:]


def verify_patient_edges(
    probs_by_key: dict[str, np.ndarray],
    thresholds_by_key: dict[str, float],
    enable_granger: bool = True,
    window_stride_sec: float = WINDOW_STRIDE_SEC_DEFAULT,
    granger_min_samples: int = 30,
) -> dict[str, EdgeReplication]:
    """
    For one patient, test every unordered concept pair on the first vs second
    half of its activation series and tally edge replication per type.

    Returns ``{edge_type: EdgeReplication}`` for "co_occurs", "temporal",
    and (optionally) "granger".  Granger is directional so ordered pairs are
    tested both ways.
    """
    out = {
        "co_occurs": EdgeReplication("co_occurs"),
        "temporal": EdgeReplication("temporal"),
    }
    if enable_granger:
        out["granger"] = EdgeReplication("granger")

    keys = list(probs_by_key.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ka, kb = keys[i], keys[j]
            a, b = probs_by_key[ka], probs_by_key[kb]
            if len(a) != len(b) or len(a) < 4:
                continue
            ta, tb = thresholds_by_key[ka], thresholds_by_key[kb]
            a1, a2 = _halves(a)
            b1, b2 = _halves(b)
            if len(a1) < 2 or len(a2) < 2:
                continue

            # --- co-evidence ---
            r = out["co_occurs"]
            c1 = co_evidence_relation(a1, b1, ta, tb)
            c2 = co_evidence_relation(a2, b2, ta, tb)
            r.n_pairs += 1
            r.weights_h1.append(c1.jaccard)
            r.weights_h2.append(c2.jaccard)
            if is_co_evidence_edge(c1):
                r.n_fired_h1 += 1
                if is_co_evidence_edge(c2):
                    r.n_replicated += 1

            # --- temporal (direction must also agree) ---
            r = out["temporal"]
            t1 = temporal_relation(a1, b1, ta, tb, window_stride_sec=window_stride_sec)
            t2 = temporal_relation(a2, b2, ta, tb, window_stride_sec=window_stride_sec)
            r.n_pairs += 1
            r.weights_h1.append(abs(t1.peak_xcorr))
            r.weights_h2.append(abs(t2.peak_xcorr))
            if is_directed_temporal_edge(t1):
                r.n_fired_h1 += 1
                if is_directed_temporal_edge(t2) and (t1.a_precedes_b == t2.a_precedes_b):
                    r.n_replicated += 1

            # --- granger (both directions) ---
            if enable_granger:
                r = out["granger"]
                for x1, y1, x2, y2 in (
                    (a1, b1, a2, b2),
                    (b1, a1, b2, a2),
                ):
                    g1 = granger_precedes_score(x1, y1, min_samples=granger_min_samples)
                    g2 = granger_precedes_score(x2, y2, min_samples=granger_min_samples)
                    r.n_pairs += 1
                    r.weights_h1.append(1.0 - g1.min_p_value)
                    r.weights_h2.append(1.0 - g2.min_p_value)
                    if is_granger_edge(g1):
                        r.n_fired_h1 += 1
                        if is_granger_edge(g2):
                            r.n_replicated += 1
    return out


def merge_edge_replication(
    per_patient: list[dict[str, EdgeReplication]],
) -> dict[str, EdgeReplication]:
    """Accumulate per-patient EdgeReplication dicts into a cohort total."""
    total: dict[str, EdgeReplication] = {}
    for d in per_patient:
        for et, rep in d.items():
            agg = total.setdefault(et, EdgeReplication(et))
            agg.n_pairs += rep.n_pairs
            agg.n_fired_h1 += rep.n_fired_h1
            agg.n_replicated += rep.n_replicated
            agg.weights_h1.extend(rep.weights_h1)
            agg.weights_h2.extend(rep.weights_h2)
    return total
