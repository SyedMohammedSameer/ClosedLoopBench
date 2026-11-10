"""
Co-evidence edge typing: ``co_occurs``.

When two concepts' active-window sets overlap above a Jaccard threshold,
we emit a symmetric ``CO_OCCURS`` edge between them.  The edge type
captures "these concepts tend to be active in the same windows" without
making a directional or causal claim — that's what PRECEDES / FOLLOWS
and GRANGER_PRECEDES are for.

The Jaccard formulation is intentionally classic (DeYoung et al. 2020
use it for ERASER evidence comparison; widely accepted in info-retrieval
and biomedical ontology overlap work).  This keeps the methodology
defensible without inventing a new metric — which matches the locked
constraint that every eval metric must come from previously published work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# Below this Jaccard, the overlap isn't strong enough to claim co-evidence.
DEFAULT_MIN_JACCARD: float = 0.10


@dataclass
class CoEvidenceRelation:
    """Result of one (A, B) co-evidence test."""

    jaccard: float
    n_intersection: int
    n_union: int
    n_a_only: int
    n_b_only: int


def _binary(probs: np.ndarray, threshold: float) -> np.ndarray:
    return np.nan_to_num(np.asarray(probs, dtype=np.float64), nan=-np.inf) >= threshold


def co_evidence_relation(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
) -> CoEvidenceRelation:
    """
    Jaccard overlap between two activation sequences treated as window sets.

    Both sequences must share a length (i.e. live on the same window axis).
    """
    a = _binary(probs_a, threshold_a)
    b = _binary(probs_b, threshold_b)
    if len(a) != len(b):
        raise ValueError(f"probs length mismatch: a={len(a)} b={len(b)}")

    inter = int((a & b).sum())
    union = int((a | b).sum())
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    jaccard = inter / union if union > 0 else 0.0

    return CoEvidenceRelation(
        jaccard=float(jaccard),
        n_intersection=inter,
        n_union=union,
        n_a_only=a_only,
        n_b_only=b_only,
    )


def is_co_evidence_edge(
    rel: CoEvidenceRelation,
    min_jaccard: float = DEFAULT_MIN_JACCARD,
) -> bool:
    """True when the Jaccard overlap clears ``min_jaccard``."""
    return rel.jaccard >= min_jaccard
