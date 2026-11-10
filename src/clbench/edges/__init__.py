"""KG edge typing modules.

Three edge types, each derived from a measurable signal property:

* :mod:`clbench.edges.temporal`     — ``PRECEDES`` / ``FOLLOWS``
* :mod:`clbench.edges.co_evidence`  — ``CO_OCCURS``
* :mod:`clbench.edges.granger`      — ``GRANGER_PRECEDES``
"""

from clbench.edges.co_evidence import (
    CoEvidenceRelation,
    co_evidence_relation,
    is_co_evidence_edge,
)
from clbench.edges.granger import (
    GrangerRelation,
    granger_edge_weight,
    granger_precedes_score,
    is_granger_edge,
)
from clbench.edges.temporal import (
    TemporalRelation,
    is_directed_temporal_edge,
    temporal_relation,
)

__all__ = [
    "TemporalRelation",
    "temporal_relation",
    "is_directed_temporal_edge",
    "CoEvidenceRelation",
    "co_evidence_relation",
    "is_co_evidence_edge",
    "GrangerRelation",
    "granger_precedes_score",
    "granger_edge_weight",
    "is_granger_edge",
]
