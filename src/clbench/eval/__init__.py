"""CLBench evaluation harness.

Tier 1 — intrinsic KG quality (concept fidelity, edge verification,
evidence-pointer faithfulness).  Tier 2 — extrinsic downstream tasks.
Bootstrap CIs and ranking metrics in :mod:`clbench.eval.stats`.
"""

from clbench.eval.eraser import FaithfulnessResult, faithfulness, rationale_mask_from_runs
from clbench.eval.stats import (
    CI,
    auprc,
    auroc,
    bootstrap_ci,
    bootstrap_metric_ci,
    paired_bootstrap_diff,
)
from clbench.eval.tier1_intrinsic import (
    ConceptFidelity,
    EdgeReplication,
    concept_fidelity,
    merge_edge_replication,
    verify_patient_edges,
)

__all__ = [
    "CI",
    "auroc",
    "auprc",
    "bootstrap_ci",
    "bootstrap_metric_ci",
    "paired_bootstrap_diff",
    "FaithfulnessResult",
    "faithfulness",
    "rationale_mask_from_runs",
    "ConceptFidelity",
    "EdgeReplication",
    "concept_fidelity",
    "verify_patient_edges",
    "merge_edge_replication",
]
