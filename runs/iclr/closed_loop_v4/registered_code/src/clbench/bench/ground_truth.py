"""
ClosedLoopBench ground truth: 18 relations with a sign known from physiology, pharmacology or clinical
practice, their primary horizons, and the sources that justify each sign.

The table is the one scored by :mod:`clbench.eval.closed_loop_bench` (``RELATIONS``); this module adds
the citations as structured data so the table can be released and cited on its own.

Conventions. Every relation has a *cause* and an *effect*. Response (action -> vital): cause = the
action's dose or set-point, effect = the vital. Policy (vital -> action): cause = the vital's 2-min trend
before the decision, effect = the clinician's titration. ``expected`` is the sign of d(effect)/d(cause);
estimators report in this coefficient space, so a correct estimate always has sign ``expected``.
``event_dir`` says whether the relation is about increases (+1) or decreases (-1) of the action (the
drug titration paradox makes up and down steps behave differently in closed-loop data).
"""

from __future__ import annotations

from clbench.eval import closed_loop_bench as _clb
from clbench.eval.closed_loop_bench import ACTIONS, RELATIONS, VITALS, Relation  # v3 table (import-time binding)

SOURCES: dict[str, str] = {
    "west": "West JB. Respiratory Physiology: The Essentials (alveolar ventilation equation, PaCO2 ~ VCO2 / VA).",
    "nunn": "Lumb AB. Nunn's Applied Respiratory Physiology (carbon dioxide stores and elimination).",
    "taskar1995": "Taskar V et al. Dynamics of carbon dioxide elimination following ventilator resetting. Chest 1995.",
    "schnider1998": "Schnider TW et al. The influence of method of administration and covariates on the pharmacokinetics "
                    "of propofol in adult volunteers. Anesthesiology 1998; and the pharmacodynamic companion, "
                    "Anesthesiology 1999 (ke0 ~0.46/min).",
    "kazama1999": "Kazama T et al. Comparison of the effect-site k(eO)s of propofol for blood pressure and EEG bispectral "
                  "index in elderly and younger patients. Anesthesiology 1999.",
    "glass1999": "Glass PS, Gan TJ, Howell S. A review of the pharmacokinetics and pharmacodynamics of remifentanil. "
                 "Anesth Analg 1999.",
    "minto1997": "Minto CF et al. Influence of age and gender on the pharmacokinetics and pharmacodynamics of "
                 "remifentanil. Anesthesiology 1997.",
    "smyth1969": "Smyth HS, Sleight P, Pickering GW. Reflex regulation of arterial pressure during sleep in man: a "
                 "quantitative method of assessing baroreflex sensitivity. Circ Res 1969.",
    "goodman_gilman": "Goodman & Gilman's The Pharmacological Basis of Therapeutics (adrenergic agonists; organic nitrates).",
    "guignard2006": "Guignard B. Monitoring analgesia. Best Pract Res Clin Anaesthesiol 2006.",
    "sessler2019": "Sessler DI et al. Perioperative Quality Initiative consensus statement on intraoperative blood "
                   "pressure, risk and outcomes for elective surgery. Br J Anaesth 2019.",
    "walsh2013": "Walsh M et al. Relationship between intraoperative mean arterial pressure and clinical outcomes after "
                 "noncardiac surgery. Anesthesiology 2013.",
    "miller": "Miller's Anesthesia (intraoperative mechanical ventilation; normocapnia target).",
    # v4 additions
    "cournand1948": "Cournand A, Motley HL, Werko L, Richards DW. Physiological studies of the effects of intermittent "
                    "positive pressure breathing on cardiac output in man. Am J Physiol 1948.",
    "luecke2005": "Luecke T, Pelosi P. Clinical review: positive end-expiratory pressure and cardiac output. Crit Care 2005.",
    "ebert1995": "Ebert TJ, Harkin CP, Muzi M. Cardiovascular responses to sevoflurane: a review. Anesth Analg 1995.",
    "katoh1998": "Katoh T, Suzuki A, Ikeda K. Electroencephalographic derivatives as a tool for predicting the depth of "
                 "sedation and anesthesia induced by sevoflurane. Anesthesiology 1998.",
    "schnider1999": "Schnider TW et al. The influence of age on propofol pharmacodynamics. Anesthesiology 1999 "
                    "(effect-site model on the EEG).",
    "punjasawadwong2014": "Punjasawadwong Y, Phongchiewboon A, Bunchungmongkol N. Bispectral index for improving "
                          "anaesthetic delivery and postoperative recovery. Cochrane Database Syst Rev 2014.",
}

# relation id -> (basis, source keys)
BASIS: dict[str, tuple[str, tuple[str, ...]]] = {
    "R01": ("physiology: higher set rate at fixed tidal volume raises alveolar ventilation and lowers EtCO2",
            ("west", "nunn", "taskar1995")),
    "R02": ("physiology: lower set rate lowers alveolar ventilation and raises EtCO2", ("west", "nunn", "taskar1995")),
    "R03": ("pharmacology: propofol lowers SVR and sympathetic tone (dose-dependent fall in MAP)",
            ("schnider1998", "kazama1999")),
    "R04": ("pharmacology: lowering the propofol target reverses the fall in MAP", ("schnider1998", "kazama1999")),
    "R05": ("pharmacology: mu-opioid, vagally mediated bradycardia", ("glass1999", "minto1997")),
    "R06": ("pharmacology: mu-opioid, dose-dependent hypotension", ("glass1999", "minto1997")),
    "R07": ("pharmacology: lowering the remifentanil target reverses the bradycardia", ("glass1999", "minto1997")),
    "R08": ("pharmacology: lowering the remifentanil target reverses the hypotension", ("glass1999", "minto1997")),
    "R09": ("pharmacology: alpha-1 agonist raises SVR and MAP", ("smyth1969", "goodman_gilman")),
    "R10": ("physiology: baroreflex slows HR when phenylephrine raises MAP", ("smyth1969",)),
    "R11": ("pharmacology: alpha-1 (+ beta-1) agonist raises MAP", ("goodman_gilman",)),
    "R12": ("pharmacology: venodilator lowers preload and MAP", ("goodman_gilman",)),
    "P01": ("practice: rising MAP read as nociception, analgesia deepened", ("guignard2006",)),
    "P02": ("practice: rising HR read as nociception, analgesia deepened", ("guignard2006",)),
    "P03": ("practice: falling MAP treated with a vasopressor", ("sessler2019", "walsh2013")),
    "P04": ("practice: falling MAP treated by lightening the hypnotic", ("sessler2019", "walsh2013")),
    "P05": ("practice: rising EtCO2 corrected by raising the set rate (normocapnia)", ("miller",)),
    "P06": ("practice: falling MAP treated with norepinephrine", ("sessler2019", "walsh2013")),
    # v4 (ClosedLoopBench 1.1)
    "R13": ("physics: larger set tidal volume at fixed rate raises alveolar ventilation and lowers EtCO2",
            ("west", "nunn", "taskar1995")),
    "R14": ("physics: smaller set tidal volume lowers alveolar ventilation and raises EtCO2", ("west", "nunn", "taskar1995")),
    "R15": ("physics: higher FiO2 raises alveolar PO2 and SpO2 below the plateau of the dissociation curve "
            "(events with pre-level SpO2 < 97 %)", ("west", "nunn")),
    "R16": ("physiology: PEEP raises intrathoracic pressure, lowers venous return, cardiac output and MAP (modest)",
            ("cournand1948", "luecke2005")),
    "R17": ("pharmacology: volatile agents lower SVR and MAP dose-dependently (MAC steps derived from measured MAC)",
            ("ebert1995",)),
    "R18": ("pharmacology: volatile agents deepen the EEG hypnotic effect, BIS falls (derived MAC steps)", ("katoh1998",)),
    "R19": ("pharmacology: propofol effect-site concentration lowers BIS", ("schnider1999",)),
    "P07": ("practice: falling SpO2 treated by raising FiO2", ("miller",)),
    "P08": ("practice: rising EtCO2 corrected by raising minute ventilation (set tidal volume)", ("miller",)),
    "P09": ("practice: rising BIS (lightening) treated by deepening propofol", ("punjasawadwong2014",)),
    "P10": ("practice: rising BIS treated by deepening the volatile agent (derived MAC steps)", ("punjasawadwong2014",)),
}

GROUND_TRUTH_VERSION = "ClosedLoopBench-1.0"  # v3 table; the v4 table (use_table("v4")) is ClosedLoopBench-1.1


def ground_truth_table():
    """pandas DataFrame, one row per relation: id, type, cause, effect, action, vital, event direction,
    expected sign, primary horizon (s), latency, label, basis, sources (keys and full references)."""
    import pandas as pd

    rows = []
    for r in _clb.RELATIONS:  # the active table (v3 by default; v4 after ``use_table("v4")``)
        basis, keys = BASIS[r.rid]
        rows.append(dict(rid=r.rid, kind=r.kind, cause=r.cause, effect=r.effect, action=r.action, vital=r.vital,
                         event_dir=r.event_dir, expected=r.expected, horizon_s=r.horizon_s,
                         latency=r.latency, label=r.label, basis=basis, source_keys=";".join(keys),
                         sources=" | ".join(SOURCES[k] for k in keys), domain=_clb.relation_domain(r),
                         pre_max=r.pre_max))
    return pd.DataFrame(rows)


__all__ = ["ACTIONS", "BASIS", "GROUND_TRUTH_VERSION", "RELATIONS", "Relation", "SOURCES", "VITALS",
           "ground_truth_table"]
