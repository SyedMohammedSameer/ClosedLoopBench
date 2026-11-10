"""
ClosedLoopBench data loader: VitalDB processed numerics + action events -> ``CaseData``.

Expected layout (produced by the CLBench VitalDB pipeline; see ``data/processed_vitaldb``):

    <root>/numerics/<case>.parquet   30 s windows at a 15 s stride: window_start_sec, window_end_sec,
                                     map, hr, etco2 and the action tracks (TRACK_COLUMNS)
    <root>/fine_2s/<case>.parquet    the same at 2 s (500 cases)
    <root>/events/<case>.parquet     patient_id, t_sec, agent, value_before, value_after

A case's action stream is the list of set-point / rate changes of each agent (``ACTIONS``). A positive
dose after the last recorded value of the action's track is unknown and masked (NaN) (``t_stop``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from clbench.eval import closed_loop_bench as clb

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "processed_vitaldb"


def scale_dir(scale: str, root: Path = DEFAULT_ROOT) -> Path:
    return Path(root) / ("numerics" if scale == "30s" else "fine_2s")


def list_case_files(scale: str, cases: str = "all", limit: int = 0, root: Path = DEFAULT_ROOT) -> list[Path]:
    """Sorted case files. ``cases``: "all" (every case with numerics at this scale) or "fine" (the cases that
    also have 2 s numerics)."""
    files = sorted(scale_dir(scale, root).glob("*.parquet"))
    if cases == "fine":
        fine = {f.stem for f in (Path(root) / "fine_2s").glob("*.parquet")}
        files = [f for f in files if f.stem in fine]
    return files[:limit] if limit else files


def load_events(pids, root: Path = DEFAULT_ROOT) -> dict:
    """patient id -> events DataFrame restricted to the benchmark's agents."""
    import pandas as pd

    agents = {a.agent for a in clb.ACTIONS.values()}
    frames = []
    for pid in sorted(pids):
        f = Path(root) / "events" / f"{pid}.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["patient_id", "t_sec", "agent", "value_before", "value_after"])
            frames.append(d[d.agent.isin(agents)])
    E = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {pid: g for pid, g in E.groupby("patient_id")} if len(E) else {}


def ev_records(df) -> dict[str, tuple]:
    """agent -> (t, before, after) arrays (picklable, compact)."""
    if df is None or len(df) == 0:
        return {}
    return {a: (g.t_sec.to_numpy(), g.value_before.to_numpy(), g.value_after.to_numpy()) for a, g in df.groupby("agent")}


def to_events(rec: dict[str, tuple], stops: dict[str, float] | None = None) -> dict[str, clb.EventSet]:
    """``stops``: per agent, the time the recorded track stops (dose masking); None = no masking."""
    out = {}
    for agent, (t, b, a) in rec.items():
        key = clb.AGENT_TO_ACTION[agent]
        ts = (stops or {}).get(agent, np.inf)
        out[key] = clb.EventSet.build(t, b, a, clb.ACTIONS[key], ts)
    return out


def track_stops(path) -> tuple[str, dict[str, float]]:
    """(case id, agent -> time the recorded track stops) from one numerics file."""
    import pandas as pd

    cols = ["window_end_sec", *clb.TRACK_COLUMNS.values()]
    d = pd.read_parquet(path, columns=cols)
    st = clb.track_stop_times(d)
    return Path(path).stem, {clb.ACTIONS[a].agent: v for a, v in st.items()}


def load_case(path, scale: str, events: dict[str, clb.EventSet]) -> clb.CaseData:
    """One case's vitals at ``scale`` with the given action streams."""
    import pandas as pd

    cfg = clb.SCALES[scale]
    d = pd.read_parquet(path, columns=["window_start_sec", *clb.VITALS])
    return clb.CaseData(Path(path).stem, d.window_start_sec.to_numpy(float) + cfg.window_sec / 2, cfg.window_sec,
                        {v: d[v].to_numpy(float) for v in clb.VITALS}, events)


def load_cases(scale: str = "30s", cases: str = "all", limit: int = 0, root: Path = DEFAULT_ROOT,
               mask_stopped_tracks: bool = True) -> list[clb.CaseData]:
    """All requested cases as ``CaseData`` (serial; for a few hundred cases or interactive use)."""
    files = list_case_files(scale, cases, limit, root)
    ev = load_events({f.stem for f in files}, root)
    out = []
    for f in files:
        stops = track_stops(f)[1] if mask_stopped_tracks else None  # as the runner: the scale's own file
        out.append(load_case(f, scale, to_events(ev_records(ev.get(f.stem)), stops)))
    return out
