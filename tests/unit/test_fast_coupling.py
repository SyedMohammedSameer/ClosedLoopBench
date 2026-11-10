"""Unit tests for beat extraction (clbench.beats) and fast coupling edges (clbench.edges.fast_coupling).

All signals are synthetic with known ground truth: a beat-by-beat closed loop with a planted
baroreflex gain is rendered to ECG / ABP waveforms, detected back, and analysed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clbench.beats import (
    build_beat_table, detect_abp_beats, detect_ppg_beats, detect_r_peaks, gate_beats, match_peaks, respiration_phase,
)
from clbench.edges.fast_coupling import (
    BeatLoopSpec, analyse_epoch, beat_regression, beat_regression_batch, bh_qvalues, bin_means, causal_transfer,
    iaaft, lagged_timescale_fit, render_beat_waveforms, seq_concordance_batch, sequence_counts_fast, sequence_method,
    simulate_beat_loop,
)

FS = 250.0


@pytest.fixture(scope="module")
def loop_beats() -> pd.DataFrame:
    return simulate_beat_loop(400, BeatLoopSpec(brs=8.0), seed=3)


@pytest.fixture(scope="module")
def waveforms(loop_beats):
    return render_beat_waveforms(loop_beats, FS, seed=1)


# --------------------------------------------------------------------------- beats


@pytest.mark.parametrize("invert", [False, True])
def test_r_peaks_match_truth(loop_beats, invert):
    ecg, _ = render_beat_waveforms(loop_beats, FS, invert=invert, seed=2)
    r = detect_r_peaks(ecg, FS) / FS
    truth = loop_beats["t_r"].to_numpy()
    tp, fn, fp = match_peaks(truth, r, 0.010)
    assert tp / (tp + fn) >= 0.99
    assert tp / (tp + fp) >= 0.99


def test_r_peaks_survive_nan_gap(loop_beats):
    ecg, _ = render_beat_waveforms(loop_beats, FS, seed=2)
    ecg = ecg.copy()
    ecg[int(60 * FS):int(65 * FS)] = np.nan
    r = detect_r_peaks(ecg, FS) / FS
    truth = loop_beats["t_r"].to_numpy()
    keep = (truth < 59.5) | (truth > 65.5)
    tp, fn, _ = match_peaks(truth[keep], r, 0.010)
    assert tp / (tp + fn) >= 0.98
    assert not np.any((r > 60.2) & (r < 64.8))


def test_abp_beats_recover_pressures(loop_beats, waveforms):
    _, abp = waveforms
    pb = detect_abp_beats(abp, FS)
    ts = pb.i_sys / FS
    truth_ts = loop_beats["t_sys"].to_numpy()
    tp, fn, fp = match_peaks(truth_ts, ts, 0.02)
    assert tp / (tp + fn) > 0.98 and fp <= 3
    j = np.searchsorted(ts, truth_ts - 0.02)
    ok = j < len(ts)
    err = np.abs(pb.sbp[j[ok]] - loop_beats["sbp"].to_numpy()[ok])
    assert np.median(err) < 1.5


def test_beat_table_pairs_ecg_and_abp(loop_beats, waveforms):
    ecg, abp = waveforms
    bt = gate_beats(build_beat_table(ecg, FS, abp, FS))
    assert bt["valid"].mean() > 0.9
    m = pd.merge_asof(bt.sort_values("t_r"), loop_beats[["t_r", "sbp", "rri"]].rename(columns={"sbp": "sbp_true",
                                                                                                  "rri": "rri_true"}),
                      on="t_r", direction="nearest", tolerance=0.01)
    v = m["valid"] & m["sbp_true"].notna()
    assert np.median(np.abs(m.loc[v, "sbp"] - m.loc[v, "sbp_true"])) < 1.5
    assert np.median(np.abs(m.loc[v, "rri"] - m.loc[v, "rri_true"])) < 5.0
    assert np.all(m.loc[v, "sys_delay"].between(0.25, 0.35))


def test_gate_rejects_ectopic_beat(loop_beats):
    b = loop_beats.copy().reset_index(drop=True)
    b["valid"] = True
    i = 100
    b.loc[i, "rri"] = b.loc[i, "rri"] * 0.55  # premature
    b.loc[i + 1, "rri"] = b.loc[i + 1, "rri"] * 1.45  # compensatory pause
    g = gate_beats(b)
    assert not g.loc[i, "valid"] and not g.loc[i + 1, "valid"]
    assert g["valid"].mean() > 0.95
    b2 = loop_beats.copy().reset_index(drop=True)
    b2.loc[50, "sbp"] = 300.0
    assert not gate_beats(b2).loc[50, "valid"]


def test_ppg_foot_and_pat():
    rng = np.random.default_rng(0)
    rr = 0.85 + 0.03 * rng.standard_normal(200)
    tr = np.concatenate([[1.0], 1.0 + np.cumsum(rr[:-1])])
    fs = 125.0
    n = int((tr[-1] + 2) * fs)
    t = np.arange(n) / fs
    ppg = np.zeros(n)
    for r in tr:
        foot, peak = r + 0.20, r + 0.33
        up = (t >= foot) & (t < peak)
        ppg[up] += 0.5 * (1 - np.cos(np.pi * (t[up] - foot) / (peak - foot)))
        dn = (t >= peak) & (t < peak + 0.8)
        ppg[dn] += np.exp(-(t[dn] - peak) / 0.25)
    ppg += 0.005 * rng.standard_normal(n)
    pg = detect_ppg_beats(ppg, fs)
    ok = np.isfinite(pg.t_foot)
    assert ok.sum() > 190
    bt = build_beat_table(None, fs, None, fs, ppg=ppg, fs_ppg=fs, r_peaks=np.round(tr * fs).astype(int))
    pat = bt["pat"].dropna()
    assert len(pat) > 180
    assert 180 < np.median(pat) < 240  # tangent foot sits at or slightly after the true 200 ms foot


def test_respiration_phase_frequency():
    fs = 25.0
    t = np.arange(0, 300, 1 / fs)
    x = np.sin(2 * np.pi * 0.25 * t) + 0.1 * np.random.default_rng(1).standard_normal(len(t))
    r = respiration_phase(x, fs)
    assert abs(r.freq_hz - 0.25) < 0.02
    assert np.isfinite(r.phase).mean() > 0.99


# --------------------------------------------------------------------------- estimators


def test_sequence_method_hand_example():
    sbp = np.array([100, 102, 104, 106, 105, 103, 101, 101, 101, 101.0])
    rri = np.array([800, 810, 820, 830, 825, 815, 805, 805, 805, 805.0])
    v = np.ones(10, dtype=bool)
    s = sequence_method(sbp, rri, v, lag=0)
    assert s.n_conc == 2 and s.n_disc == 0
    assert s.brs == pytest.approx(5.0, rel=1e-6)
    assert s.concordance == 1.0
    s2 = sequence_method(sbp, -rri + 1600, v, lag=0)  # mirrored RRI
    assert s2.n_conc == 0 and s2.n_disc == 2 and s2.concordance == -1.0


def test_batch_versions_match_scalar(loop_beats):
    v = loop_beats["valid"].to_numpy()
    sbp, rri = loop_beats["sbp"].to_numpy(), loop_beats["rri"].to_numpy()
    r = beat_regression(sbp, rri, v)
    rb = beat_regression_batch(sbp[None, :], rri, v)
    assert rb[0] == pytest.approx(r.total, rel=1e-8)
    c = d = 0
    for k in (0, 1, 2):
        cc, dd = sequence_counts_fast(sbp[:len(sbp) - k], rri[k:], v[:len(sbp) - k] & v[k:])
        c, d = c + cc, d + dd
    assert seq_concordance_batch(sbp[None, :], rri, v)[0] == pytest.approx((c - d) / (c + d))


def test_iaaft_properties():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(512))
    x = x - x.mean()
    s = iaaft(x, 5, rng)
    assert s.shape == (5, 512)
    for row in s:
        assert np.allclose(np.sort(row), np.sort(x))
        a, b = np.abs(np.fft.rfft(x))[1:60], np.abs(np.fft.rfft(row))[1:60]
        assert np.corrcoef(np.log(a), np.log(b))[0, 1] > 0.8
    assert abs(np.corrcoef(s[0], x)[0, 1]) < 0.9  # not a copy


def test_arx_recovers_planted_gain_and_null():
    b = simulate_beat_loop(3000, BeatLoopSpec(brs=8.0), seed=5)
    v = b["valid"].to_numpy()
    r = beat_regression(b["sbp"].to_numpy(), b["rri"].to_numpy(), v)
    assert 6.0 < r.total < 11.0
    ct = causal_transfer(r, (0, 1, 2), b["rri"].mean() / 1000.0)
    assert ct["re_sign"] > 0.5
    b0 = simulate_beat_loop(3000, BeatLoopSpec(brs=0.0), seed=5)
    r0 = beat_regression(b0["sbp"].to_numpy(), b0["rri"].to_numpy(), b0["valid"].to_numpy())
    assert abs(r0.total) < 1.5


def test_epoch_analysis_detects_reflex_and_respects_null():
    rng = np.random.default_rng(0)
    b = simulate_beat_loop(400, BeatLoopSpec(brs=8.0), seed=11)
    res = analyse_epoch(b, 0.0, 300.0, n_surr=39, rng=rng)
    assert res["reg_brs"] > 4 and res["p_reg"] <= 0.05
    assert res["seq_conc_raw"] > 0.3  # the sequence surrogate test has little power in one 5-min epoch
    assert res["tf_LF_resign"] > 0.3 and res["p_tf_coh"] <= 0.05
    assert res["arx_LF_resign"] > 0.5
    small = 0
    for k in range(8):
        bn = simulate_beat_loop(400, BeatLoopSpec(brs=0.0), seed=200 + k)
        rn = analyse_epoch(bn, 0.0, 300.0, n_surr=39, rng=np.random.default_rng(k))
        small += rn["p_reg"] <= 0.05
    assert small <= 3


def test_end_to_end_from_waveforms_matches_beat_truth():
    beats = simulate_beat_loop(420, BeatLoopSpec(brs=8.0), seed=21)
    ecg, abp = render_beat_waveforms(beats, FS, seed=4)
    bt = gate_beats(build_beat_table(ecg, FS, abp, FS))
    est = analyse_epoch(bt, 5.0, 305.0, n_surr=0)
    truth = analyse_epoch(beats, 5.0, 305.0, n_surr=0)
    assert est["reg_brs"] == pytest.approx(truth["reg_brs"], abs=1.5)
    assert np.sign(est["seq_conc_raw"]) == np.sign(truth["seq_conc_raw"]) == 1


def test_slow_common_driver_flips_window_sign_not_beat_sign():
    """The mechanism the paper claims: a slow driver that raises pressure and heart rate together
    makes lagged window-level MAP -> HR positive (wrong) while the beat-level detrended ARX keeps
    the planted positive SBP -> RRI gain."""
    spec = BeatLoopSpec(brs=8.0, driver_sd=6.0, driver_tau=300.0)
    wrong_30s, right_beat = 0, 0
    for seed in range(3):
        b = simulate_beat_loop(6000, spec, seed=seed)
        v = b["valid"].to_numpy()
        r = beat_regression(b["sbp"].to_numpy(), b["rri"].to_numpy(), v)
        right_beat += r.total > 0
        T = float(b["t_r"].iloc[-1])
        hr = 60000.0 / b["rri"].to_numpy()
        H = bin_means((b["t_r"] + b["rri"] / 1000).to_numpy(), hr, v, 0.0, T, 30.0)
        M = bin_means(b["t_sys"].to_numpy(), b["map_beat"].to_numpy(), v, 0.0, T, 30.0)
        w = lagged_timescale_fit([(H, M)], n_lags=1)
        wrong_30s += w.total > 0
    assert right_beat == 3
    assert wrong_30s >= 2


def test_bh_qvalues():
    p = np.array([0.01, 0.04, 0.03, np.nan, 0.5])
    q = bh_qvalues(p)
    assert np.isnan(q[3])
    assert q[0] == pytest.approx(0.04)
    assert np.all(q[np.isfinite(q)] >= p[np.isfinite(p)])
