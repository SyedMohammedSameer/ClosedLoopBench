"""
Beat-level feature extraction from raw waveforms (ECG, arterial pressure, PPG, respiration).

This module turns continuous waveforms into a *beat table*: one row per cardiac cycle with the
R-peak time, the RR interval that follows it, the systolic / diastolic / mean pressure of the pulse
ejected by that beat, the PPG foot and peak, pulse arrival time, and the respiratory phase at the
R-peak. Everything downstream (``clbench.edges.fast_coupling``) works on this table.

Conventions (fixed, used by every estimator):

* ``t_r[n]`` is the n-th R-peak (seconds from the start of the signal).
* ``rri[n] = t_r[n+1] - t_r[n]`` in milliseconds: the interval that *starts* at beat n.
* ``sbp[n]`` is the systolic peak of the pulse ejected at R-peak n. It occurs inside ``rri[n]``
  (typically 150-400 ms after the R-peak), so SBP(n) -> RRI(n) is causally admissible
  ("lag 0" in the baroreflex sequence literature) while RRI(n) -> SBP(n) is not.
* ``dbp[n]`` is the end-diastolic minimum just before the upstroke of pulse n.
* ``map_beat[n]`` is the mean pressure from the foot of pulse n to the foot of pulse n+1.

Detection methods are deliberately classical and transparent:

* R-peaks: a Pan-Tompkins style detector (band-pass 5-18 Hz, derivative, squaring,
  120 ms moving-window integration, adaptive block threshold, T-wave rejection and search-back),
  with polarity estimated per call so inverted leads (aVR) work.
* Arterial pressure: systolic peaks by prominence relative to the local pulse pressure (rejects
  dicrotic notches), diastolic minimum between consecutive systoles.
* PPG: peak by prominence, foot by the intersecting-tangent method.
* Respiration: band-pass + Hilbert phase, and the dominant respiratory frequency.

Quality gating (:func:`gate_beats`) enforces physiological ranges (RRI 300-2000 ms, SBP 40-250,
DBP 15-150, pulse pressure 10-150 mmHg) and rejects local outliers (ectopy, missed/extra
detections, line flushes) against an 11-beat running median.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, welch

# Physiological gates
RRI_MIN_MS, RRI_MAX_MS = 300.0, 2000.0
SBP_MIN, SBP_MAX = 40.0, 250.0
DBP_MIN, DBP_MAX = 15.0, 150.0
PP_MIN, PP_MAX = 10.0, 150.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fill_nan(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate NaNs (edges held); return (filled, nan_mask)."""
    x = np.asarray(x, dtype=np.float64)
    bad = ~np.isfinite(x)
    if not bad.any():
        return x.copy(), bad
    if bad.all():
        return np.zeros_like(x), bad
    idx = np.arange(len(x))
    y = x.copy()
    y[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return y, bad


def _band(x: np.ndarray, fs: float, lo: float | None, hi: float | None, order: int = 2) -> np.ndarray:
    nyq = fs / 2.0
    if hi is not None and hi >= nyq * 0.95:
        hi = None
    if lo is not None and hi is not None:
        sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    elif lo is not None:
        sos = butter(order, lo, btype="high", fs=fs, output="sos")
    elif hi is not None:
        sos = butter(order, hi, btype="low", fs=fs, output="sos")
    else:
        return np.asarray(x, dtype=np.float64)
    if len(x) <= 3 * (2 * order + 1) * 3:
        return np.asarray(x, dtype=np.float64)
    return sosfiltfilt(sos, x)


def _block_level(env: np.ndarray, fs: float, block_sec: float = 2.5, q: float = 98.0,
                 smooth_blocks: int = 5) -> np.ndarray:
    """Slowly varying signal level: per-block percentile, running median over blocks, interpolated."""
    b = max(int(block_sec * fs), 1)
    nb = int(np.ceil(len(env) / b))
    lv = np.array([np.percentile(env[i * b:(i + 1) * b], q) for i in range(nb)])
    if nb >= smooth_blocks:
        pad = smooth_blocks // 2
        lp = np.pad(lv, pad, mode="edge")
        lv = np.array([np.median(lp[i:i + smooth_blocks]) for i in range(nb)])
    centres = (np.arange(nb) + 0.5) * b
    return np.interp(np.arange(len(env)), centres, lv)


def _local_argmax(x: np.ndarray, centres: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Index of the maximum of ``x`` within [c - lo, c + hi] for each centre (vectorised)."""
    n = len(x)
    offs = np.arange(-lo, hi + 1)
    idx = np.clip(centres[:, None] + offs[None, :], 0, n - 1)
    return idx[np.arange(len(centres)), np.argmax(x[idx], axis=1)]


def _running_median(x: np.ndarray, k: int = 11) -> np.ndarray:
    """Centred running median that ignores NaN (edges use a shrunken window)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x.copy()
    h = k // 2
    xp = np.pad(x, h, mode="constant", constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(xp, k)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(win, axis=1)


# ---------------------------------------------------------------------------
# ECG
# ---------------------------------------------------------------------------


def detect_r_peaks(ecg: np.ndarray, fs: float, *, refractory_sec: float = 0.25,
                   threshold_frac: float = 0.25, search_back: bool = True) -> np.ndarray:
    """
    R-peak sample indices from one ECG lead (Pan-Tompkins style, polarity-adaptive).

    Returns sorted int indices into ``ecg``. Peaks within 0.25 s of a NaN run are dropped.
    """
    x, bad = _fill_nan(ecg)
    n = len(x)
    if n < int(2 * fs) or np.ptp(x) == 0:
        return np.empty(0, dtype=np.int64)
    f = _band(x, fs, 5.0, 18.0, order=2)
    d = np.gradient(f) * fs
    w = max(int(round(0.12 * fs)), 1)
    mwi = np.convolve(d * d, np.ones(w) / w, mode="same")
    level = _block_level(mwi, fs)
    level = np.maximum(level, 1e-12)
    dist = max(int(refractory_sec * fs), 1)
    peaks, props = find_peaks(mwi, height=threshold_frac * level, distance=dist)
    if len(peaks) == 0:
        return np.empty(0, dtype=np.int64)

    # search-back for missed beats in long gaps
    if search_back and len(peaks) > 4:
        rr = np.diff(peaks)
        med = _running_median(rr.astype(float), 9)
        add = []
        for i in np.where(rr > 1.6 * med)[0]:
            a, b = peaks[i] + dist, peaks[i + 1] - dist
            if b <= a:
                continue
            seg = mwi[a:b]
            cand, _ = find_peaks(seg, height=0.4 * threshold_frac * level[a:b], distance=dist)
            if len(cand):
                add.append(a + cand[np.argmax(seg[cand])])
        if add:
            peaks = np.sort(np.concatenate([peaks, np.array(add, dtype=peaks.dtype)]))

    # T-wave rejection: a peak < 0.36 s after the previous one with < 50 % of its energy
    keep = np.ones(len(peaks), dtype=bool)
    tw = int(0.36 * fs)
    for i in range(1, len(peaks)):
        j = i - 1
        while j >= 0 and not keep[j]:
            j -= 1
        if j >= 0 and peaks[i] - peaks[j] < tw and mwi[peaks[i]] < 0.5 * mwi[peaks[j]]:
            keep[i] = False
    peaks = peaks[keep]

    # refine to the R apex on the band-passed then raw signal, with estimated polarity
    lo = hi = int(round(0.075 * fs))
    cand_pos = _local_argmax(f, peaks, lo, hi)
    cand_neg = _local_argmax(-f, peaks, lo, hi)
    pol = 1.0 if np.median(f[cand_pos]) >= np.median(-f[cand_neg]) else -1.0
    r = _local_argmax(pol * f, peaks, lo, hi)
    k = max(int(round(0.015 * fs)), 1)
    r = _local_argmax(pol * x, r, k, k)
    r = np.unique(r)
    # enforce refractory after refinement (keep the larger apex)
    if len(r) > 1:
        keep = np.ones(len(r), dtype=bool)
        for i in range(1, len(r)):
            if r[i] - r[i - 1] < dist:
                if pol * x[r[i]] > pol * x[r[i - 1]]:
                    keep[i - 1] = False
                else:
                    keep[i] = False
        r = r[keep]
    if bad.any():
        near = np.convolve(bad.astype(float), np.ones(max(int(0.25 * fs), 1)), mode="same") > 0
        r = r[~near[r]]
    return r.astype(np.int64)


# ---------------------------------------------------------------------------
# Arterial pressure
# ---------------------------------------------------------------------------


@dataclass
class PressureBeats:
    """Per-pulse arterial pressure features (sample indices into the ABP array)."""

    i_sys: np.ndarray
    i_foot: np.ndarray
    sbp: np.ndarray
    dbp: np.ndarray
    map_beat: np.ndarray


def detect_abp_beats(abp: np.ndarray, fs: float, *, min_prominence: float = 5.0,
                     rel_prominence: float = 0.4) -> PressureBeats:
    """
    Systolic peaks and diastolic feet of an arterial pressure waveform (mmHg).

    Peaks need a prominence of at least ``rel_prominence`` x the local median pulse prominence
    (this rejects dicrotic notches / reflected waves). The foot of each pulse is the minimum
    between the previous systolic peak (or 1.5 s earlier) and this one.
    """
    x, bad = _fill_nan(abp)
    empty = PressureBeats(*(np.empty(0, dtype=np.int64),) * 2, *(np.empty(0),) * 3)
    if len(x) < int(2 * fs) or np.ptp(x) < 5:
        return empty
    xf = _band(x, fs, None, 16.0, order=2)
    dist = max(int(0.27 * fs), 1)
    pk, pr = find_peaks(xf, distance=dist, prominence=min_prominence)
    if len(pk) < 3:
        return empty
    prom = pr["prominences"]
    ref = _running_median(prom, 15)
    pk = pk[prom >= rel_prominence * ref]
    if len(pk) < 3:
        return empty
    foot = np.empty(len(pk), dtype=np.int64)
    maxback = int(1.5 * fs)
    for j, p in enumerate(pk):
        a = pk[j - 1] if j > 0 else max(p - maxback, 0)
        a = max(a, p - maxback)
        foot[j] = a + int(np.argmin(xf[a:p + 1])) if p > a else p
    sbp = xf[pk]
    dbp = xf[foot]
    cs = np.concatenate([[0.0], np.cumsum(xf)])
    mp = np.full(len(pk), np.nan)
    nxt = np.append(foot[1:], -1)
    ok = nxt > foot
    mp[ok] = (cs[nxt[ok]] - cs[foot[ok]]) / (nxt[ok] - foot[ok])
    if bad.any():
        near = np.convolve(bad.astype(float), np.ones(max(int(0.5 * fs), 1)), mode="same") > 0
        drop = near[pk] | near[foot]
        sbp = np.where(drop, np.nan, sbp)
        dbp = np.where(drop, np.nan, dbp)
        mp = np.where(drop, np.nan, mp)
    return PressureBeats(pk.astype(np.int64), foot, sbp, dbp, mp)


# ---------------------------------------------------------------------------
# PPG
# ---------------------------------------------------------------------------


@dataclass
class PPGBeats:
    t_foot: np.ndarray  # seconds, intersecting-tangent foot
    t_peak: np.ndarray  # seconds
    amp: np.ndarray  # peak - preceding minimum (signal units)


def detect_ppg_beats(ppg: np.ndarray, fs: float, *, rel_prominence: float = 0.35) -> PPGBeats:
    """PPG pulse peaks and feet (intersecting tangent at the maximum upslope)."""
    x, bad = _fill_nan(ppg)
    empty = PPGBeats(np.empty(0), np.empty(0), np.empty(0))
    if len(x) < int(2 * fs) or np.ptp(x) == 0:
        return empty
    xf = _band(x, fs, 0.4, 10.0, order=2)
    scale = np.percentile(xf, 95) - np.percentile(xf, 5)
    if scale <= 0:
        return empty
    pk, pr = find_peaks(xf, distance=max(int(0.27 * fs), 1), prominence=0.05 * scale)
    if len(pk) < 3:
        return empty
    prom = pr["prominences"]
    pk = pk[prom >= rel_prominence * _running_median(prom, 15)]
    if len(pk) < 3:
        return empty
    dx = np.gradient(xf) * fs
    t_foot = np.full(len(pk), np.nan)
    amp = np.full(len(pk), np.nan)
    maxback = int(0.8 * fs)
    for j, p in enumerate(pk):
        a = max(pk[j - 1] if j > 0 else 0, p - maxback)
        if p - a < 3:
            continue
        imin = a + int(np.argmin(xf[a:p + 1]))
        if p - imin < 2:
            continue
        ims = imin + int(np.argmax(dx[imin:p + 1]))
        slope = dx[ims]
        if slope <= 0:
            continue
        t_foot[j] = ims / fs - (xf[ims] - xf[imin]) / slope  # slope is per second
        amp[j] = xf[p] - xf[imin]
    t_peak = pk / fs
    if bad.any():
        near = np.convolve(bad.astype(float), np.ones(max(int(0.5 * fs), 1)), mode="same") > 0
        drop = near[pk]
        t_foot[drop] = np.nan
        amp[drop] = np.nan
    return PPGBeats(t_foot, t_peak, amp)


# ---------------------------------------------------------------------------
# Respiration
# ---------------------------------------------------------------------------


@dataclass
class Respiration:
    phase: np.ndarray  # radians, per sample of the input, NaN where unavailable
    freq_hz: float  # dominant respiratory frequency (NaN if not found)
    filtered: np.ndarray  # band-passed respiratory signal


def respiration_phase(resp: np.ndarray, fs: float, *, band: tuple[float, float] = (0.08, 0.8)) -> Respiration:
    """
    Instantaneous respiratory phase from an impedance, capnogram or airway-pressure signal.

    The signal is band-passed around the dominant respiratory frequency (found by Welch within
    ``band``) and its Hilbert phase is returned. Phase 0 is the maximum of the filtered signal
    (end-inspiration for impedance/airway pressure; end-expiration for CO2).
    """
    x, bad = _fill_nan(resp)
    n = len(x)
    if n < int(20 * fs) or np.ptp(x) == 0:
        return Respiration(np.full(n, np.nan), float("nan"), np.full(n, np.nan))
    xb = _band(x, fs, band[0], band[1], order=2)
    f, p = welch(xb, fs=fs, nperseg=min(n, int(64 * fs)))
    m = (f >= band[0]) & (f <= band[1])
    if not m.any() or p[m].max() <= 0:
        return Respiration(np.full(n, np.nan), float("nan"), xb)
    f0 = float(f[m][np.argmax(p[m])])
    lo, hi = max(band[0], 0.6 * f0), min(band[1] * 1.2, 1.6 * f0)
    xn = _band(x, fs, lo, hi, order=2)
    ph = np.angle(hilbert(xn))
    ph[bad] = np.nan
    return Respiration(ph, f0, xn)


# ---------------------------------------------------------------------------
# Beat table
# ---------------------------------------------------------------------------


BEAT_COLUMNS = ["t_r", "rri", "sbp", "dbp", "map_beat", "pp", "t_sys", "sys_delay",
                "t_ppg_foot", "pat", "ppg_amp", "resp_phase", "valid"]


def build_beat_table(
    ecg: np.ndarray | None, fs_ecg: float,
    abp: np.ndarray | None, fs_abp: float,
    *,
    ppg: np.ndarray | None = None, fs_ppg: float | None = None,
    resp: np.ndarray | None = None, fs_resp: float | None = None,
    t0: float = 0.0,
    r_peaks: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    One row per R-peak with RRI and the matched arterial pulse (and PPG / respiration if given).

    All times are seconds from ``t0`` (the time of sample 0; the signals must share it). The
    pressure pulse matched to R-peak n is the first systolic peak at least 50 ms after it, provided
    its delay is within 150 ms of the segment's median R-to-systole delay and it precedes
    R-peak n+1 by no more than 100 ms. Unmatched beats have NaN pressures. Run :func:`gate_beats`
    afterwards.
    """
    if r_peaks is None:
        if ecg is None:
            raise ValueError("need ecg or r_peaks")
        r_peaks = detect_r_peaks(ecg, fs_ecg)
    tr = np.asarray(r_peaks, dtype=np.float64) / fs_ecg
    nb = len(tr)
    out = {c: np.full(nb, np.nan) for c in BEAT_COLUMNS}
    out["t_r"] = tr + t0
    out["valid"] = np.zeros(nb, dtype=bool)
    if nb < 3:
        return pd.DataFrame(out)
    rri = np.full(nb, np.nan)
    rri[:-1] = np.diff(tr) * 1000.0
    out["rri"] = rri

    if abp is not None:
        pb = detect_abp_beats(abp, fs_abp)
        if len(pb.i_sys):
            ts = pb.i_sys / fs_abp
            j = np.searchsorted(ts, tr + 0.05)
            okj = j < len(ts)
            delay = np.full(nb, np.nan)
            delay[okj] = ts[j[okj]] - tr[okj]
            med = np.nanmedian(delay[(delay > 0.05) & (delay < 0.7)]) if np.any((delay > 0.05) & (delay < 0.7)) else np.nan
            nxt = np.append(tr[1:], np.inf)
            good = okj & np.isfinite(delay) & (np.abs(delay - med) < 0.15) & (tr + delay < nxt + 0.1)
            jj = j[good]
            for name, arr in (("sbp", pb.sbp), ("dbp", pb.dbp), ("map_beat", pb.map_beat)):
                v = np.full(nb, np.nan)
                v[good] = arr[jj]
                out[name] = v
            ts_full = np.full(nb, np.nan)
            ts_full[good] = ts[jj] + t0
            out["t_sys"] = ts_full
            out["sys_delay"] = np.where(good, delay, np.nan)
            out["pp"] = out["sbp"] - out["dbp"]
            # a systolic peak may be claimed twice only if R detection doubled; keep the first
            if good.any():
                dup = np.zeros(nb, dtype=bool)
                gi = np.where(good)[0]
                dup[gi[1:]] = jj[1:] == jj[:-1]
                for c in ("sbp", "dbp", "map_beat", "pp", "t_sys", "sys_delay"):
                    out[c][dup] = np.nan

    if ppg is not None and fs_ppg:
        pg = detect_ppg_beats(ppg, fs_ppg)
        tf = pg.t_foot
        okf = np.isfinite(tf)
        tf_, amp_ = tf[okf], pg.amp[okf]
        if len(tf_):
            j = np.searchsorted(tf_, tr + 0.08)
            okj = j < len(tf_)
            pat = np.full(nb, np.nan)
            pat[okj] = tf_[j[okj]] - tr[okj]
            nxt = np.append(tr[1:], np.inf)
            good = okj & (pat > 0.08) & (pat < 0.6) & (tr + pat < nxt + 0.15)
            out["pat"] = np.where(good, pat * 1000.0, np.nan)
            out["t_ppg_foot"] = np.where(good, tr + pat + t0, np.nan)
            a = np.full(nb, np.nan)
            a[good] = amp_[j[good]]
            out["ppg_amp"] = a

    if resp is not None and fs_resp:
        rp = respiration_phase(resp, fs_resp)
        idx = np.clip(np.round(tr * fs_resp).astype(np.int64), 0, len(rp.phase) - 1)
        out["resp_phase"] = rp.phase[idx]

    return pd.DataFrame(out)


def gate_beats(beats: pd.DataFrame, *, rel_rri: float = 0.2, abs_sbp_jump: float = 30.0,
               k: int = 11, require_pressure: bool = True) -> pd.DataFrame:
    """
    Add a boolean ``valid`` column: physiological ranges plus local-outlier rejection.

    A beat is invalid if its RRI is outside 300-2000 ms or deviates > ``rel_rri`` from the
    ``k``-beat running median (ectopy, missed or extra detections; the beat after an invalid RRI
    is also dropped as a possible compensatory pause), or if its pressures are outside range,
    pulse pressure is implausible, or SBP deviates by more than ``abs_sbp_jump`` mmHg from the
    running median (flush, damping, artefact).
    """
    b = beats.copy()
    rri = b["rri"].to_numpy(dtype=float)
    ok = np.isfinite(rri) & (rri >= RRI_MIN_MS) & (rri <= RRI_MAX_MS)
    med = _running_median(np.where(ok, rri, np.nan), k)
    ok &= np.abs(rri - med) <= rel_rri * med
    prev_bad = np.concatenate([[False], ~ok[:-1]])
    ok &= ~prev_bad
    if require_pressure:
        sbp = b["sbp"].to_numpy(dtype=float)
        dbp = b["dbp"].to_numpy(dtype=float)
        pp = sbp - dbp
        okp = (np.isfinite(sbp) & np.isfinite(dbp) & (sbp >= SBP_MIN) & (sbp <= SBP_MAX)
               & (dbp >= DBP_MIN) & (dbp <= DBP_MAX) & (pp >= PP_MIN) & (pp <= PP_MAX))
        msb = _running_median(np.where(okp, sbp, np.nan), k)
        okp &= np.abs(sbp - msb) <= abs_sbp_jump
        ok &= okp
    b["valid"] = ok
    return b


def beat_hr(beats: pd.DataFrame) -> np.ndarray:
    """Instantaneous heart rate (bpm) per beat from the RRI that starts at that beat."""
    return 60000.0 / beats["rri"].to_numpy(dtype=float)


def match_peaks(ref: np.ndarray, test: np.ndarray, tol: float) -> tuple[int, int, int]:
    """(true positives, false negatives, false positives) matching sorted event times within tol."""
    ref = np.sort(np.asarray(ref, dtype=float))
    test = np.sort(np.asarray(test, dtype=float))
    if len(ref) == 0 or len(test) == 0:
        return 0, len(ref), len(test)
    j = np.searchsorted(test, ref)
    used = np.zeros(len(test), dtype=bool)
    tp = 0
    for i, r in enumerate(ref):
        best, bd = -1, tol
        for c in (j[i] - 1, j[i]):
            if 0 <= c < len(test) and not used[c] and abs(test[c] - r) <= bd:
                best, bd = c, abs(test[c] - r)
        if best >= 0:
            used[best] = True
            tp += 1
    return tp, len(ref) - tp, int((~used).sum())
