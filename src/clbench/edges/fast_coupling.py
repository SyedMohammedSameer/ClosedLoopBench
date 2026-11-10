"""
Beat-level fast coupling edges: baroreflex, respiratory sinus arrhythmia, pulse arrival time and
respiratory pressure variation, with surrogate nulls and a timescale sweep.

Why beat level. Monitor numerics (2 s) and 30 s windows mix three things that act on heart rate
and pressure: the fast reflexes (baroreflex latency 0.3-2 s, RSA within a breath), slow common
drivers (surgical stimulation, anaesthetic depth, volume status) and the clinician's control loop
(minutes). At beat resolution the reflex has a known causal ordering inside the cardiac cycle
(SBP(n) is ejected inside RRI(n); RRI(n) cannot affect SBP(n)), and slow drivers are nearly
constant over a few beats, so the reflex is estimable from lags. After averaging to windows the
reflex becomes contemporaneous and the slow drivers dominate the lagged covariance.

Estimators (all take a gated beat table from :mod:`clbench.beats`):

``sequence_method``     Bertinieri / Parati sequence technique. Concordant sequences (>= 3 beats,
                        SBP and RRI moving the same way by >= 1 mmHg / >= 5 ms per beat, r >= 0.85)
                        at lag 0/1/2 beats. BRS (ms/mmHg) is POSITIVE BY CONSTRUCTION, so the sign
                        test uses the *concordance index* (n_concordant - n_discordant) /
                        (n_concordant + n_discordant), where discordant sequences are the mirror
                        (SBP up / RRI down); 0 under no coupling.
``beat_regression``     RRI(n) on its own two lags and SBP(n), SBP(n-1), SBP(n-2) after removing a
                        31-beat running mean (closed-loop ARX with the physiological ordering).
                        The summed SBP coefficient (ms/mmHg) should be positive.
``transfer_function``   Welch cross-spectrum of SBP -> RRI (4 Hz resampled). LF band 0.04-0.15 Hz:
                        gain (mean |H| where coherence >= 0.5), coherence, phase and the
                        coherence-weighted sign of Re(H) (should be positive).
``band_sign_spectrum``  Coherence-weighted sign of the HR-MAP cross-spectrum in bands from 0.0017 Hz
                        (10 min) to 0.4 Hz on long contiguous blocks: which timescales carry the
                        reflex (negative HR-MAP co-movement) and which carry common drivers.
``lagged_timescale_fit`` The September estimator transplanted to any step: HR_t on HR lags and MAP
                        lags (lags only), pooled across blocks with block intercepts. Sign of the
                        summed MAP coefficient (should be negative).
``rsa_coupling``        Respiration -> RRI gain and coherence at the respiratory frequency, and RSA
                        amplitude by respiratory-phase binning.
``pat_coupling``        Pulse arrival time (R -> PPG foot) and its beat-level coupling with SBP
                        (should be negative: higher pressure, stiffer arteries, faster arrival).
``respiratory_pressure_variation``  SPV (mmHg) and PPV (%) per breath.

Surrogates: IAAFT (iterative amplitude-adjusted Fourier transform) of the input series, which keeps
its distribution and spectrum but destroys cross-dependence, and circular block shifts.
``simulate_beat_loop`` is a beat-by-beat closed-loop generator (DeBoer-style) with a planted
baroreflex, a Mayer-wave oscillator, respiration and a slow common driver, used for tests and for
calibrating the timescale-reversal claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import csd, welch

LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)
SIGN_BANDS: dict[str, tuple[float, float]] = {
    "b1_10-3min": (1 / 600, 1 / 180),
    "b2_3-1min": (1 / 180, 1 / 60),
    "b3_60-25s": (1 / 60, 0.04),
    "b4_LF_25-7s": (0.04, 0.15),
    "b5_HF_7-2.5s": (0.15, 0.40),
}


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Starts and lengths of maximal runs of True."""
    m = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    d = np.diff(m)
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return s, e - s


def running_mean_detrend(x: np.ndarray, valid: np.ndarray, k: int = 31) -> np.ndarray:
    """Subtract a centred k-beat running mean computed over valid beats only (NaN elsewhere)."""
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(valid, dtype=bool) & np.isfinite(x)
    xv = np.where(v, x, 0.0)
    c = np.convolve(xv, np.ones(k), mode="same")
    n = np.convolve(v.astype(float), np.ones(k), mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        m = c / n
    out = np.where(v & (n >= k // 2), x - m, np.nan)
    return out


def iaaft(x: np.ndarray, n_surr: int, rng: np.random.Generator, n_iter: int = 20) -> np.ndarray:
    """
    ``n_surr`` IAAFT surrogates of a gap-free 1-D series, shape (n_surr, len(x)).

    Each surrogate has exactly the values of ``x`` (rank-ordered) and approximately its power
    spectrum; its dependence on any other series is destroyed.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    amp = np.abs(np.fft.rfft(x))
    xs = np.sort(x)
    s = np.stack([rng.permutation(x) for _ in range(n_surr)])
    xs_b = np.broadcast_to(xs, s.shape)
    for _ in range(n_iter):
        S = np.fft.rfft(s, axis=1)
        mag = np.abs(S)
        S = S * (amp[None, :] / np.where(mag > 0, mag, 1.0))
        s = np.fft.irfft(S, n, axis=1)
        order = np.argsort(s, axis=1, kind="stable")
        s = np.empty_like(s)
        np.put_along_axis(s, order, xs_b, axis=1)
    return s


def circular_shift_surrogates(x: np.ndarray, n_surr: int, rng: np.random.Generator,
                              min_shift: int) -> np.ndarray:
    """Circularly shifted copies of ``x`` (shift >= min_shift from either end)."""
    n = len(x)
    lo, hi = min_shift, max(n - min_shift, min_shift + 1)
    shifts = rng.integers(lo, hi, size=n_surr)
    return np.stack([np.roll(x, s) for s in shifts])


def _fill_invalid(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(valid, dtype=bool) & np.isfinite(x)
    if v.all():
        return x.copy()
    if not v.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    y = x.copy()
    y[~v] = np.interp(idx[~v], idx[v], x[v])
    return y


# ---------------------------------------------------------------------------
# Sequence method
# ---------------------------------------------------------------------------


@dataclass
class SequenceResult:
    lag: int
    n_conc: int  # baroreflex sequences (up-up + down-down) passing r >= r_thr
    n_disc: int  # mirror sequences (up-down + down-up) passing |r| >= r_thr
    n_ramps: int  # SBP ramps (>= min_len beats)
    brs: float  # mean slope of concordant sequences, ms/mmHg (NaN if none)
    bei: float  # baroreflex effectiveness index n_conc / n_ramps
    concordance: float  # (n_conc - n_disc) / (n_conc + n_disc), NaN if no sequences
    slopes: np.ndarray = field(default_factory=lambda: np.empty(0))


def sequence_method(sbp: np.ndarray, rri: np.ndarray, valid: np.ndarray, lag: int = 0, *,
                    min_len: int = 3, sbp_thr: float = 1.0, rri_thr: float = 5.0,
                    r_thr: float = 0.85) -> SequenceResult:
    """Sequence-method BRS at a beat lag (SBP(n) paired with RRI(n + lag))."""
    sbp = np.asarray(sbp, dtype=np.float64)
    rri = np.asarray(rri, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    n = len(sbp)
    if n - lag < min_len + 1:
        return SequenceResult(lag, 0, 0, 0, np.nan, np.nan, np.nan)
    x = sbp[: n - lag]
    y = rri[lag:]
    v = valid[: n - lag] & valid[lag:] & np.isfinite(x) & np.isfinite(y)
    dx = np.diff(x)
    dy = np.diff(y)
    vs = v[:-1] & v[1:]
    up = vs & (dx >= sbp_thr)
    dn = vs & (dx <= -sbp_thr)
    yu = dy >= rri_thr
    yd = dy <= -rri_thr
    need = min_len - 1

    def seqs(mask: np.ndarray, sign: int) -> list[float]:
        st, ln = _runs(mask)
        out = []
        for s, L in zip(st, ln):
            if L < need:
                continue
            xs = x[s:s + L + 1]
            ys = y[s:s + L + 1]
            sx, sy = xs.std(), ys.std()
            if sx == 0 or sy == 0:
                continue
            r = float(np.corrcoef(xs, ys)[0, 1])
            if sign * r >= r_thr:
                out.append(float(np.polyfit(xs, ys, 1)[0]))
        return out

    conc = seqs(up & yu, +1) + seqs(dn & yd, +1)
    disc = seqs(up & yd, -1) + seqs(dn & yu, -1)
    su, lu = _runs(up)
    sd, ld = _runs(dn)
    n_ramps = int((lu >= need).sum() + (ld >= need).sum())
    nc, nd = len(conc), len(disc)
    brs = float(np.mean(conc)) if nc else np.nan
    bei = nc / n_ramps if n_ramps else np.nan
    conc_idx = (nc - nd) / (nc + nd) if (nc + nd) else np.nan
    return SequenceResult(lag, nc, nd, n_ramps, brs, bei, conc_idx, np.asarray(conc))


def sequence_counts_fast(x: np.ndarray, y: np.ndarray, v: np.ndarray, *, min_len: int = 3,
                         sbp_thr: float = 1.0, rri_thr: float = 5.0) -> tuple[int, int]:
    """Concordant and discordant sequence counts without the r criterion (for surrogates)."""
    dx = np.diff(x)
    dy = np.diff(y)
    vs = v[:-1] & v[1:]
    up = vs & (dx >= sbp_thr)
    dn = vs & (dx <= -sbp_thr)
    yu = dy >= rri_thr
    yd = dy <= -rri_thr
    need = min_len - 1

    def cnt(m: np.ndarray) -> int:
        _, ln = _runs(m)
        return int((ln >= need).sum())

    return cnt(up & yu) + cnt(dn & yd), cnt(up & yd) + cnt(dn & yu)


# ---------------------------------------------------------------------------
# Beat-level closed-loop regression
# ---------------------------------------------------------------------------


@dataclass
class RegressionResult:
    coefs: np.ndarray  # input coefficients at each lag
    total: float  # sum of input coefficients
    se_total: float
    n: int
    ar: np.ndarray = field(default_factory=lambda: np.empty(0))  # own-lag coefficients (if any)


def _lag_design(y: np.ndarray, x: np.ndarray, ar: int, lags: tuple[int, ...],
                covariates: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rows n >= max lag: [y(n-1..n-ar), covariates(n), x(n-k) for k in lags]; inputs last."""
    n = len(y)
    L = max(max(lags), ar)
    rows = np.arange(L, n)
    cov_cols = [] if covariates is None else [np.asarray(covariates)[rows, j] for j in range(np.asarray(covariates).shape[1])]
    cols = [y[rows - i] for i in range(1, ar + 1)] + cov_cols + [x[rows - k] for k in lags]
    X = np.column_stack(cols) if cols else np.empty((len(rows), 0))
    Y = y[rows]
    ok = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    return Y[ok], X[ok]


def ols_total(Y: np.ndarray, X: np.ndarray, n_in: int) -> RegressionResult:
    """OLS with intercept; the last ``n_in`` columns are the input lags summed into ``total``."""
    n = len(Y)
    if n < X.shape[1] + 10:
        return RegressionResult(np.full(n_in, np.nan), np.nan, np.nan, n)
    Xc = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
    resid = Y - Xc @ beta
    s2 = resid @ resid / max(n - Xc.shape[1], 1)
    try:
        cov = s2 * np.linalg.inv(Xc.T @ Xc)
    except np.linalg.LinAlgError:
        return RegressionResult(np.full(n_in, np.nan), np.nan, np.nan, n)
    k = Xc.shape[1]
    idx = np.arange(k - n_in, k)
    w = np.zeros(k)
    w[idx] = 1.0
    total = float(beta[idx].sum())
    se = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return RegressionResult(beta[idx], total, se, n, beta[1:k - n_in])


def beat_regression(sbp: np.ndarray, rri: np.ndarray, valid: np.ndarray, *, lags: tuple[int, ...] = (0, 1, 2),
                    ar: int = 2, detrend: int | None = 31, covariates: np.ndarray | None = None) -> RegressionResult:
    """RRI(n) = a1 RRI(n-1) + a2 RRI(n-2) [+ c' z(n)] + sum_k b_k SBP(n-k) + e on (detrended) beat
    series. ``covariates`` (n_beats x p) enter at the current beat (e.g. respiratory-phase
    harmonics). ``RegressionResult.ar`` then holds the AR coefficients followed by the covariate
    coefficients; use ``ar_only`` slicing when building a transfer function."""
    if detrend:
        x = running_mean_detrend(sbp, valid, detrend)
        y = running_mean_detrend(rri, valid, detrend)
    else:
        v = np.asarray(valid, dtype=bool)
        x = np.where(v, sbp, np.nan)
        y = np.where(v, rri, np.nan)
    Y, X = _lag_design(y, x, ar, lags, covariates)
    return ols_total(Y, X, len(lags))


def resp_harmonics(phase: np.ndarray, k: int = 2) -> np.ndarray:
    """[sin(j phi), cos(j phi)] for j = 1..k (NaN where the phase is missing)."""
    ph = np.asarray(phase, dtype=np.float64)
    return np.column_stack([f(j * ph) for j in range(1, k + 1) for f in (np.sin, np.cos)])


def causal_transfer(res: RegressionResult, lags: tuple[int, ...], beat_sec: float,
                    band: tuple[float, float] = LF_BAND, n_freq: int = 12, n_ar: int | None = None) -> dict[str, float]:
    """
    Frequency response of the fitted closed-loop ARX feedback path,
    H(w) = sum_k b_k e^{-iwk} / (1 - sum_i a_i e^{-iwi}), w in radians per beat, evaluated over
    ``band`` (Hz converted with the mean beat period). Unlike the Welch transfer function
    (Pxy / Pxx), this keeps the physiological ordering, so the RRI -> SBP feed-forward path does
    not leak into it. Returns mean gain, the gain-weighted mean of Re(H)/|H| and the DC gain.
    """
    arc = res.ar if n_ar is None else res.ar[:n_ar]
    if not np.all(np.isfinite(res.coefs)) or not np.all(np.isfinite(arc)) or not np.isfinite(beat_sec):
        return {"gain": np.nan, "re_sign": np.nan, "dc_gain": np.nan}
    f = np.linspace(band[0], band[1], n_freq)
    w = 2 * np.pi * f * beat_sec
    B = sum(b * np.exp(-1j * w * k) for b, k in zip(res.coefs, lags))
    A = 1 - sum(a * np.exp(-1j * w * (i + 1)) for i, a in enumerate(arc))
    H = B / A
    g = np.abs(H)
    re = np.real(H) / np.where(g > 0, g, 1.0)
    dc = float(np.sum(res.coefs) / (1 - np.sum(arc))) if abs(1 - np.sum(arc)) > 1e-6 else np.nan
    return {"gain": float(g.mean()), "re_sign": float(np.sum(g * re) / np.sum(g)) if g.sum() > 0 else np.nan,
            "dc_gain": dc}


# ---------------------------------------------------------------------------
# Resampling and spectra
# ---------------------------------------------------------------------------


def resample_uniform(t: np.ndarray, v: np.ndarray, valid: np.ndarray, fs: float, t0: float, t1: float,
                     max_gap: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of valid beat values onto a uniform grid; also returns a mask that is
    False where the nearest valid beat is more than ``max_gap`` seconds away."""
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(t) & np.isfinite(v)
    grid = np.arange(t0, t1, 1.0 / fs)
    if ok.sum() < 3:
        return np.full(len(grid), np.nan), np.zeros(len(grid), dtype=bool)
    tt, vv = t[ok], v[ok]
    order = np.argsort(tt)
    tt, vv = tt[order], vv[order]
    out = np.interp(grid, tt, vv)
    j = np.clip(np.searchsorted(tt, grid), 1, len(tt) - 1)
    near = np.minimum(np.abs(grid - tt[j - 1]), np.abs(tt[j] - grid))
    mask = near <= max_gap
    return out, mask


@dataclass
class BandTF:
    gain: float  # mean |H| over bins with coherence >= coh_thr (NaN if none)
    gain_all: float  # mean |H| over the band
    coh_mean: float
    coh_max: float
    re_sign: float  # coherence-weighted mean of Re(H)/|H| in [-1, 1]
    phase_at_max: float  # radians
    n_bins: int


def cross_spectrum(x: np.ndarray, y: np.ndarray, fs: float, nperseg: int):
    """Welch auto- and cross-spectra with linear detrending (x input, y output). H = Pxy / Pxx."""
    nper = min(nperseg, len(x))
    f, pxx = welch(x, fs=fs, nperseg=nper, detrend="linear")
    _, pyy = welch(y, fs=fs, nperseg=nper, detrend="linear")
    _, pxy = csd(x, y, fs=fs, nperseg=nper, detrend="linear")
    return f, pxx, pyy, pxy


def band_tf(f, pxx, pyy, pxy, band: tuple[float, float], coh_thr: float = 0.5) -> BandTF:
    m = (f >= band[0]) & (f < band[1])
    if not m.any():
        return BandTF(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        H = pxy[m] / pxx[m]
        coh = np.abs(pxy[m]) ** 2 / (pxx[m] * pyy[m])
    coh = np.nan_to_num(coh)
    g = np.abs(H)
    hi = coh >= coh_thr
    re_unit = np.real(H) / np.where(g > 0, g, 1.0)
    re_sign = float(np.sum(coh * re_unit) / np.sum(coh)) if np.sum(coh) > 0 else np.nan
    i = int(np.argmax(coh))
    return BandTF(float(g[hi].mean()) if hi.any() else np.nan, float(g.mean()), float(coh.mean()),
                  float(coh.max()), re_sign, float(np.angle(H[i])), int(m.sum()))


def transfer_function(x: np.ndarray, y: np.ndarray, fs: float = 4.0, nperseg: int = 256,
                      bands: dict[str, tuple[float, float]] | None = None, coh_thr: float = 0.5) -> dict[str, BandTF]:
    """Band transfer-function summaries for input x -> output y (uniformly sampled)."""
    bands = bands or {"LF": LF_BAND, "HF": HF_BAND}
    f, pxx, pyy, pxy = cross_spectrum(x, y, fs, nperseg)
    return {k: band_tf(f, pxx, pyy, pxy, b, coh_thr) for k, b in bands.items()}


def band_sign_spectrum(hr: np.ndarray, mp: np.ndarray, fs: float, nperseg: int,
                       bands: dict[str, tuple[float, float]] | None = None) -> dict[str, dict[str, float]]:
    """Per band: coherence-weighted sign of Re(S_map,hr) (negative = reflex-like), mean coherence."""
    bands = bands or SIGN_BANDS
    f, pxx, pyy, pxy = cross_spectrum(mp, hr, fs, nperseg)
    out = {}
    for k, b in bands.items():
        r = band_tf(f, pxx, pyy, pxy, b)
        out[k] = {"re_sign": r.re_sign, "coh_mean": r.coh_mean, "gain_all": r.gain_all, "n_bins": r.n_bins}
    return out


# ---------------------------------------------------------------------------
# Timescale sweep
# ---------------------------------------------------------------------------


def bin_means(t: np.ndarray, v: np.ndarray, valid: np.ndarray, t0: float, t1: float, step: float,
              min_frac: float = 0.5, beat_sec: float | None = None) -> np.ndarray:
    """Mean of valid beat values in consecutive ``step``-second bins; NaN if coverage < min_frac."""
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(t) & np.isfinite(v) & (t >= t0) & (t < t1)
    nb = int(np.floor((t1 - t0) / step))
    if nb <= 0:
        return np.empty(0)
    idx = np.floor((t[ok] - t0) / step).astype(np.int64)
    keep = idx < nb
    idx, vals = idx[keep], v[ok][keep]
    s = np.bincount(idx, weights=vals, minlength=nb)
    c = np.bincount(idx, minlength=nb).astype(float)
    if beat_sec is None:
        tv = np.sort(t[ok])
        beat_sec = float(np.median(np.diff(tv))) if len(tv) > 2 else 1.0
    need = max(1.0, min_frac * step / beat_sec)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / c
    m[c < need] = np.nan
    return m


def lagged_timescale_fit(blocks: list[tuple[np.ndarray, np.ndarray]], *, n_lags: int = 1,
                         include_lag0: bool = False) -> RegressionResult:
    """
    HR_t = c_block + sum_{i=1..p} a_i HR_{t-i} + sum_{j in lags} b_j MAP_{t-j} + e, pooled over
    blocks (block intercepts by within-block demeaning). Returns the summed MAP coefficient.
    """
    lags = tuple(range(0 if include_lag0 else 1, n_lags + 1))
    Ys, Xs = [], []
    for hr, mp in blocks:
        hr = np.asarray(hr, dtype=np.float64)
        mp = np.asarray(mp, dtype=np.float64)
        if len(hr) <= n_lags + 2:
            continue
        Y, X = _lag_design(hr, mp, n_lags, lags)
        if len(Y) < 3:
            continue
        Ys.append(Y - Y.mean())
        Xs.append(X - X.mean(axis=0))
    if not Ys:
        return RegressionResult(np.full(len(lags), np.nan), np.nan, np.nan, 0)
    return ols_total(np.concatenate(Ys), np.concatenate(Xs), len(lags))


def contemporaneous_corr(blocks: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Within-block (demeaned) correlation of HR and MAP at the same step."""
    a, b = [], []
    for hr, mp in blocks:
        ok = np.isfinite(hr) & np.isfinite(mp)
        if ok.sum() < 3:
            continue
        a.append(hr[ok] - hr[ok].mean())
        b.append(mp[ok] - mp[ok].mean())
    if not a:
        return np.nan
    a, b = np.concatenate(a), np.concatenate(b)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# RSA, PAT, respiratory pressure variation
# ---------------------------------------------------------------------------


def rsa_coupling(resp_u: np.ndarray, rri_u: np.ndarray, fs: float, resp_freq: float,
                 nperseg: int = 256, half_width: float = 0.03) -> BandTF:
    """Respiration -> RRI transfer function in a band around the respiratory frequency."""
    if not np.isfinite(resp_freq):
        return BandTF(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    f, pxx, pyy, pxy = cross_spectrum(resp_u, rri_u, fs, nperseg)
    return band_tf(f, pxx, pyy, pxy, (max(resp_freq - half_width, 0.05), resp_freq + half_width))


def rsa_amplitude(rri: np.ndarray, phase: np.ndarray, valid: np.ndarray, n_bins: int = 8) -> tuple[float, float]:
    """Peak-to-trough of the mean detrended RRI across respiratory-phase bins, and the phase (rad)
    of the RRI maximum."""
    y = running_mean_detrend(rri, valid, 15)
    ok = np.isfinite(y) & np.isfinite(phase)
    if ok.sum() < 4 * n_bins:
        return np.nan, np.nan
    b = np.floor((np.asarray(phase)[ok] + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    means = np.array([y[ok][b == i].mean() if np.any(b == i) else np.nan for i in range(n_bins)])
    if np.isnan(means).any():
        return np.nan, np.nan
    centres = -np.pi + (np.arange(n_bins) + 0.5) * 2 * np.pi / n_bins
    return float(means.max() - means.min()), float(centres[int(np.argmax(means))])


def pat_coupling(pat: np.ndarray, sbp: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """Median PAT and the beat-level (31-beat detrended) PAT-SBP slope and correlation."""
    ok = np.asarray(valid, dtype=bool) & np.isfinite(pat) & np.isfinite(sbp)
    if ok.sum() < 30:
        return {"pat_median": np.nan, "pat_sbp_slope": np.nan, "pat_sbp_corr": np.nan, "n_pat": int(ok.sum())}
    a = running_mean_detrend(pat, ok, 31)
    b = running_mean_detrend(sbp, ok, 31)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30 or a[m].std() == 0 or b[m].std() == 0:
        return {"pat_median": float(np.nanmedian(pat[ok])), "pat_sbp_slope": np.nan, "pat_sbp_corr": np.nan,
                "n_pat": int(ok.sum())}
    return {"pat_median": float(np.nanmedian(pat[ok])), "pat_sbp_slope": float(np.polyfit(b[m], a[m], 1)[0]),
            "pat_sbp_corr": float(np.corrcoef(a[m], b[m])[0, 1]), "n_pat": int(ok.sum())}


def respiratory_pressure_variation(sbp: np.ndarray, pp: np.ndarray, phase: np.ndarray,
                                   valid: np.ndarray, min_beats: int = 3) -> dict[str, float]:
    """Median over breaths of SPV (max-min SBP, mmHg) and PPV (100 (PPmax-PPmin)/mean, %).
    Breaths are delimited by respiratory-phase wraps (pi -> -pi)."""
    ph = np.asarray(phase, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(ph) & np.isfinite(sbp) & np.isfinite(pp)
    if ok.sum() < 20:
        return {"spv": np.nan, "ppv": np.nan, "n_breaths": 0}
    wraps = np.where(np.diff(np.unwrap(np.where(np.isfinite(ph), ph, 0.0)) // (2 * np.pi)) > 0)[0] + 1
    edges = np.concatenate([[0], wraps, [len(ph)]])
    spv, ppv = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = ok[a:b]
        if m.sum() < min_beats or m.sum() < (b - a) * 0.7:
            continue
        s = np.asarray(sbp)[a:b][m]
        p = np.asarray(pp)[a:b][m]
        spv.append(s.max() - s.min())
        ppv.append(100.0 * (p.max() - p.min()) / ((p.max() + p.min()) / 2.0))
    if not spv:
        return {"spv": np.nan, "ppv": np.nan, "n_breaths": 0}
    return {"spv": float(np.median(spv)), "ppv": float(np.median(ppv)), "n_breaths": len(spv)}


# ---------------------------------------------------------------------------
# Epoch analysis with surrogate nulls
# ---------------------------------------------------------------------------


def _uniform_pair(beats: pd.DataFrame, t0: float, t1: float, fs: float):
    v = beats["valid"].to_numpy(bool)
    tr = beats["t_r"].to_numpy(float)
    rri = beats["rri"].to_numpy(float)
    t_end = tr + rri / 1000.0  # RRI(n) is known at R(n+1)
    t_sys = beats["t_sys"].to_numpy(float)
    ts = np.where(np.isfinite(t_sys), t_sys, tr + 0.3)
    sb, ms = resample_uniform(ts, beats["sbp"].to_numpy(float), v, fs, t0, t1)
    rr, mr = resample_uniform(t_end, rri, v, fs, t0, t1)
    return sb, rr, ms & mr


def analyse_epoch(beats: pd.DataFrame, t0: float, t1: float, *, n_surr: int = 99,
                  rng: np.random.Generator | None = None, fs_u: float = 4.0, nperseg: int = 256,
                  resp_u: np.ndarray | None = None, resp_freq: float = np.nan) -> dict[str, float]:
    """
    All fast-coupling statistics for one epoch (beats restricted to [t0, t1)), with IAAFT nulls.

    Surrogates replace the SBP beat series (and the resampled SBP series for the transfer function)
    by IAAFT surrogates, keeping RRI intact; each statistic's two-sided surrogate p-value is
    (1 + #{|null| >= |observed|}) / (n_surr + 1), centred on the null median.
    """
    rng = rng or np.random.default_rng(0)
    b = beats[(beats["t_r"] >= t0) & (beats["t_r"] < t1)]
    v = b["valid"].to_numpy(bool)
    sbp = b["sbp"].to_numpy(float)
    rri = b["rri"].to_numpy(float)
    out: dict[str, float] = {"n_beats": len(b), "n_valid": int(v.sum()),
                             "valid_frac": float(v.mean()) if len(v) else np.nan}
    if v.sum() < 60:
        return out
    out.update({"rri_mean": float(np.nanmean(rri[v])), "sbp_mean": float(np.nanmean(sbp[v])),
                "map_mean": float(np.nanmean(b["map_beat"].to_numpy(float)[v])),
                "hr_mean": float(60000.0 / np.nanmean(rri[v])),
                "sdnn": float(np.nanstd(rri[v])), "sbp_sd": float(np.nanstd(sbp[v]))})

    # sequence method, lags 0..2
    for lag in (0, 1, 2):
        s = sequence_method(sbp, rri, v, lag)
        out.update({f"seq{lag}_nconc": s.n_conc, f"seq{lag}_ndisc": s.n_disc, f"seq{lag}_brs": s.brs,
                    f"seq{lag}_bei": s.bei, f"seq{lag}_conc": s.concordance, f"seq{lag}_nramps": s.n_ramps})
    nc = sum(out[f"seq{k}_nconc"] for k in (0, 1, 2))
    nd = sum(out[f"seq{k}_ndisc"] for k in (0, 1, 2))
    out["seq_pooled_conc"] = (nc - nd) / (nc + nd) if (nc + nd) else np.nan
    slopes = np.concatenate([sequence_method(sbp, rri, v, k).slopes for k in (0, 1, 2)])
    out["seq_pooled_brs"] = float(np.mean(slopes)) if len(slopes) else np.nan

    # beat regression (detrended, AR(2), SBP lags 0..2) and a MAP-harmonised version (lags 1..3)
    r = beat_regression(sbp, rri, v)
    out.update({"reg_brs": r.total, "reg_se": r.se_total, "reg_n": r.n,
                "reg_b0": r.coefs[0], "reg_b1": r.coefs[1], "reg_b2": r.coefs[2]})
    ct = causal_transfer(r, (0, 1, 2), out["rri_mean"] / 1000.0)
    out.update({"arx_LF_gain": ct["gain"], "arx_LF_resign": ct["re_sign"], "arx_dc_gain": ct["dc_gain"]})
    r0 = beat_regression(sbp, rri, v, lags=(0,))
    out["reg_lag0_brs"] = r0.total
    rf = beat_regression(sbp, rri, v, lags=(1, 2, 3))
    out["reg_lag123_brs"] = rf.total  # excludes the within-beat SBP(n) -> RRI(n) term
    # time available for a within-beat reflex: systolic peak -> next R-peak
    t_sys_e = b["t_sys"].to_numpy(float)
    gap = b["t_r"].to_numpy(float) + rri / 1000.0 - t_sys_e
    out["sys_to_next_r"] = float(np.nanmedian(gap[v])) if np.isfinite(gap[v]).any() else np.nan
    # respiration-adjusted ARX (respiratory-phase harmonics at the current beat as covariates)
    ph_e = b["resp_phase"].to_numpy(float)
    if np.isfinite(ph_e[v]).mean() > 0.8:
        ra = beat_regression(sbp, rri, v, covariates=resp_harmonics(ph_e))
        out.update({"reg_resp_brs": ra.total, "reg_resp_b0": ra.coefs[0]})
        cr = causal_transfer(ra, (0, 1, 2), out["rri_mean"] / 1000.0, n_ar=2)
        out.update({"arx_resp_LF_resign": cr["re_sign"], "arx_resp_LF_gain": cr["gain"]})
        ra0 = beat_regression(sbp, rri, v, lags=(0,), covariates=resp_harmonics(ph_e))
        out["reg_resp_lag0_brs"] = ra0.total
    rr_ff = beat_regression(rri, sbp, v, lags=(1, 2), ar=2)
    out["ff_rri_to_sbp"] = rr_ff.total  # feed-forward RRI(n-1, n-2) -> SBP(n), mmHg/ms
    hr = 60000.0 / rri
    mp = b["map_beat"].to_numpy(float)
    rh = beat_regression(mp, hr, v, lags=(1, 2, 3))
    out["reg_map_hr"] = rh.total  # bpm / mmHg, should be negative

    # transfer function at 4 Hz
    sb_u, rr_u, mask = _uniform_pair(b, t0, t1, fs_u)
    out["uniform_ok_frac"] = float(mask.mean())
    tf = None
    if mask.mean() >= 0.8:
        tf = transfer_function(sb_u, rr_u, fs_u, nperseg)
        for k, bt in tf.items():
            out.update({f"tf_{k}_gain": bt.gain, f"tf_{k}_gain_all": bt.gain_all, f"tf_{k}_coh": bt.coh_mean,
                        f"tf_{k}_cohmax": bt.coh_max, f"tf_{k}_resign": bt.re_sign, f"tf_{k}_phase": bt.phase_at_max})
        if resp_u is not None and len(resp_u) == len(rr_u) and np.isfinite(resp_freq):
            rs = rsa_coupling(resp_u, rr_u, fs_u, resp_freq, nperseg)
            out.update({"rsa_gain_all": rs.gain_all, "rsa_coh": rs.coh_mean, "rsa_cohmax": rs.coh_max,
                        "rsa_phase": rs.phase_at_max})
            out["resp_freq"] = resp_freq

    # PAT and respiratory pressure variation
    out.update(pat_coupling(b["pat"].to_numpy(float), sbp, v))
    ph = b["resp_phase"].to_numpy(float)
    if np.isfinite(ph).mean() > 0.8:
        amp, phmax = rsa_amplitude(rri, ph, v)
        out.update({"rsa_amp": amp, "rsa_phase_max": phmax})
        out.update(respiratory_pressure_variation(sbp, b["pp"].to_numpy(float), ph, v))

    # pooled lags 0-2 concordance without the r criterion (like-for-like with the surrogate null)
    out["seq_conc_raw"] = float(seq_concordance_batch(sbp[None, :], rri, v)[0])

    # ---- surrogate nulls (vectorised over surrogates) ----
    if n_surr > 0:
        surr = iaaft(_fill_invalid(sbp, v), n_surr, rng)
        null_conc = seq_concordance_batch(surr, rri, v)
        out.update(_pval("p_seq", out["seq_conc_raw"], null_conc))
        out.update(_pval("p_reg", out["reg_brs"], beat_regression_batch(surr, rri, v)))
        if tf is not None:
            su = iaaft(sb_u, n_surr, rng)
            coh_n, sgn_n = tf_band_batch(su, rr_u, fs_u, nperseg, LF_BAND)
            obs = tf["LF"]
            out.update(_pval("p_tf_coh", obs.coh_mean, coh_n, one_sided=True))
            out.update(_pval("p_tf_signed", obs.re_sign * obs.coh_mean, sgn_n))
    return out


def _count_runs_batch(M: np.ndarray, L: int) -> np.ndarray:
    """Per row, the number of maximal runs of True of length >= L."""
    m, n = M.shape
    if n < L:
        return np.zeros(m, dtype=np.int64)
    W = M[:, : n - L + 1].copy()
    for k in range(1, L):
        W &= M[:, k: n - L + 1 + k]
    prev = np.concatenate([np.zeros((m, 1), dtype=bool), M[:, : n - L]], axis=1)
    return (W & ~prev).sum(axis=1)


def seq_concordance_batch(S: np.ndarray, rri: np.ndarray, v: np.ndarray, *, lags=(0, 1, 2),
                          min_len: int = 3, sbp_thr: float = 1.0, rri_thr: float = 5.0) -> np.ndarray:
    """Pooled (lags 0-2) concordance index for each row of ``S`` (SBP series), without the r criterion."""
    S = np.atleast_2d(S)
    n = S.shape[1]
    need = min_len - 1
    nc = np.zeros(S.shape[0])
    nd = np.zeros(S.shape[0])
    for lag in lags:
        x = S[:, : n - lag]
        y = rri[lag:]
        vv = v[: n - lag] & v[lag:] & np.isfinite(y)
        dx = np.diff(x, axis=1)
        dy = np.diff(y)
        vs = (vv[:-1] & vv[1:])[None, :]
        up = vs & (dx >= sbp_thr)
        dn = vs & (dx <= -sbp_thr)
        yu = (dy >= rri_thr)[None, :]
        yd = (dy <= -rri_thr)[None, :]
        nc += _count_runs_batch(up & yu, need) + _count_runs_batch(dn & yd, need)
        nd += _count_runs_batch(up & yd, need) + _count_runs_batch(dn & yu, need)
    tot = nc + nd
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, (nc - nd) / tot, np.nan)


def _moving_sum_rows(X: np.ndarray, k: int) -> np.ndarray:
    """Centred moving sum of width k along axis 1 ('same' length, zero padded)."""
    m, n = X.shape
    h = k // 2
    P = np.concatenate([np.zeros((m, h + 1)), X, np.zeros((m, h))], axis=1)
    c = np.cumsum(P, axis=1)
    return c[:, k:k + n] - c[:, :n]


def beat_regression_batch(S: np.ndarray, rri: np.ndarray, v: np.ndarray, *, lags=(0, 1, 2), ar: int = 2,
                          detrend: int = 31) -> np.ndarray:
    """:func:`beat_regression` summed coefficient for each row of ``S`` (same RRI, same validity),
    via Frisch-Waugh (the RRI lags and intercept are partialled out once)."""
    S = np.atleast_2d(S)
    y = running_mean_detrend(rri, v, detrend)
    vv = np.asarray(v, dtype=bool) & np.isfinite(rri)
    cnt = np.convolve(vv.astype(float), np.ones(detrend), mode="same")
    Sv = np.where(vv[None, :], S, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Xd = np.where((vv & (cnt >= detrend // 2))[None, :], S - _moving_sum_rows(Sv, detrend) / cnt[None, :], np.nan)
    n = len(y)
    L = max(max(lags), ar)
    rows = np.arange(L, n)
    Z = np.column_stack([np.ones(len(rows))] + [y[rows - i] for i in range(1, ar + 1)])
    Yr = y[rows]
    Xl = np.stack([Xd[:, rows - k] for k in lags], axis=2)  # (m, rows, lags)
    ok = np.isfinite(Yr) & np.all(np.isfinite(Z), axis=1) & np.all(np.isfinite(Xl[0]), axis=1)
    if ok.sum() < Z.shape[1] + len(lags) + 10:
        return np.full(S.shape[0], np.nan)
    Z, Yr, Xl = Z[ok], Yr[ok], Xl[:, ok, :]
    Q, _ = np.linalg.qr(Z)
    yres = Yr - Q @ (Q.T @ Yr)
    Xres = Xl - np.einsum("ij,mjk->mik", Q, np.einsum("ji,mjk->mik", Q, Xl))
    XtX = np.einsum("mij,mik->mjk", Xres, Xres)
    Xty = np.einsum("mij,i->mj", Xres, yres)
    try:
        beta = np.linalg.solve(XtX, Xty[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.full(S.shape[0], np.nan)
    return beta.sum(axis=1)


def tf_band_batch(S_u: np.ndarray, y_u: np.ndarray, fs: float, nperseg: int, band: tuple[float, float],
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Mean band coherence and (coherence-weighted Re-sign x coherence) for each input row."""
    nper = min(nperseg, S_u.shape[1])
    f, pxx = welch(S_u, fs=fs, nperseg=nper, detrend="linear", axis=-1)
    _, pyy = welch(y_u, fs=fs, nperseg=nper, detrend="linear")
    _, pxy = csd(S_u, np.broadcast_to(y_u, S_u.shape), fs=fs, nperseg=nper, detrend="linear", axis=-1)
    m = (f >= band[0]) & (f < band[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        H = pxy[:, m] / pxx[:, m]
        coh = np.nan_to_num(np.abs(pxy[:, m]) ** 2 / (pxx[:, m] * pyy[None, m]))
        g = np.abs(H)
        re_unit = np.real(H) / np.where(g > 0, g, 1.0)
        sgn = np.sum(coh * re_unit, axis=1) / np.sum(coh, axis=1)
    return coh.mean(axis=1), sgn * coh.mean(axis=1)


def _pval(name: str, obs: float, null: np.ndarray, one_sided: bool = False) -> dict[str, float]:
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or len(null) == 0:
        return {name: np.nan, name + "_z": np.nan}
    med = np.median(null)
    sd = null.std()
    if one_sided:
        p = (1 + np.sum(null >= obs)) / (len(null) + 1)
    else:
        p = (1 + np.sum(np.abs(null - med) >= abs(obs - med))) / (len(null) + 1)
    return {name: float(p), name + "_z": float((obs - null.mean()) / sd) if sd > 0 else np.nan}


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaN-preserving)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def window_sequence_evidence(beats: pd.DataFrame, t0: float, t1: float, win: float = 30.0) -> pd.DataFrame:
    """Per-``win`` second window: sequence counts, concordance and BRS (lags 0-2 pooled), with the
    beat index range as an evidence pointer."""
    rows = []
    tr = beats["t_r"].to_numpy(float)
    for a in np.arange(t0, t1 - win + 1e-9, win):
        m = (tr >= a) & (tr < a + win)
        if m.sum() < 10:
            continue
        b = beats[m]
        v = b["valid"].to_numpy(bool)
        sbp = b["sbp"].to_numpy(float)
        rri = b["rri"].to_numpy(float)
        res = [sequence_method(sbp, rri, v, k) for k in (0, 1, 2)]
        nc = sum(r.n_conc for r in res)
        nd = sum(r.n_disc for r in res)
        sl = np.concatenate([r.slopes for r in res])
        idx = np.where(m)[0]
        rows.append({"t_start": a, "t_end": a + win, "beat_first": int(b.index[0]), "beat_last": int(b.index[-1]),
                     "n_valid": int(v.sum()), "n_conc": nc, "n_disc": nd,
                     "brs": float(np.mean(sl)) if len(sl) else np.nan,
                     "hr": float(60000.0 / np.nanmean(rri[v])) if v.any() else np.nan,
                     "sbp": float(np.nanmean(sbp[v])) if v.any() else np.nan})
        del idx
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Beat-level closed-loop simulator (ground truth for tests and calibration)
# ---------------------------------------------------------------------------


@dataclass
class BeatLoopSpec:
    """
    Beat-to-beat closed loop (DeBoer-style), deviations from baseline:

    SBP(n) = a_s SBP(n-1) + c_ff RRI(n-1) + mayer(n) + k_s D(n) + r_s resp(n) + e_s
    RRI(n) = a_r RRI(n-1) + brs SBP(n - lag) - k_r D(n) + r_r resp(n) + e_r

    ``D`` is a slow common driver (AR(1), time constant ``driver_tau`` seconds) that raises SBP and
    shortens RRI together (e.g. nociception), so it induces the *opposite* co-movement to the
    baroreflex. ``mayer`` is a 0.1 Hz resonant AR(2) oscillation in pressure.
    """

    brs: float = 8.0  # ms / mmHg
    lag: int = 0
    a_s: float = 0.3
    a_r: float = 0.4
    c_ff: float = -0.01  # mmHg / ms (longer diastole -> lower next SBP)
    mayer_sd: float = 2.0
    k_s: float = 1.0  # mmHg per unit driver
    k_r: float = 12.0  # ms per unit driver
    driver_sd: float = 0.0  # stationary SD of the driver (0 = no driver)
    driver_tau: float = 300.0
    resp_rate: float = 0.25  # Hz
    r_s: float = 1.5
    r_r: float = 10.0
    e_s: float = 1.5
    e_r: float = 8.0
    sbp0: float = 120.0
    rri0: float = 900.0
    dbp_ratio: float = 0.55


def simulate_beat_loop(n_beats: int, spec: BeatLoopSpec | None = None, seed: int = 0) -> pd.DataFrame:
    """Beat table (t_r, rri, sbp, dbp, map_beat, pp, valid) generated by :class:`BeatLoopSpec`."""
    spec = spec or BeatLoopSpec()
    rng = np.random.default_rng(seed)
    burn = 500
    N = n_beats + burn
    sbp = np.zeros(N)
    rri = np.zeros(N)
    D = np.zeros(N)
    m1 = m2 = 0.0
    t = 0.0
    tr = np.zeros(N)
    # Mayer AR(2) resonance ~0.1 Hz at the beat rate
    beat_sec = spec.rri0 / 1000.0
    w = 2 * np.pi * 0.1 * beat_sec
    rad = 0.93
    p1, p2 = 2 * rad * np.cos(w), -rad * rad
    mayer_innov = spec.mayer_sd * np.sqrt((1 + p2) * ((1 - p2) ** 2 - p1 ** 2) / (1 - p2))
    phi_d = np.exp(-beat_sec / spec.driver_tau) if spec.driver_sd > 0 else 0.0
    d_innov = spec.driver_sd * np.sqrt(1 - phi_d ** 2)
    for n in range(N):
        m = p1 * m1 + p2 * m2 + rng.normal(0, mayer_innov)
        m2, m1 = m1, m
        D[n] = (phi_d * D[n - 1] if n else 0.0) + (rng.normal(0, d_innov) if spec.driver_sd > 0 else 0.0)
        resp = np.sin(2 * np.pi * spec.resp_rate * t)
        sp = (spec.a_s * sbp[n - 1] if n else 0.0) + (spec.c_ff * rri[n - 1] if n else 0.0) + m \
            + spec.k_s * D[n] + spec.r_s * resp + rng.normal(0, spec.e_s)
        sbp[n] = sp
        src = sbp[n - spec.lag] if n - spec.lag >= 0 else 0.0
        rr = (spec.a_r * rri[n - 1] if n else 0.0) + spec.brs * src - spec.k_r * D[n] \
            + spec.r_r * resp + rng.normal(0, spec.e_r)
        rri[n] = rr
        tr[n] = t
        t += (spec.rri0 + rr) / 1000.0
    s = slice(burn, N)
    SBP = spec.sbp0 + sbp[s]
    RRI = np.clip(spec.rri0 + rri[s], 320.0, 1900.0)
    prev_rri = np.concatenate([[spec.rri0], RRI[:-1]])
    DBP = spec.dbp_ratio * SBP - 0.02 * (prev_rri - spec.rri0)
    t_r = np.concatenate([[0.0], np.cumsum(RRI[:-1] / 1000.0)])
    df = pd.DataFrame({"t_r": t_r, "rri": RRI, "sbp": SBP, "dbp": DBP,
                       "map_beat": DBP + (SBP - DBP) / 3.0, "pp": SBP - DBP,
                       "t_sys": t_r + 0.3, "sys_delay": 0.3, "t_ppg_foot": np.nan, "pat": np.nan,
                       "ppg_amp": np.nan, "resp_phase": np.angle(np.exp(1j * 2 * np.pi * spec.resp_rate * t_r)),
                       "driver": D[s], "valid": True})
    return df


def render_beat_waveforms(beats: pd.DataFrame, fs: float = 250.0, *, noise: float = 0.02,
                          wander: float = 0.2, invert: bool = False, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """ECG (mV) and ABP (mmHg) waveforms whose beats reproduce ``beats`` (t_r, sbp, dbp, rri).

    ECG: Gaussian P-QRS-T complexes at each R-peak, baseline wander and white noise.
    ABP: per beat, an exponential diastolic decay to DBP(n) at the foot (R + 80 ms), then a
    systolic upstroke to SBP(n) at R + 300 ms.
    """
    rng = np.random.default_rng(seed)
    tr = beats["t_r"].to_numpy(float)
    rri = beats["rri"].to_numpy(float) / 1000.0
    T = tr[-1] + rri[-1] + 1.0
    n = int(T * fs)
    t = np.arange(n) / fs
    ecg = np.zeros(n)
    for c, a, wdt in ((-0.16, 0.12, 0.025), (-0.03, -0.10, 0.008), (0.0, 1.0, 0.009), (0.03, -0.25, 0.009),
                      (0.25, 0.30, 0.04)):
        for r in tr:
            lo, hi = int((r + c - 5 * wdt) * fs), int((r + c + 5 * wdt) * fs) + 1
            lo, hi = max(lo, 0), min(hi, n)
            tt = t[lo:hi]
            ecg[lo:hi] += a * np.exp(-0.5 * ((tt - r - c) / wdt) ** 2)
    ecg += wander * np.sin(2 * np.pi * 0.2 * t) + rng.normal(0, noise, n)
    if invert:
        ecg = -ecg
    sbp = beats["sbp"].to_numpy(float)
    dbp = beats["dbp"].to_numpy(float)
    abp = np.full(n, np.nan)
    feet = tr + 0.08
    peaks = tr + 0.30
    for i in range(len(tr)):
        a = int(feet[i] * fs)
        p = int(peaks[i] * fs)
        e = int(feet[i + 1] * fs) if i + 1 < len(tr) else min(int((feet[i] + rri[i]) * fs), n)
        if p <= a or e <= p or e > n:
            continue
        # upstroke: raised cosine from DBP(i) to SBP(i)
        u = np.arange(p - a) / (p - a)
        abp[a:p] = dbp[i] + (sbp[i] - dbp[i]) * 0.5 * (1 - np.cos(np.pi * u))
        # decay: exponential from SBP(i) to DBP(i+1) at the next foot
        d_next = dbp[i + 1] if i + 1 < len(tr) else dbp[i]
        L = e - p
        k = np.arange(L) / L
        abp[p:e] = d_next + (sbp[i] - d_next) * (np.exp(-3.0 * k) - np.exp(-3.0)) / (1 - np.exp(-3.0))
    first = int(feet[0] * fs)
    abp[:first] = dbp[0]
    abp = np.where(np.isnan(abp), np.nanmean(abp), abp) + rng.normal(0, 0.3, n)
    return ecg, abp
