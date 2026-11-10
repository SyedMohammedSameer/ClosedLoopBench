# ClosedLoopBench 1.1 (v4): pre-specified predictions (protocol.md §7)

Protocol sha256 now: `418281f18a34c6c9c66111250b0d17d8f9296e8718c469a08515ba386793d10e`; registered: `418281f18a34c6c9c66111250b0d17d8f9296e8718c469a08515ba386793d10e`.

## Q1 (cross-null false admissions <= 1 of 29 per held-out draw, every primary method)

| method | mean FA per draw | max | draws with any |
|---|---|---|---|
| naive | 0.00 | 0 | 0/8 |
| naive_lvl | 0.00 | 0 | 0/8 |
| xcorr_v1 | 0.00 | 0 | 0/8 |
| xcorr_dir | 0.00 | 0 | 0/8 |
| xcorr_pos | 0.00 | 0 | 0/8 |
| granger | 0.00 | 0 | 0/8 |
| varx | 0.00 | 0 | 0/8 |
| pcmciplus | 0.00 | 0 | 0/8 |
| varlingam | 0.00 | 0 | 0/8 |
| matched | 0.00 | 0 | 0/8 |
| kink | 0.00 | 0 | 0/8 |
| seqtrial | 0.00 | 0 | 0/8 |

**Q1 HOLDS**

## Q2 (uncalibrated null discoveries >= 4 of 29 per cross draw: naive, xcorr_pos, granger)

naive 18.25, xcorr_pos 17.62, granger 19.75

**Q2 HOLDS**

## Q3 (share of GAS relations admitted correct > share of DRUG relations admitted correct, >= 7 of 12 methods)

Shares use the full domain sizes (5 GAS, 11 DRUG) as denominators.

| method | GAS correct / wrong (of 5) | DRUG correct / wrong (of 11) | GAS share | DRUG share | GAS > DRUG |
|---|---|---|---|---|---|
| naive | 4 / 0 | 2 / 4 | 0.80 | 0.18 | yes |
| naive_lvl | 3 / 0 | 5 / 2 | 0.60 | 0.45 | yes |
| xcorr_v1 | 2 / 2 | 5 / 0 | 0.40 | 0.45 | no |
| xcorr_dir | 2 / 2 | 6 / 0 | 0.40 | 0.55 | no |
| xcorr_pos | 1 / 3 | 1 / 6 | 0.20 | 0.09 | yes |
| granger | 2 / 3 | 8 / 0 | 0.40 | 0.73 | no |
| varx | 4 / 1 | 4 / 1 | 0.80 | 0.36 | yes |
| pcmciplus | 0 / 0 | 0 / 0 | 0.00 | 0.00 | no |
| varlingam | 5 / 0 | 8 / 0 | 1.00 | 0.73 | yes |
| matched | 4 / 0 | 3 / 0 | 0.80 | 0.27 | yes |
| kink | 4 / 0 | 4 / 0 | 0.80 | 0.36 | yes |
| seqtrial | 4 / 0 | 3 / 0 | 0.80 | 0.27 | yes |

8 of 12 methods. **Q3 HOLDS**

## Q4 (>= 80% of admitted GAS cells have the expected sign, and >= 5 correct cells)

Admitted GAS cells: 35 correct, 11 wrong (76% correct).

**Q4 FAILS**

## Q5 (R15 FiO2 -> SpO2 admitted correct by >= 1 event design, wrong by none)

| method | est [95% CI] | p (BY) | stable | n cases | call |
|---|---|---|---|---|---|
| kink | 0.451 [0.0113, 0.89] | 0.26 | 0.38 | 387 | ns_right |
| matched | 0.196 [-0.24, 0.631] | 1 | 0.00 | 384 | ns_right |
| naive_lvl | 1.04 [0.505, 1.56] | 0.00066 | 0.62 | 510 | ns_right |
| seqtrial | 0.536 [0.0336, 1.04] | 0.2 | 0.38 | 485 | ns_right |

**Q5 FAILS**

## Q6 (>= 1 primary method admits a DRUG relation with the wrong sign)

Wrong-sign drug admissions: naive R04 (propofol Ct down -> MAP up) 1.19 [0.841, 1.54], naive R06 (remi Ct up -> MAP down) 0.575 [0.354, 0.797], naive R07 (remi Ct down -> HR up) 0.877 [0.745, 1.01], naive R08 (remi Ct down -> MAP up) 3.36 [3.03, 3.69], naive_lvl R07 (remi Ct down -> HR up) 0.731 [0.601, 0.861], naive_lvl R08 (remi Ct down -> MAP up) 0.815 [0.512, 1.12], varx R07 (remi Ct down -> HR up) 0.222 [0.123, 0.321], xcorr_pos R03 (propofol Ct up -> MAP down) 0.0223 [0.0121, 0.0325], xcorr_pos R04 (propofol Ct down -> MAP up) 0.0182 [0.01, 0.0265], xcorr_pos R05 (remi Ct up -> HR down) 0.0525 [0.0423, 0.0628], xcorr_pos R06 (remi Ct up -> MAP down) 0.0662 [0.0565, 0.0758], xcorr_pos R07 (remi Ct down -> HR up) 0.0569 [0.0466, 0.0672], xcorr_pos R08 (remi Ct down -> MAP up) 0.0599 [0.0503, 0.0695]

**Q6 HOLDS**

## Summary

Q1 holds; Q2 holds; Q3 holds; Q4 FAILS; Q5 FAILS; Q6 holds.
