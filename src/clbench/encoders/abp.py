"""ABP (systemic arterial blood pressure) signal encoder — single channel."""

from __future__ import annotations

from clbench.encoders.base import D_MODEL, DROPOUT, FF_DIM, N_HEADS, N_LAYERS, SignalEncoder

ABP_IN_CHANNELS: int = 1


class ABPEncoder(SignalEncoder):  # type: ignore[misc]
    """
    Signal encoder for systemic ABP (ABP/ART channel).

    Input : (B, 1, T)
    Output: (B, d_model)

    Two ABP-specific architectural choices, both Phase 2.1 fix-ups:

    * **No bandpass** at preprocess time — the DC component encodes mean
      arterial pressure, which is the quantity hypotension and hypertension
      detectors need.  See :mod:`clbench.data.preprocess`.
    * **GroupNorm**, not BatchNorm, inside the conv stem.  BatchNorm subtracts
      the per-channel batch mean from feature maps, which makes the encoder
      *invariant to additive offsets in the input* — exactly the property we
      do NOT want for ABP, where the absolute pressure level is the signal.
      GroupNorm normalises within each sample's channels, so per-patient
      MAP differences survive.

    Together with the population-scale ``(x − 85) / 20`` normalisation in
    preprocessing, these let the linear probe recover hypotension /
    hypertension concepts.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
        norm_type: str = "group",
    ) -> None:
        super().__init__(
            in_channels=ABP_IN_CHANNELS,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            norm_type=norm_type,
        )
