"""RESP (respiration) signal encoder — single channel."""

from __future__ import annotations

from clbench.encoders.base import D_MODEL, DROPOUT, FF_DIM, N_HEADS, N_LAYERS, SignalEncoder

RESP_IN_CHANNELS: int = 1


class RESPEncoder(SignalEncoder):  # type: ignore[misc]
    """
    Signal encoder for thoracic impedance respiration (Resp channel,
    0.1-1 Hz bandpass filtered, z-scored).

    Input : (B, 1, T)
    Output: (B, d_model)
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__(
            in_channels=RESP_IN_CHANNELS,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
