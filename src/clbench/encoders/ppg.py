"""PPG (photoplethysmography) signal encoder — single channel (Pleth)."""

from __future__ import annotations

from clbench.encoders.base import D_MODEL, DROPOUT, FF_DIM, N_HEADS, N_LAYERS, SignalEncoder

PPG_IN_CHANNELS: int = 1


class PPGEncoder(SignalEncoder):  # type: ignore[misc]
    """
    Signal encoder for PPG (Pleth channel).

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
            in_channels=PPG_IN_CHANNELS,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
