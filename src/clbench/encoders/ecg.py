"""ECG signal encoder (3-lead: II, V, aVR)."""

from __future__ import annotations

from clbench.encoders.base import D_MODEL, DROPOUT, FF_DIM, N_HEADS, N_LAYERS, SignalEncoder

ECG_IN_CHANNELS: int = 3  # II, V, aVR (see preprocess.ECG_MAX_LEADS)


class ECGEncoder(SignalEncoder):  # type: ignore[misc]
    """
    Signal encoder for 3-lead ECG.

    Input : (B, 3, T)  — leads [II, V, aVR]
    Output: (B, d_model)

    Differs from PPG/ABP/RESP encoders in two respects, both motivated by
    Phase 2.1 diagnostics in which the original (mean-pool, token-feature
    MAE) ECG encoder produced AUROC ≈ 0.5 on tachycardia / bradycardia
    while PPG achieved 0.97 on the same kind of probe:

    * ``use_cls_token=True`` — gives the transformer an explicit aggregation
      slot for rhythm-level features that mean-pooling averaged away.
    * Paired with signal-patch MAE (set ``target='signal'`` in
      :class:`MaskedSignalEncoder` at pretraining time) so the encoder must
      reproduce the raw QRS-bearing waveform, not just its own token
      features.  The MSE on a QRS spike only goes down if the encoder
      encodes *where* spikes are, which is the same information as rate.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__(
            in_channels=ECG_IN_CHANNELS,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            use_cls_token=use_cls_token,
        )
