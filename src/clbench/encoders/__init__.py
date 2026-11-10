"""
CLBench signal encoders.

Per-modality encoders
---------------------
ECGEncoder   -- 3-lead ECG  (II, V, aVR)
PPGEncoder   -- PPG / Pleth
ABPEncoder   -- systemic arterial blood pressure
RESPEncoder  -- thoracic impedance respiration

Base class
----------
SignalEncoder  -- 1D Conv stem + Transformer (shared architecture)

Pre-training
------------
MaskedSignalEncoder  -- MAE-style SSL wrapper
WindowDataset        -- loads data/processed/ parquets
save_encoder         -- write checkpoint
load_encoder         -- restore from checkpoint

Go/no-go validation
--------------------
run_linear_probe     -- freeze encoder, train linear head, check AUROC gate
LinearProbeResult    -- result dataclass
ProbeConfig          -- probe hyperparameters

Feature caching
---------------
cache_encoder_features  -- run frozen encoder over all patients, save to disk
load_cached_features    -- load cached (N_windows, d_model) tensors
list_cached_patients    -- list patients with cached features
"""

from clbench.encoders.abp import ABPEncoder
from clbench.encoders.base import ConvStem, SignalEncoder, SinusoidalPosEnc
from clbench.encoders.cache import (
    cache_encoder_features,
    list_cached_patients,
    load_cached_features,
)
from clbench.encoders.ecg import ECGEncoder
from clbench.encoders.linear_probe import (
    LinearProbeResult,
    ProbeConfig,
    run_linear_probe,
)
from clbench.encoders.ppg import PPGEncoder
from clbench.encoders.pretraining import (
    MaskedSignalEncoder,
    WindowDataset,
    evaluate,
    load_encoder,
    save_encoder,
    train_one_epoch,
)
from clbench.encoders.resp import RESPEncoder

__all__ = [
    # Base
    "SignalEncoder",
    "ConvStem",
    "SinusoidalPosEnc",
    # Per-modality
    "ECGEncoder",
    "PPGEncoder",
    "ABPEncoder",
    "RESPEncoder",
    # Pre-training
    "MaskedSignalEncoder",
    "WindowDataset",
    "train_one_epoch",
    "evaluate",
    "save_encoder",
    "load_encoder",
    # Go/no-go
    "run_linear_probe",
    "LinearProbeResult",
    "ProbeConfig",
    # Feature caching
    "cache_encoder_features",
    "load_cached_features",
    "list_cached_patients",
]
