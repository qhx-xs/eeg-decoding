"""EDF preprocessing and grouped EEG classification utilities."""

from .preprocessing import DEFAULT_BANDS, EEG_CHANNELS, build_trials, extract_log_bandpower
from .splits import build_cv_splits, make_trial_fold_assignment

__all__ = [
    "DEFAULT_BANDS",
    "EEG_CHANNELS",
    "build_trials",
    "extract_log_bandpower",
    "build_cv_splits",
    "make_trial_fold_assignment",
]
