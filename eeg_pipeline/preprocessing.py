from __future__ import annotations

import json
import math
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

from inspect_edf import annotation_bytes, parse_tal, read_digital_signals, read_header


EEG_CHANNELS = ("O1", "Oz", "O2", "Pz", "C3", "C4", "Cz", "Fz")
DEFAULT_BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("low_beta", 13.0, 20.0),
    ("high_beta", 20.0, 30.0),
    ("low_gamma", 30.0, 45.0),
)
TRIGGER_RE = re.compile(r"^Trigger#(\d+)$", re.IGNORECASE)
TRIAL_CODES = {20, 30, 41, 42, 51, 52, 60, 70}
FOUR_CLASS_BY_CODE = {41: 0, 42: 1, 51: 2, 52: 3}
OBJECT_BY_CODE = {41: 0, 42: 1, 51: 0, 52: 1}
PHASE_BY_CODE = {41: 0, 42: 0, 51: 1, 52: 1}


class TrialValidationError(ValueError):
    """Raised when an EDF trigger sequence cannot be interpreted safely."""


def trigger_code(annotation: dict[str, Any]) -> int | None:
    code = annotation.get("trigger_code")
    if code is not None:
        return int(code)
    match = TRIGGER_RE.fullmatch(str(annotation.get("description", "")))
    return int(match.group(1)) if match else None


def build_trials(annotations: Iterable[dict[str, Any]], run_id: int) -> list[dict[str, Any]]:
    """Reconstruct and strictly validate trials from EDF+ annotations."""
    trials: list[dict[str, Any]] = []
    current: list[tuple[int, float]] | None = None

    for annotation in sorted(annotations, key=lambda item: float(item["onset_seconds"])):
        code = trigger_code(annotation)
        if code not in TRIAL_CODES:
            continue
        onset = float(annotation["onset_seconds"])
        if code == 20:
            if current is not None:
                raise TrialValidationError(
                    f"run {run_id}: encountered Trigger#20 before the previous trial ended"
                )
            current = [(code, onset)]
            continue
        if current is None:
            raise TrialValidationError(f"run {run_id}: Trigger#{code} occurred outside a trial")
        current.append((code, onset))
        if code != 70:
            continue

        codes = [event_code for event_code, _ in current]
        if len(codes) != 6 or codes[0] != 20 or codes[1] != 30 or codes[-2:] != [60, 70]:
            raise TrialValidationError(
                f"run {run_id}: expected [20,30,41/42,51/52,60,70], got {codes}"
            )
        stimulus_code, imagery_code = codes[2], codes[3]
        if stimulus_code not in (41, 42) or imagery_code not in (51, 52):
            raise TrialValidationError(f"run {run_id}: invalid stimulus/imagery pair {codes[2:4]}")
        if OBJECT_BY_CODE[stimulus_code] != OBJECT_BY_CODE[imagery_code]:
            raise TrialValidationError(f"run {run_id}: object mismatch {stimulus_code}->{imagery_code}")

        onsets = {event_code: event_onset for event_code, event_onset in current}
        stimulus_duration = onsets[imagery_code] - onsets[stimulus_code]
        imagery_duration = onsets[60] - onsets[imagery_code]
        if stimulus_duration < 4.0 or imagery_duration < 4.0:
            raise TrialValidationError(
                f"run {run_id}: phase shorter than 4 s "
                f"(stimulus={stimulus_duration:.6f}, imagery={imagery_duration:.6f})"
            )
        trials.append(
            {
                "run_id": run_id,
                "trial_in_run": len(trials),
                "object_label": OBJECT_BY_CODE[stimulus_code],
                "stimulus_code": stimulus_code,
                "stimulus_onset": onsets[stimulus_code],
                "imagery_code": imagery_code,
                "imagery_onset": onsets[imagery_code],
                "stimulus_duration": stimulus_duration,
                "imagery_duration": imagery_duration,
            }
        )
        current = None

    if current is not None:
        raise TrialValidationError(f"run {run_id}: final trial has no Trigger#70")
    if not trials:
        raise TrialValidationError(f"run {run_id}: no valid trials found")
    return trials


def _sos_bandpass(data: np.ndarray, sample_rate: float, low: float, high: float, order: int) -> np.ndarray:
    if not 0 < low < high < sample_rate / 2:
        raise ValueError(f"Invalid band [{low}, {high}] for sample rate {sample_rate}")
    sos = butter(order, (low, high), btype="bandpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, data, axis=1)


def extract_log_bandpower(
    eeg_uv: np.ndarray,
    sample_rate: float,
    target_rate: float = 100.0,
    bands: Iterable[tuple[str, float, float]] = DEFAULT_BANDS,
    filter_order: int = 4,
    log_epsilon: float = 1e-12,
) -> np.ndarray:
    """Create [channel, downsampled_time, band] log-Hilbert-power features."""
    eeg_uv = np.asarray(eeg_uv, dtype=np.float64)
    if eeg_uv.ndim != 2:
        raise ValueError(f"eeg_uv must have shape [channel,time], got {eeg_uv.shape}")
    if not np.isfinite(eeg_uv).all():
        raise ValueError("EEG contains non-finite values before filtering")
    ratio = Fraction(str(target_rate)) / Fraction(str(sample_rate))
    if ratio <= 0:
        raise ValueError("target_rate and sample_rate must be positive")

    features = []
    for _, low, high in bands:
        filtered = _sos_bandpass(eeg_uv, sample_rate, low, high, filter_order)
        power = np.abs(hilbert(filtered, axis=1)) ** 2
        log_power = np.log10(power + log_epsilon)
        downsampled = resample_poly(log_power, ratio.numerator, ratio.denominator, axis=1)
        features.append(downsampled)
    output = np.stack(features, axis=-1).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Frequency features contain non-finite values")
    return output


def _qc_metrics(
    broadband_uv: np.ndarray,
    raw_digital: np.ndarray,
    amplitude_threshold_uv: float,
    flat_std_threshold_uv: float,
) -> tuple[dict[str, float | int | bool], bool]:
    finite = np.isfinite(broadband_uv)
    nonfinite_count = int(broadband_uv.size - np.count_nonzero(finite))
    if finite.all():
        channel_ptp = np.ptp(broadband_uv, axis=1)
        channel_std = np.std(broadband_uv, axis=1)
        max_abs = float(np.max(np.abs(broadband_uv)))
        max_ptp = float(np.max(channel_ptp))
        min_std = float(np.min(channel_std))
    else:
        max_abs = max_ptp = min_std = math.nan
    clipped = bool(np.any((raw_digital == -32768) | (raw_digital == 32767)))
    flat = bool(np.isfinite(min_std) and min_std < flat_std_threshold_uv)
    over_amplitude = bool(np.isfinite(max_ptp) and max_ptp > amplitude_threshold_uv)
    flag = bool(nonfinite_count or clipped or flat or over_amplitude)
    return (
        {
            "max_channel_ptp_uv": max_ptp,
            "max_abs_uv": max_abs,
            "min_channel_std_uv": min_std,
            "nonfinite_count": nonfinite_count,
            "clipped": clipped,
            "flat": flat,
            "over_amplitude": over_amplitude,
        },
        flag,
    )


def preprocess_edf_directory(input_dir: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read all EDFs in a directory and return a serializable PT payload and summary."""
    files = sorted({*input_dir.glob("*.edf"), *input_dir.glob("*.EDF")})
    if not files:
        raise FileNotFoundError(f"No EDF files found in {input_dir}")

    channel_names = tuple(config["eeg_channels"])
    bands = tuple((str(item[0]), float(item[1]), float(item[2])) for item in config["bands"])
    target_rate = float(config["target_sample_rate_hz"])
    window_seconds = float(config["window_seconds"])
    window_offsets = tuple(float(value) for value in config["window_offsets_seconds"])
    filter_order = int(config["filter_order"])
    amplitude_threshold = float(config["qc_amplitude_threshold_uv"])
    flat_threshold = float(config["qc_flat_std_threshold_uv"])

    records: list[dict[str, Any]] = []
    all_trials: list[dict[str, Any]] = []
    global_trial_id = 0

    for run_id, path in enumerate(files):
        header = read_header(path)
        digital_signals = read_digital_signals(path, header)
        label_to_index = {signal.label.strip(): signal.index for signal in header.signals}
        missing = [name for name in channel_names if name not in label_to_index]
        if missing:
            raise ValueError(f"run {run_id}: missing EEG channels {missing}")
        eeg_indices = [label_to_index[name] for name in channel_names]
        sample_rates = [header.signals[index].sample_rate(header.record_duration) for index in eeg_indices]
        if not np.allclose(sample_rates, sample_rates[0]):
            raise ValueError(f"run {run_id}: EEG channels have different sample rates {sample_rates}")
        source_rate = float(sample_rates[0])
        expected_source_rate = float(config["source_sample_rate_hz"])
        if not math.isclose(source_rate, expected_source_rate, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"run {run_id}: expected {expected_source_rate} Hz EEG, got {source_rate} Hz"
            )

        eeg_uv = np.stack(
            [header.signals[index].digital_to_physical(digital_signals[index]) for index in eeg_indices]
        )
        raw_digital = np.stack([digital_signals[index] for index in eeg_indices])
        eeg_uv = eeg_uv - eeg_uv.mean(axis=0, keepdims=True)

        annotation_indices = [signal.index for signal in header.signals if signal.is_annotation]
        if len(annotation_indices) != 1:
            raise ValueError(f"run {run_id}: expected one EDF Annotations channel, got {len(annotation_indices)}")
        annotations = parse_tal(annotation_bytes(digital_signals[annotation_indices[0]]))
        trials = build_trials(annotations, run_id)

        band_features = extract_log_bandpower(
            eeg_uv,
            source_rate,
            target_rate=target_rate,
            bands=bands,
            filter_order=filter_order,
            log_epsilon=float(config["log_epsilon"]),
        )
        broadband = _sos_bandpass(eeg_uv, source_rate, 1.0, 45.0, filter_order)
        target_window_samples = round(window_seconds * target_rate)
        source_window_samples = round(window_seconds * source_rate)

        for trial in trials:
            trial["trial_id"] = global_trial_id
            all_trials.append(trial.copy())
            for phase_name, event_code, onset in (
                ("stimulus", trial["stimulus_code"], trial["stimulus_onset"]),
                ("imagery", trial["imagery_code"], trial["imagery_onset"]),
            ):
                for window_offset in window_offsets:
                    target_start = round((onset + window_offset) * target_rate)
                    source_start = round((onset + window_offset) * source_rate)
                    feature_window = band_features[
                        :, target_start : target_start + target_window_samples, :
                    ]
                    qc_window = broadband[:, source_start : source_start + source_window_samples]
                    digital_window = raw_digital[:, source_start : source_start + source_window_samples]
                    if feature_window.shape != (len(channel_names), target_window_samples, len(bands)):
                        raise ValueError(
                            f"run {run_id}, trial {trial['trial_in_run']}: incomplete feature window "
                            f"at {onset + window_offset:.6f} s"
                        )
                    qc, qc_flag = _qc_metrics(
                        qc_window, digital_window, amplitude_threshold, flat_threshold
                    )
                    records.append(
                        {
                            "features": feature_window,
                            "label_four": FOUR_CLASS_BY_CODE[event_code],
                            "label_object": trial["object_label"],
                            "phase_id": PHASE_BY_CODE[event_code],
                            "run_id": run_id,
                            "trial_id": global_trial_id,
                            "trial_in_run": trial["trial_in_run"],
                            "event_code": event_code,
                            "event_onset_seconds": onset,
                            "window_offset_seconds": window_offset,
                            "qc": qc,
                            "qc_flag": qc_flag,
                        }
                    )
            global_trial_id += 1

    from .splits import make_trial_fold_assignment

    fold_by_trial = make_trial_fold_assignment(all_trials, n_splits=5, random_state=42)
    qc_keys = (
        "max_channel_ptp_uv",
        "max_abs_uv",
        "min_channel_std_uv",
        "nonfinite_count",
        "clipped",
        "flat",
        "over_amplitude",
    )
    payload = {
        "format_version": "eeg-bandpower-v1",
        "features": torch.from_numpy(np.stack([record["features"] for record in records])),
        "label_four": torch.tensor([record["label_four"] for record in records], dtype=torch.long),
        "label_object": torch.tensor([record["label_object"] for record in records], dtype=torch.long),
        "phase": torch.tensor([record["phase_id"] for record in records], dtype=torch.long),
        "run_id": torch.tensor([record["run_id"] for record in records], dtype=torch.long),
        "trial_id": torch.tensor([record["trial_id"] for record in records], dtype=torch.long),
        "trial_in_run": torch.tensor([record["trial_in_run"] for record in records], dtype=torch.long),
        "trial_fold": torch.tensor([fold_by_trial[record["trial_id"]] for record in records], dtype=torch.long),
        "event_code": torch.tensor([record["event_code"] for record in records], dtype=torch.long),
        "event_onset_seconds": torch.tensor(
            [record["event_onset_seconds"] for record in records], dtype=torch.float64
        ),
        "window_offset_seconds": torch.tensor(
            [record["window_offset_seconds"] for record in records], dtype=torch.float32
        ),
        "qc_metrics": torch.tensor(
            [[float(record["qc"][key]) for key in qc_keys] for record in records], dtype=torch.float32
        ),
        "qc_flag": torch.tensor([record["qc_flag"] for record in records], dtype=torch.bool),
        "metadata": {
            "channel_names": list(channel_names),
            "phase_names": ["stimulus", "imagery"],
            "four_class_names": [
                "apple_stimulus",
                "hammer_stimulus",
                "apple_imagery",
                "hammer_imagery",
            ],
            "object_names": ["apple", "hammer"],
            "bands_hz": [[name, low, high] for name, low, high in bands],
            "source_sample_rate_hz": source_rate,
            "feature_sample_rate_hz": target_rate,
            "window_seconds": window_seconds,
            "window_offsets_seconds": list(window_offsets),
            "filter": {"type": "butterworth_sos_zero_phase", "order": filter_order},
            "feature": "log10_hilbert_power",
            "average_reference": True,
            "qc_metric_names": list(qc_keys),
            "qc_thresholds": {
                "max_channel_ptp_uv": amplitude_threshold,
                "min_channel_std_uv": flat_threshold,
            },
            "source_ids": [f"run_{index + 1:02d}" for index in range(len(files))],
            "trial_fold_random_state": 42,
        },
    }

    four_counts = Counter(int(value) for value in payload["label_four"].tolist())
    object_counts = Counter(
        int(payload["label_object"][index])
        for index in torch.nonzero(payload["phase"] == 1, as_tuple=False).flatten().tolist()
    )
    summary = {
        "format_version": payload["format_version"],
        "run_count": len(files),
        "trial_count": len(all_trials),
        "phase_count": len(all_trials) * 2,
        "window_count": len(records),
        "feature_shape": list(payload["features"].shape),
        "four_class_window_counts": dict(sorted(four_counts.items())),
        "imagery_object_window_counts": dict(sorted(object_counts.items())),
        "qc_flagged_windows": int(payload["qc_flag"].sum().item()),
        "source_ids": payload["metadata"]["source_ids"],
    }
    return payload, summary


def save_preprocessed(payload: dict[str, Any], summary: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path
