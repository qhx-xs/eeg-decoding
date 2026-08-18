from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from EEG_Model import EEG_CNN_BiLSTM_Attention
from eeg_pipeline.preprocessing import TrialValidationError, build_trials, extract_log_bandpower
from eeg_pipeline.reporting import save_cv_artifacts
from eeg_pipeline.splits import build_cv_splits, make_trial_fold_assignment
from eeg_pipeline.training import load_pt_dataset, metrics_from_logits
from inspect_edf import parse_tal, read_header


def annotation(code: int, onset: float) -> dict:
    return {"description": f"Trigger#{code}", "onset_seconds": onset}


def valid_annotations() -> list[dict]:
    return [
        annotation(12, 0.0),
        annotation(20, 1.0),
        annotation(30, 2.0),
        annotation(41, 3.0),
        annotation(51, 7.01),
        annotation(60, 11.02),
        annotation(70, 15.03),
        annotation(13, 16.0),
    ]


class TriggerAndEdfTests(unittest.TestCase):
    def test_tal_annotation_parser(self):
        raw = b"+1.250\x14Trigger#41\x14\x00+5.250\x14Trigger#51\x14\x00"
        parsed = parse_tal(raw)
        self.assertEqual([item["trigger_code"] for item in parsed], [41, 51])
        self.assertEqual(parsed[0]["meaning"], "Apple stimulus")

    def test_build_trials_and_reject_mismatch(self):
        trials = build_trials(valid_annotations(), run_id=0)
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0]["object_label"], 0)
        bad = valid_annotations()
        bad[4] = annotation(52, 7.01)
        with self.assertRaises(TrialValidationError):
            build_trials(bad, run_id=0)

    def test_minimal_edf_header(self):
        def field(value: str, width: int) -> bytes:
            return value.encode("ascii").ljust(width)

        fixed = b"".join(
            [
                field("0", 8), field("X", 80), field("test", 80), field("01.01.26", 8),
                field("00.00.00", 8), field("512", 8), field("EDF+C", 44), field("0", 8),
                field("1", 8), field("1", 4),
            ]
        )
        signal = b"".join(
            [
                field("Cz", 16), field("AgAgCl", 80), field("uV", 8), field("-100", 8),
                field("100", 8), field("-32768", 8), field("32767", 8), field("HP:0.1Hz", 80),
                field("500", 8), field("", 32),
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "minimal.edf"
            path.write_bytes(fixed + signal)
            header = read_header(path)
        self.assertEqual(header.signals[0].label, "Cz")
        self.assertEqual(header.signals[0].sample_rate(header.record_duration), 500.0)


class FeatureTests(unittest.TestCase):
    def test_six_band_features_are_finite(self):
        sample_rate = 500
        time = np.arange(sample_rate * 6) / sample_rate
        eeg = np.stack(
            [20 * np.sin(2 * np.pi * 10 * time), 10 * np.sin(2 * np.pi * 22 * time)]
        )
        output = extract_log_bandpower(eeg, sample_rate=sample_rate, target_rate=100)
        self.assertEqual(output.shape, (2, 600, 6))
        self.assertTrue(np.isfinite(output).all())
        self.assertLess(float(np.abs(output).max()), 100.0)

    def test_safe_pt_round_trip(self):
        payload = {
            "format_version": "eeg-bandpower-v1",
            "features": torch.zeros(1, 8, 200, 6),
            "label_four": torch.tensor([0]),
            "label_object": torch.tensor([0]),
            "phase": torch.tensor([0]),
            "run_id": torch.tensor([0]),
            "trial_id": torch.tensor([0]),
            "trial_fold": torch.tensor([0]),
            "event_code": torch.tensor([41]),
            "qc_flag": torch.tensor([False]),
            "metadata": {"channel_names": ["O1", "Oz", "O2", "Pz", "C3", "C4", "Cz", "Fz"]},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.pt"
            torch.save(payload, path)
            loaded = load_pt_dataset(path)
        self.assertEqual(tuple(loaded["features"].shape), (1, 8, 200, 6))


class SplitTests(unittest.TestCase):
    @staticmethod
    def dataset() -> dict:
        trials = []
        trial_id = 0
        for run_id in range(4):
            for object_label in (0, 1):
                for _ in range(10):
                    trials.append(
                        {"trial_id": trial_id, "run_id": run_id, "object_label": object_label}
                    )
                    trial_id += 1
        assignment = make_trial_fold_assignment(trials)
        run_ids, trial_ids, trial_folds, phases, labels_object, labels_four, event_codes = ([] for _ in range(7))
        for trial in trials:
            for phase in (0, 1):
                for _ in range(3):
                    run_ids.append(trial["run_id"])
                    trial_ids.append(trial["trial_id"])
                    trial_folds.append(assignment[trial["trial_id"]])
                    phases.append(phase)
                    labels_object.append(trial["object_label"])
                    labels_four.append(phase * 2 + trial["object_label"])
                    event_codes.append((41, 42, 51, 52)[phase * 2 + trial["object_label"]])
        return {
            "features": torch.zeros(len(run_ids), 1),
            "run_id": torch.tensor(run_ids),
            "trial_id": torch.tensor(trial_ids),
            "trial_fold": torch.tensor(trial_folds),
            "phase": torch.tensor(phases),
            "label_object": torch.tensor(labels_object),
            "label_four": torch.tensor(labels_four),
            "event_code": torch.tensor(event_codes),
        }

    def test_run_and_trial_splits_do_not_leak(self):
        dataset = self.dataset()
        run_splits = build_cv_splits(dataset, "run", "four_class")
        self.assertEqual(len(run_splits), 4)
        self.assertEqual([len(run_splits[0][key]) for key in ("train", "val", "test")], [240, 120, 120])
        trial_splits = build_cv_splits(dataset, "trial", "four_class")
        self.assertEqual(len(trial_splits), 5)
        self.assertEqual([len(trial_splits[0][key]) for key in ("train", "val", "test")], [288, 96, 96])
        imagery_splits = build_cv_splits(dataset, "run", "imagery_binary")
        self.assertEqual([len(imagery_splits[0][key]) for key in ("train", "val", "test")], [120, 60, 60])

    def test_window_logits_are_aggregated(self):
        logits = torch.tensor([[4.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 4.0]])
        labels = torch.tensor([0, 0, 1, 1])
        groups = torch.tensor([10, 10, 11, 11])
        metrics = metrics_from_logits(logits, labels, groups, num_classes=2)
        self.assertEqual(metrics["phase_or_trial"]["sample_count"], 2)
        self.assertEqual(metrics["phase_or_trial"]["accuracy"], 1.0)


class ModelTests(unittest.TestCase):
    def test_model_output_shapes(self):
        samples = torch.randn(2, 8, 200, 6)
        for class_count in (4, 2):
            model = EEG_CNN_BiLSTM_Attention(channels=8, num_classes=class_count)
            with torch.no_grad():
                output = model(samples)
            self.assertEqual(tuple(output.shape), (2, class_count))


class ReportingTests(unittest.TestCase):
    def test_confusion_matrix_and_learning_curve_files(self):
        metrics = {
            "phase_or_trial": {"confusion_matrix": [[2, 1], [0, 3]]},
            "window": {"confusion_matrix": [[5, 2], [1, 4]]},
        }
        report = {
            "task": "imagery_binary",
            "cv": "run",
            "folds": [
                {
                    "fold": 0,
                    "metrics": {"test": metrics},
                    "history": [
                        {
                            "epoch": 1,
                            "train_loss": 0.8,
                            "val_phase_macro_f1": 0.5,
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            save_cv_artifacts(report, output, ["apple", "hammer"])
            for name in (
                "confusion_phase_or_trial.png",
                "confusion_phase_or_trial_counts.csv",
                "confusion_phase_or_trial_normalized.csv",
                "confusion_window.png",
                "fold_0_confusion_phase_or_trial.png",
                "learning_curves.png",
            ):
                self.assertGreater((output / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
