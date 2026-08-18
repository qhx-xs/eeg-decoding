#!/usr/bin/env python3
"""Inspect EDF/EDF+ files, including annotations and trigger-like channels.

This script deliberately uses only Python's standard library plus NumPy, so it
can be run in this project without installing MNE or pyEDFlib.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRIGGER_DEFINITIONS = {
    41: "Apple stimulus",
    42: "Hammer stimulus",
    51: "Apple visual imagery",
    52: "Hammer visual imagery",
}
TRIGGER_WORDS = ("trigger", "status", "event", "marker", "stim")
TRIGGER_PATTERN = re.compile(r"^Trigger#(\d+)$", re.IGNORECASE)


def _text(value: bytes) -> str:
    return value.decode("latin-1", errors="replace").strip()


def _number(value: bytes, kind, field: str):
    text = _text(value)
    try:
        return kind(text)
    except ValueError as exc:
        raise ValueError(f"Invalid EDF {field}: {text!r}") from exc


@dataclass
class SignalHeader:
    index: int
    label: str
    transducer: str
    physical_dimension: str
    physical_minimum: float
    physical_maximum: float
    digital_minimum: int
    digital_maximum: int
    prefiltering: str
    samples_per_record: int
    reserved: str

    @property
    def is_annotation(self) -> bool:
        return self.label.strip().lower() == "edf annotations"

    def sample_rate(self, record_duration: float) -> float:
        return self.samples_per_record / record_duration

    def digital_to_physical(self, values: np.ndarray) -> np.ndarray:
        digital_range = self.digital_maximum - self.digital_minimum
        if digital_range == 0:
            return values.astype(np.float64)
        scale = (self.physical_maximum - self.physical_minimum) / digital_range
        return (values.astype(np.float64) - self.digital_minimum) * scale + self.physical_minimum


@dataclass
class EdfHeader:
    path: str
    version: str
    patient: str
    recording: str
    start_date: str
    start_time: str
    header_bytes: int
    reserved: str
    data_records: int
    record_duration: float
    signals: list[SignalHeader]


def read_header(path: Path) -> EdfHeader:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError("File is too short to contain an EDF header")

        version = _text(fixed[0:8])
        patient = _text(fixed[8:88])
        recording = _text(fixed[88:168])
        start_date = _text(fixed[168:176])
        start_time = _text(fixed[176:184])
        header_bytes = _number(fixed[184:192], int, "header byte count")
        reserved = _text(fixed[192:236])
        data_records = _number(fixed[236:244], int, "data record count")
        record_duration = _number(fixed[244:252], float, "record duration")
        signal_count = _number(fixed[252:256], int, "signal count")

        signal_block = handle.read(signal_count * 256)
        if len(signal_block) != signal_count * 256:
            raise ValueError("Truncated EDF signal header")

    cursor = 0

    def fields(width: int) -> list[str]:
        nonlocal cursor
        result = [
            _text(signal_block[cursor + i * width : cursor + (i + 1) * width])
            for i in range(signal_count)
        ]
        cursor += signal_count * width
        return result

    labels = fields(16)
    transducers = fields(80)
    dimensions = fields(8)
    physical_minima = fields(8)
    physical_maxima = fields(8)
    digital_minima = fields(8)
    digital_maxima = fields(8)
    prefiltering = fields(80)
    samples_per_record = fields(8)
    signal_reserved = fields(32)

    signals = []
    for index in range(signal_count):
        signals.append(
            SignalHeader(
                index=index,
                label=labels[index],
                transducer=transducers[index],
                physical_dimension=dimensions[index],
                physical_minimum=float(physical_minima[index]),
                physical_maximum=float(physical_maxima[index]),
                digital_minimum=int(digital_minima[index]),
                digital_maximum=int(digital_maxima[index]),
                prefiltering=prefiltering[index],
                samples_per_record=int(samples_per_record[index]),
                reserved=signal_reserved[index],
            )
        )

    expected_header_bytes = 256 + signal_count * 256
    if header_bytes < expected_header_bytes:
        raise ValueError(
            f"Header says {header_bytes} bytes, smaller than expected {expected_header_bytes}"
        )
    if record_duration <= 0:
        raise ValueError(f"Invalid EDF record duration: {record_duration}")

    return EdfHeader(
        path=str(path.resolve()),
        version=version,
        patient=patient,
        recording=recording,
        start_date=start_date,
        start_time=start_time,
        header_bytes=header_bytes,
        reserved=reserved,
        data_records=data_records,
        record_duration=record_duration,
        signals=signals,
    )


def read_digital_signals(path: Path, header: EdfHeader) -> list[np.ndarray]:
    """Read all EDF samples as signed 16-bit digital values."""
    samples_per_record = sum(signal.samples_per_record for signal in header.signals)
    bytes_per_record = samples_per_record * 2
    file_data_bytes = path.stat().st_size - header.header_bytes
    record_count = header.data_records
    if record_count < 0:  # EDF+ may use -1 when recording is interrupted.
        record_count = file_data_bytes // bytes_per_record
    expected = record_count * bytes_per_record
    if file_data_bytes < expected:
        raise ValueError(
            f"Truncated EDF data: expected {expected} bytes for {record_count} records, "
            f"found {file_data_bytes}"
        )

    output = [
        np.empty(record_count * signal.samples_per_record, dtype="<i2")
        for signal in header.signals
    ]
    offsets = [0] * len(header.signals)

    with path.open("rb") as handle:
        handle.seek(header.header_bytes)
        for _ in range(record_count):
            raw_record = handle.read(bytes_per_record)
            cursor = 0
            for i, signal in enumerate(header.signals):
                byte_count = signal.samples_per_record * 2
                values = np.frombuffer(raw_record, dtype="<i2", count=signal.samples_per_record, offset=cursor)
                start = offsets[i]
                output[i][start : start + signal.samples_per_record] = values
                offsets[i] += signal.samples_per_record
                cursor += byte_count
    return output


def parse_tal(raw: bytes) -> list[dict]:
    """Parse EDF+ Time-stamped Annotation Lists (TALs)."""
    annotations: list[dict] = []
    for tal in raw.split(b"\x00"):
        tal = tal.strip(b"\x00")
        if not tal:
            continue
        parts = tal.split(b"\x14")
        timing = parts[0]
        if b"\x15" in timing:
            onset_raw, duration_raw = timing.split(b"\x15", 1)
        else:
            onset_raw, duration_raw = timing, b""
        try:
            onset = float(_text(onset_raw))
        except ValueError:
            continue
        try:
            duration = float(_text(duration_raw)) if duration_raw else None
        except ValueError:
            duration = None
        for description_raw in parts[1:]:
            description = _text(description_raw)
            if description:
                match = TRIGGER_PATTERN.match(description)
                code = int(match.group(1)) if match else None
                annotations.append({
                    "onset_seconds": onset,
                    "duration_seconds": duration,
                    "description": description,
                    "trigger_code": code,
                    "meaning": TRIGGER_DEFINITIONS.get(code),
                })
    return annotations


def annotation_bytes(values: np.ndarray) -> bytes:
    # EDF+ stores two annotation bytes in each little-endian int16 sample.
    return values.astype("<i2", copy=False).tobytes()


def trigger_events(values: np.ndarray, sample_rate: float) -> list[dict]:
    """Return non-zero runs/value transitions from a digital trigger channel."""
    if values.size == 0:
        return []
    changes = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    ends = np.r_[changes[1:], values.size]
    events = []
    for start, end in zip(changes, ends):
        signed_value = int(values[start])
        unsigned_value = signed_value & 0xFFFF
        low_byte = unsigned_value & 0xFF
        if unsigned_value == 0:
            continue
        decoded = TRIGGER_DEFINITIONS.get(unsigned_value) or TRIGGER_DEFINITIONS.get(low_byte)
        events.append(
            {
                "sample": int(start),
                "onset_seconds": float(start / sample_rate),
                "duration_seconds": float((end - start) / sample_rate),
                "digital_value_signed": signed_value,
                "digital_value_unsigned": unsigned_value,
                "low_byte": low_byte,
                "meaning": decoded,
            }
        )
    return events


def signal_summary(signal: SignalHeader, values: np.ndarray, duration: float) -> dict:
    if signal.is_annotation:
        annotations = parse_tal(annotation_bytes(values))
        return {
            "index": signal.index,
            "label": signal.label,
            "kind": "annotation",
            "sample_rate_hz": signal.sample_rate(duration),
            "annotation_count": len(annotations),
            "annotations": annotations,
        }

    physical = signal.digital_to_physical(values)
    return {
        "index": signal.index,
        "label": signal.label,
        "kind": "signal",
        "unit": signal.physical_dimension,
        "sample_rate_hz": signal.sample_rate(duration),
        "sample_count": int(values.size),
        "digital_min_observed": int(values.min()) if values.size else None,
        "digital_max_observed": int(values.max()) if values.size else None,
        "physical_min_observed": float(np.min(physical)) if values.size else None,
        "physical_max_observed": float(np.max(physical)) if values.size else None,
        "physical_mean": float(np.mean(physical)) if values.size else None,
        "physical_std": float(np.std(physical)) if values.size else None,
        "first_10_physical": physical[:10].tolist(),
    }


def is_trigger_candidate(signal: SignalHeader, values: np.ndarray) -> bool:
    label = signal.label.lower()
    if any(word in label for word in TRIGGER_WORDS):
        return True
    if signal.is_annotation or values.size == 0:
        return False
    # An unnamed trigger/status channel is usually dimensionless, mostly zero,
    # and contains known event codes. Requiring these properties avoids treating
    # low-variation accelerometer channels as triggers.
    if signal.physical_dimension.strip():
        return False
    unique = np.unique(values)
    low_bytes = {int(value) & 0xFF for value in unique}
    return (
        unique.size <= 64
        and 0 in low_bytes
        and bool(low_bytes.intersection(TRIGGER_DEFINITIONS))
    )


def inspect_file(path: Path, max_events: int) -> dict:
    header = read_header(path)
    values = read_digital_signals(path, header)
    duration_seconds = (
        (header.data_records if header.data_records >= 0 else values[0].size / header.signals[0].samples_per_record)
        * header.record_duration
    )
    summaries = [
        signal_summary(signal, signal_values, header.record_duration)
        for signal, signal_values in zip(header.signals, values)
    ]

    candidates = []
    for signal, signal_values in zip(header.signals, values):
        if not is_trigger_candidate(signal, signal_values):
            continue
        unique_counts = Counter(int(x) for x in signal_values.tolist())
        events = trigger_events(signal_values, signal.sample_rate(header.record_duration))
        candidates.append(
            {
                "index": signal.index,
                "label": signal.label,
                "unique_digital_values": dict(sorted(unique_counts.items())),
                "event_count": len(events),
                "events": events[:max_events],
                "events_truncated": len(events) > max_events,
            }
        )

    annotation_events = []
    for summary in summaries:
        if summary["kind"] == "annotation":
            annotation_events.extend(summary["annotations"])
    annotation_counts = Counter(
        event["description"] for event in annotation_events if event["description"]
    )

    return {
        "file": str(path.resolve()),
        "format": header.reserved or "EDF",
        "patient": header.patient,
        "recording": header.recording,
        "start": f"{header.start_date} {header.start_time}",
        "data_records": header.data_records,
        "record_duration_seconds": header.record_duration,
        "total_duration_seconds": duration_seconds,
        "signal_count": len(header.signals),
        "signals": summaries,
        "annotations": annotation_events[:max_events],
        "annotation_counts": dict(sorted(annotation_counts.items())),
        "annotations_truncated": len(annotation_events) > max_events,
        "trigger_candidates": candidates,
    }


def format_report(report: dict) -> str:
    lines = [
        "=" * 80,
        report["file"],
        f"格式: {report['format']} | 开始: {report['start']} | "
        f"时长: {report['total_duration_seconds']:.3f} s | 数据记录: {report['data_records']}",
        f"受试者: {report['patient'] or '(空)'}",
        f"记录信息: {report['recording'] or '(空)'}",
        f"通道数: {report['signal_count']}",
        "",
        "通道结构:",
    ]
    for signal in report["signals"]:
        if signal["kind"] == "annotation":
            lines.append(
                f"  [{signal['index']:02d}] {signal['label']:<20} EDF+注释 "
                f"({signal['annotation_count']} 条)"
            )
        else:
            lines.append(
                f"  [{signal['index']:02d}] {signal['label']:<20} "
                f"{signal['sample_rate_hz']:8.3f} Hz  {signal['unit'] or '(无单位)':<8} "
                f"范围 [{signal['physical_min_observed']:.6g}, {signal['physical_max_observed']:.6g}] "
                f"均值 {signal['physical_mean']:.6g} 标准差 {signal['physical_std']:.6g}"
            )

    lines.extend(["", f"EDF+ annotations（计数={report['annotation_counts']}）:"])
    if report["annotations"]:
        for event in report["annotations"]:
            meaning = f" -> {event['meaning']}" if event["meaning"] else ""
            lines.append(
                f"  t={event['onset_seconds']:.6f}s  duration={event['duration_seconds']}  "
                f"description={event['description']!r}{meaning}"
            )
    else:
        lines.append("  (未发现)")

    lines.extend(["", "Trigger/Status 候选通道:"])
    if not report["trigger_candidates"]:
        lines.append("  (未发现命名像 trigger 的通道，也未发现低基数非零数字通道)")
    for candidate in report["trigger_candidates"]:
        lines.append(
            f"  [{candidate['index']:02d}] {candidate['label']} | "
            f"取值计数={candidate['unique_digital_values']} | 事件段={candidate['event_count']}"
        )
        for event in candidate["events"]:
            meaning = f" -> {event['meaning']}" if event["meaning"] else ""
            lines.append(
                f"      t={event['onset_seconds']:.6f}s, duration={event['duration_seconds']:.6f}s, "
                f"raw={event['digital_value_signed']}, uint16={event['digital_value_unsigned']}, "
                f"low8={event['low_byte']}{meaning}"
            )
        if candidate["events_truncated"]:
            lines.append("      ...（事件过多，已截断；可用 --max-events 调大）")

    return "\n".join(lines)


def find_edf_files(targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        path = Path(target).expanduser()
        if path.is_dir():
            files.extend(path.rglob("*.edf"))
            files.extend(path.rglob("*.EDF"))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"路径不存在: {path}")
    return sorted(set(file.resolve() for file in files))


def main() -> int:
    parser = argparse.ArgumentParser(description="查看 EDF/EDF+ 的通道、数值范围、注释和 trigger")
    parser.add_argument("paths", nargs="+", help="一个或多个 EDF 文件/目录")
    parser.add_argument("--max-events", type=int, default=100, help="每个文件最多展示多少个事件")
    parser.add_argument("--json", dest="json_path", help="同时将完整结果写入 JSON 文件")
    args = parser.parse_args()

    files = find_edf_files(args.paths)
    if not files:
        parser.error("指定路径下没有找到 EDF 文件")

    reports = []
    failures = 0
    for path in files:
        try:
            report = inspect_file(path, max_events=max(0, args.max_events))
            reports.append(report)
            print(format_report(report))
        except Exception as exc:  # Continue so one corrupt file does not hide the rest.
            failures += 1
            print(f"\n读取失败: {path}\n  {type(exc).__name__}: {exc}")

    if args.json_path:
        output_path = Path(args.json_path).expanduser().resolve()
        output_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON 已写入: {output_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
