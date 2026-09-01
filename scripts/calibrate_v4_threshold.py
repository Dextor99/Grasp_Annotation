"""Calibrate a V4 HQ threshold from human G/B/U labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LABELS = {"G", "B", "U"}


def _metrics(samples, threshold):
    good = [sample for sample in samples if sample["label"] == "G"]
    bad = [sample for sample in samples if sample["label"] == "B"]
    tp = sum(sample["score"] >= threshold for sample in good)
    tn = sum(sample["score"] < threshold for sample in bad)
    sensitivity = tp / len(good) if good else 0.0
    specificity = tn / len(bad) if bad else 0.0
    balanced = (sensitivity + specificity) / 2.0 if good and bad else 0.0
    return {
        "threshold": float(threshold),
        "good_count": len(good),
        "bad_count": len(bad),
        "true_positive": tp,
        "true_negative": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced,
    }


def _best_threshold(samples):
    thresholds = np.linspace(0.0, 1.0, 101)
    metrics = [_metrics(samples, threshold) for threshold in thresholds]
    return max(metrics, key=lambda item: (item["balanced_accuracy"], -abs(item["threshold"] - 0.5), -item["threshold"])), metrics


def calibrate_rows(label_rows, manifest_rows):
    """Calibrate globally and with leave-one-object-out validation."""
    manifest = {str(row["sample_id"]): row for row in manifest_rows}
    samples = []
    uncertain_count = 0
    pending_count = 0
    for row in label_rows:
        sample_id = str(row["sample_id"])
        if sample_id not in manifest:
            raise ValueError(f"missing manifest row for sample_id={sample_id}")
        label = str(row.get("human_label", "")).strip().upper()
        if label == "":
            pending_count += 1
            continue
        if label not in LABELS:
            raise ValueError(f"invalid human_label={label!r} for sample_id={sample_id}")
        if label == "U":
            uncertain_count += 1
            continue
        score = float(manifest[sample_id]["score_total_v4"])
        if not np.isfinite(score):
            raise ValueError(f"non-finite score for sample_id={sample_id}")
        samples.append({"sample_id": sample_id, "object": row.get("object", ""), "label": label, "score": score})
    if not samples:
        raise ValueError("no labeled G/B samples available")
    best, metrics = _best_threshold(samples)
    loo = []
    for object_name in sorted({sample["object"] for sample in samples}):
        train = [sample for sample in samples if sample["object"] != object_name]
        test = [sample for sample in samples if sample["object"] == object_name]
        if not train or not any(sample["label"] == "G" for sample in train) or not any(sample["label"] == "B" for sample in train):
            continue
        train_best, _ = _best_threshold(train)
        evaluation = _metrics(test, train_best["threshold"])
        loo.append({"held_out_object": object_name, "train_threshold": train_best["threshold"], **evaluation})
    return {
        "best_threshold": best["threshold"],
        "best_balanced_accuracy": best["balanced_accuracy"],
        "best_sensitivity": best["sensitivity"],
        "best_specificity": best["specificity"],
        "labeled_count": len(samples),
        "uncertain_count": uncertain_count,
        "pending_count": pending_count,
        "threshold_metrics": metrics,
        "leave_one_object_out": loo,
    }


def calibrate_files(labels_csv, manifest_csv):
    with Path(labels_csv).open("r", encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))
    with Path(manifest_csv).open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    return calibrate_rows(labels, manifest)


def write_outputs(report, metrics_csv, report_json):
    metrics_csv = Path(metrics_csv)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(report["threshold_metrics"][0].keys())
    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "held_out_object", *fields])
        writer.writeheader()
        for metric in report["threshold_metrics"]:
            writer.writerow({"scope": "global", "held_out_object": "", **metric})
        for metric in report["leave_one_object_out"]:
            writer.writerow({"scope": "leave_one_object_out", **{key: metric.get(key, "") for key in ["held_out_object", *fields]}})
    report_json = Path(report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--report-json", required=True)
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    report = calibrate_files(args.labels_csv, args.manifest_csv)
    write_outputs(report, args.metrics_csv, args.report_json)
    print(
        f"Best V4 threshold={report['best_threshold']:.2f}, "
        f"balanced_accuracy={report['best_balanced_accuracy']:.3f}, "
        f"labeled={report['labeled_count']}, uncertain={report['uncertain_count']}, pending={report['pending_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
