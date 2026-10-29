"""Score the screen-type classifier against hand-labelled frames.

    python tools/evaluate_perception.py --labels data/phase1/labels.json

Labels are the ground truth; the classifier never sees them. Prints per-class
recall, overall accuracy, and a confusion matrix, and exits non-zero if accuracy
is below --threshold so it can be wired into a gate check later.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

# Allow `python tools/evaluate_perception.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.game_hsr.perception import HsrPerception


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate screen-type classification")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--no-ocr", action="store_true", help="visual features only")
    parser.add_argument("--show-errors", type=int, default=12)
    args = parser.parse_args(argv)

    records = json.loads(args.labels.read_text(encoding="utf-8"))
    labelled = [r for r in records if r.get("label")]
    if not labelled:
        raise SystemExit("no labelled entries")

    ocr = None
    if not args.no_ocr:
        from adapters.game_hsr.perception import RapidOcr

        ocr = RapidOcr()

    perception = HsrPerception(ocr=ocr, ocr_enabled=not args.no_ocr)

    correct = 0
    per_class_total: Counter[str] = Counter()
    per_class_hit: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    errors: list[tuple[str, str, str, float]] = []

    for record in labelled:
        frame = cv2.imread(record["path"])
        if frame is None:
            print(f"  unreadable: {record['path']}")
            continue
        predicted, confidence, _features = perception.classify(frame)

        truth = record["label"]
        per_class_total[truth] += 1
        confusion[truth][predicted] += 1
        if predicted == truth:
            correct += 1
            per_class_hit[truth] += 1
        else:
            errors.append((truth, predicted, record["path"], confidence))

    total = sum(per_class_total.values())
    accuracy = correct / total if total else 0.0

    print(f"\nlabelled frames: {total}")
    print(f"accuracy: {correct}/{total} = {accuracy:.1%}\n")
    print(f"{'class':<16}{'n':>4}{'recall':>9}")
    for name in sorted(per_class_total):
        hits = per_class_hit[name]
        n = per_class_total[name]
        print(f"{name:<16}{n:>4}{hits / n:>9.0%}")

    print("\nconfusion (truth -> predicted):")
    for truth in sorted(confusion):
        row = ", ".join(f"{k}:{v}" for k, v in sorted(confusion[truth].items()))
        print(f"  {truth:<16} {row}")

    if errors and args.show_errors:
        print(f"\nfirst {min(args.show_errors, len(errors))} errors:")
        for truth, predicted, path, confidence in errors[: args.show_errors]:
            print(f"  {truth:>14} -> {predicted:<14} conf={confidence:.2f}  {Path(path).name}")

    if accuracy < args.threshold:
        print(f"\nGATE 1 (screen type): FAIL (need {args.threshold:.0%})")
        return 1
    print(f"\nGATE 1 (screen type): PASS (>= {args.threshold:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
