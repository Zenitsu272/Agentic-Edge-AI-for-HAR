"""Evaluate a labeled single-person video. Unknown predictions count as errors."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import InferencePipeline
from pose_activity import VERSION


def evaluate(video, labels, output, stride=2):
    annotations = json.loads(Path(labels).read_text())
    for segment in annotations:
        if segment['label'] not in ('Standing', 'Sitting', 'Fall', 'Lying', 'Unknown') or segment['end'] <= segment['start']:
            raise ValueError('Each annotation needs start < end and a supported label')
    pipeline = InferencePipeline()
    # Evaluation must not create user-facing live alerts/history.
    pipeline._persist_changes = lambda *args: None
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f'Cannot open {video}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError('Video has no valid frame rate')
    confusion, totals, latencies, timeline = defaultdict(Counter), Counter(), [], []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t = index / fps
            index += 1
            if (index - 1) % stride:
                continue
            result = pipeline.process_unified(frame, timestamp=t)
            predicted = result['stage2_posture']['category']
            expected = [a['label'] for a in annotations if a['start'] <= t < a['end']]
            if len(expected) > 1:
                raise ValueError(f'Overlapping labels at {t:.2f}s')
            if expected:
                confusion[expected[0]][predicted] += 1
                totals[expected[0]] += 1
            latencies.append(result['inference_ms'])
            timeline.append(dict(time=round(t, 3), expected=expected[0] if expected else None,
                                 predicted=predicted, provisional=result['stage2_posture']['provisional'],
                                 people=len(result['poses'])))
    finally:
        capture.release()
    count = sum(totals.values())
    correct = sum(confusion[c][c] for c in totals)
    report = dict(classifier_version=VERSION, model=pipeline.model_path.name, video=str(video), labels=str(labels),
                  labeled_frames=count, accuracy=round(correct / count, 4) if count else None,
                  per_class_recall={c: round(confusion[c][c] / n, 4) for c, n in totals.items()},
                  confusion_matrix={c: dict(row) for c, row in confusion.items()},
                  mean_inference_ms=round(float(np.mean(latencies)), 1) if latencies else None,
                  limitation='Single-person primary-activity evaluation; frame recall is not fall event recall.', timeline=timeline)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'timeline'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video')
    parser.add_argument('labels')
    parser.add_argument('--output', default='data/activity_evaluation.json')
    parser.add_argument('--stride', type=int, default=2)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be positive')
    evaluate(args.video, args.labels, args.output, args.stride)
