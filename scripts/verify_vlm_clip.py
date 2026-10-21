"""Run actual local VLM verification on a short video window, independently of pose triggering."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlm_adapter import VLMAdapter


def verify_clip(adapter, video, start=0, end=None, trigger=None, rgb_right_half=False):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened(): raise ValueError(f'Cannot open {video}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total < 3:
        capture.release()
        raise ValueError('Video requires a valid frame rate and at least three frames.')
    end = min(end if end is not None else (total-1)/fps, (total-1)/fps)
    if start < 0 or start >= end:
        capture.release()
        raise ValueError('Choose 0 <= start < end within the video.')
    trigger = trigger if trigger is not None else (start+end)/2
    images = []
    try:
        for t in np.linspace(start, end, 12):
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(t*fps))
            ok, frame = capture.read()
            if not ok: continue
            if rgb_right_half: frame = frame[:,frame.shape[1]//2:]
            images.append({'frame':frame,'relative_sec':float(t-trigger)})
    finally:
        capture.release()
    return adapter.verify_video_event({'id':Path(video).stem,'camera_id':'Offline validation'},images)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video')
    parser.add_argument('--start',type=float,default=0)
    parser.add_argument('--end',type=float)
    parser.add_argument('--trigger',type=float)
    parser.add_argument('--model-dir')
    parser.add_argument('--rgb-right-half',action='store_true',help='For URFD preview videos that combine depth and RGB.')
    parser.add_argument('--output')
    args=parser.parse_args()
    adapter=VLMAdapter(model_dir=args.model_dir)
    print('Model status:',adapter.status,adapter.status_detail,flush=True)
    result=verify_clip(adapter,args.video,args.start,args.end,args.trigger,args.rgb_right_half)
    if args.output: Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if result.get('verification_method') != 'VLM_MODEL': raise SystemExit(1)
