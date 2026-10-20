import cv2
import os

video_path = r'D:\Recording 2026-10-20 130626.mp4'
out_dir = r'C:\Users\VISHAL\.gemini\antigravity-ide\brain\1155642b-567e-4ba1-aa9b-27067b97d5c7\scratch\video_frames'
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Sample every 4 seconds (124 seconds total -> ~31 snapshots)
sample_interval_sec = 4
sample_frames = [int(i * sample_interval_sec * fps) for i in range(int(125 / sample_interval_sec))]

extracted = []
for f_idx in sample_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret:
        break
    sec = f_idx / fps
    out_file = os.path.join(out_dir, f"frame_{int(sec):03d}s.jpg")
    # Save slightly downscaled to keep inspection fast
    resized = cv2.resize(frame, (960, 460))
    cv2.imwrite(out_file, resized)
    extracted.append((sec, out_file))

cap.release()
print(f"Extracted {len(extracted)} representative frames across the video.")
