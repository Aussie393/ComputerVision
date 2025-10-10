# save_frames_10fps.py
# Usage:
#   python save_frames_10fps.py input.mov output_frames  --fps 10
#
# Requires: pip install opencv-python

import os
import cv2
import argparse

def extract_frames(input_path, output_dir, target_fps=10, img_ext=".jpg"):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    # Time step in milliseconds for target fps
    step_ms = 1000.0 / float(target_fps)
    next_save_time = 0.0  # in ms
    saved = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    print(f"[INFO] Reading: {input_path}")
    print(f"[INFO] Saving frames at {target_fps} FPS to: {output_dir}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Current timestamp in ms (best-effort; works with VFR too)
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # Save on schedule (allow small jitter)
        if t_ms + 0.5 >= next_save_time:
            # Filename with index and timestamp (seconds with 3 decimals)
            t_sec = t_ms / 1000.0
            fname = f"frame_{saved:06d}_t{t_sec:010.3f}{img_ext}"
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, frame)
            saved += 1
            next_save_time += step_ms

    cap.release()
    print(f"[DONE] Saved {saved} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a .mov at fixed FPS.")
    parser.add_argument("input", help="Path to input .mov (or any video readable by OpenCV)")
    parser.add_argument("output_dir", help="Directory to save frames")
    parser.add_argument("--fps", type=float, default=10.0, help="Target frames per second (default: 10)")
    parser.add_argument("--ext", type=str, default=".jpg", choices=[".jpg", ".png"], help="Image extension")
    args = parser.parse_args()

    extract_frames(args.input, args.output_dir, target_fps=args.fps, img_ext=args.ext)
