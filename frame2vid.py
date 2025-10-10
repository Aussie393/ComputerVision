# frames_to_video.py
# Usage examples:
#   python frames_to_video.py /path/to/frames out.mp4 --fps 10
#   python frames_to_video.py /path/to/frames out.avi --fps 30 --size keep
#
# Requires: pip install opencv-python

import argparse, os, re, cv2
from glob import glob

def natural_key(s):
    # Sort "frame_2.jpg" before "frame_10.jpg"
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', os.path.basename(s))]

def pick_fourcc(outfile):
    ext = os.path.splitext(outfile)[1].lower()
    if ext in [".mp4", ".m4v", ".mov"]:
        return cv2.VideoWriter_fourcc(*"mp4v")   # widely supported
    elif ext in [".avi"]:
        return cv2.VideoWriter_fourcc(*"XVID")
    else:
        # default to mp4v if unknown
        return cv2.VideoWriter_fourcc(*"mp4v")

def main():
    ap = argparse.ArgumentParser(description="Assemble frames into a video using OpenCV.")
    ap.add_argument("frames_dir", help="Folder containing frames (jpg/png/jpeg).")
    ap.add_argument("out_video", help="Output video path (e.g., out.mp4).")
    ap.add_argument("--fps", type=float, default=10.0, help="Output frames per second (default: 10).")
    ap.add_argument("--pattern", default="*.jpg,*.png,*.jpeg", help="Glob patterns (comma-separated).")
    ap.add_argument("--size", default="auto", choices=["auto","keep","WxH"],
                    help="Output size: 'auto' (use first frame size), 'keep' (same), or 'WxH' (e.g., 1280x720).")
    args = ap.parse_args()

    # Collect frames
    patterns = [p.strip() for p in args.pattern.split(",")]
    paths = []
    for p in patterns:
        paths.extend(glob(os.path.join(args.frames_dir, p)))
    if not paths:
        raise SystemExit("No frames found. Adjust --pattern or check the folder.")

    paths.sort(key=natural_key)

    # Read first frame to get size
    first = cv2.imread(paths[0])
    if first is None:
        raise SystemExit(f"Failed to read first frame: {paths[0]}")

    if args.size in ("auto", "keep"):
        out_w, out_h = first.shape[1], first.shape[0]
    else:
        m = re.match(r"(\d+)[xX](\d+)", args.size)
        if not m:
            raise SystemExit("Invalid --size. Use 'auto', 'keep', or like '1280x720'.")
        out_w, out_h = int(m.group(1)), int(m.group(2))

    fourcc = pick_fourcc(args.out_video)
    writer = cv2.VideoWriter(args.out_video, fourcc, args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit("VideoWriter failed to open. Try a different extension/codec (e.g., .mp4).")

    count = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] Skipping unreadable frame: {p}")
            continue
        h, w = img.shape[:2]
        if (w, h) != (out_w, out_h):
            # Resize to match video size
            img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        count += 1

    writer.release()
    print(f"[DONE] Wrote {count} frames → {args.out_video} @ {args.fps} FPS, size {out_w}x{out_h}.")

if __name__ == "__main__":
    main()


