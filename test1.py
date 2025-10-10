# periphery_mask_webcam.py
# pip install opencv-python

import cv2
import numpy as np

# ---- Fixed trapezium (normalized 0..1) ----
TRAP = dict(
    cx=0.5,      # horizontal center of trapezium
    y_top=0.25,  # top edge (fraction from top)
    y_bot=1,  # bottom edge (must be > y_top)
    w_top=0.25,  # width fraction at top
    w_bot=1,  # width fraction at bottom
)

def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def trap_points(h, w, p=TRAP):
    cx = p["cx"] * w
    y_top = p["y_top"] * h
    y_bot = p["y_bot"] * h
    half_top = 0.5 * p["w_top"] * w
    half_bot = 0.5 * p["w_bot"] * w
    tl = (int(clamp(cx - half_top, 0, w-1)), int(clamp(y_top, 0, h-1)))
    tr = (int(clamp(cx + half_top, 0, w-1)), int(clamp(y_top, 0, h-1)))
    br = (int(clamp(cx + half_bot, 0, w-1)), int(clamp(y_bot, 0, h-1)))
    bl = (int(clamp(cx - half_bot, 0, w-1)), int(clamp(y_bot, 0, h-1)))
    return np.array([tl, tr, br, bl], dtype=np.int32)

def mask_trapezium_out(frame, poly, blackout_alpha=0.75):
    """Return two views:
       1) overlay_view: frame with the trapezium darkened (masked out)
       2) periphery_only: everything outside the trapezium, inside is removed
    """
    h, w = frame.shape[:2]
    # Binary mask where trapezium == 255 (inside), outside == 0
    trap_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(trap_mask, [poly], 255)

    # ---- 1) Darken the trapezium region on the original frame
    overlay = frame.copy()
    black = np.zeros_like(frame)
    # Blend only inside the trapezium
    mask_3 = cv2.merge([trap_mask, trap_mask, trap_mask]) / 255.0
    overlay = (frame * (1 - mask_3 * blackout_alpha) + black * (mask_3 * blackout_alpha)).astype(np.uint8)

    # Draw a border for clarity
    cv2.polylines(overlay, [poly], isClosed=True, color=(0, 0, 0), thickness=2)

    # ---- 2) Periphery-only view (keep outside, remove inside)
    periphery_mask = cv2.bitwise_not(trap_mask)  # 255 outside, 0 inside
    periphery_only = cv2.bitwise_and(frame, frame, mask=periphery_mask)

    return overlay, periphery_only

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Try a secondary index (some laptops use 1)
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        poly = trap_points(h, w)
        overlay, periphery = mask_trapezium_out(frame, poly, blackout_alpha=0.75)

        cv2.imshow("Frame (Trapezium masked out)", overlay)
        cv2.imshow("Periphery only (outside trapezium)", periphery)

        # Quit with ESC or q
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
