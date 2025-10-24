import cv2
import matplotlib.pyplot as plt
import pickle
import Funcs_sgbm as F
from pathlib import Path
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Config & Paths
# ──────────────────────────────────────────────────────────────────────────────

CALIB_PATH = "A2_DiveData_4Test/calib_stereo_diver.pkl"
POSE_PATH  = "A2_DiveData_4Test/camera_pose_data.pkl"
LEFT_DIR   = "A2_DiveData_4Test/images_left"
RIGHT_DIR  = "A2_DiveData_4Test/images_right"
TERRAIN_GT = "A2_DiveData_4Test/terrain_data.pkl"   # ground-truth surface (for plotting)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: Load calibration and pose
# ──────────────────────────────────────────────────────────────────────────────

def load_calib_and_pose(calib_path: str, pose_path: str):
    """
    Load stereo calibration (intrinsics/distortion/extrinsics) and per-frame pose.

    Args:
        calib_path: path to calib_stereo_diver.pkl
        pose_path:  path to camera_pose_data.pkl

    Returns:
        calib (dict), pose (dict)
    """
    calib = pickle.load(open(calib_path, "rb"))
    pose  = pickle.load(open(pose_path,  "rb"))
    return calib, pose


# ──────────────────────────────────────────────────────────────────────────────
# Demo / Entry
# ──────────────────────────────────────────────────────────────────────────────

def main():
    calib, pose = load_calib_and_pose(CALIB_PATH, POSE_PATH)

    # Create SGBM depth estimator (keeps your __init__/read_img)
    stereo = F.StereoDepthSGBM(LEFT_DIR, RIGHT_DIR, calib, pose)

    # Load images into memory (49 frames expected)
    stereo.read_img()

    # Build rectification maps and Q from intrinsics/extrinsics
    stereo.build_rectification(alpha=0.0)   # set to 1.0 if you want to keep more FOV

    # Choose a frame index to view (e.g., 0)
    i = 20
    depth_m, disp_f, rectL, rectR = stereo.depth_for_index(i)

    # Plot per-pixel depth
    F.StereoDepthSGBM.show_depth(depth_m, vmin=0.3, vmax=6.0, title=f"Depth (m) — frame {i}")

    # Optional: also visualize disparity
    # disp_vis = F.StereoDepthSGBM.colorize_disparity(disp_f)
    # cv2.imshow(f"Disparity (colored) — frame {i}", disp_vis)
    # cv2.imshow(f"Rectified Left — frame {i}", rectL)
    # cv2.imshow(f"Rectified Right — frame {i}", rectR)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Optional: save depth in millimetres
    # Path("out_depth").mkdir(exist_ok=True)
    # F.StereoDepthSGBM.save_depth_mm(depth_m, f"out_depth/depth_frame_{i:02d}_mm.png")

if __name__ == "__main__":
    main()

