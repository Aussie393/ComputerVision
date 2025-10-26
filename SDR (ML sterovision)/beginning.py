import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import cv2.ximgproc as xip

# === CONFIG ===
DATA_ROOT = "data_scene_flow/training"
baseline_m = 0.54           # KITTI stereo baseline
focal_length_px = 721.5377  # KITTI calibration
bad_px_threshold = 3.0      # pixels
max_depth_clip = 30         # meters for visualization

# === Load image paths ===
left_dir  = os.path.join(DATA_ROOT, "image_2")
right_dir = os.path.join(DATA_ROOT, "image_3")
gt_dir    = os.path.join(DATA_ROOT, "disp_noc_0")

left_images  = sorted(glob.glob(os.path.join(left_dir, "*.png")))
right_images = sorted(glob.glob(os.path.join(right_dir, "*.png")))
gt_files     = sorted(glob.glob(os.path.join(gt_dir, "*.png")))

print(f"Found {len(left_images)} training pairs")

# === Stereo matcher setup ===
window_size = 9
numDisparities = 256  # must be multiple of 16
minDisparity = 0

left_matcher = cv2.StereoSGBM_create(
    minDisparity=minDisparity,
    numDisparities=numDisparities,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=5,
    speckleWindowSize=100,
    speckleRange=32
)
right_matcher = xip.createRightMatcher(left_matcher)

# WLS filter for smooth disparity
wls_filter = xip.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000.0)
wls_filter.setSigmaColor(1.5)

# === Helper functions ===
def load_ground_truth_disp(path):
    disp_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
    disp_gt[disp_gt <= 0] = np.nan
    return disp_gt

def evaluate_disparity(disp_pred, disp_gt):
    mask = ~np.isnan(disp_gt)
    valid_pred = disp_pred[mask]
    valid_gt = disp_gt[mask]
    error = valid_pred - valid_gt
    rmse = np.sqrt(np.nanmean(error**2))
    bad_px = np.mean(np.abs(error) > bad_px_threshold) * 100.0
    return rmse, bad_px

def disparity_to_depth(disp):
    """Convert disparity map to depth in meters."""
    with np.errstate(divide='ignore'):
        depth = focal_length_px * baseline_m / disp
        depth[disp <= 0] = np.nan
    return depth

# === Process first 3 image pairs ===
for left_path, right_path, gt_path in zip(left_images[:3], right_images[:3], gt_files[:3]):
    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    grayL = clahe.apply(cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY))
    grayR = clahe.apply(cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY))

    # --- Compute disparities ---
    disp_left = left_matcher.compute(grayL, grayR).astype(np.float32) / 16.0
    disp_right = right_matcher.compute(grayR, grayL).astype(np.float32) / 16.0
    filtered_disp = wls_filter.filter(disp_left, grayL, None, disp_right)
    disp_pred = np.where(filtered_disp > 0, filtered_disp, np.nan)

    # --- Mask sky region (top 1/4) ---
    sky_cut = disp_pred.shape[0] // 4
    disp_pred[:sky_cut, :] = np.nan

    # --- Convert to depth ---
    depth_map = disparity_to_depth(disp_pred)
    depth_vis = np.clip(depth_map, 0, max_depth_clip)

    # Auto-contrast
    valid = ~np.isnan(depth_vis)
    if np.any(valid):
        vmin = np.nanmin(depth_vis[valid])
        vmax = np.nanmax(depth_vis[valid])
    else:
        vmin, vmax = 0, max_depth_clip

    # --- Edge detection to highlight small objects ---
    edges = cv2.Canny(grayL, 50, 150)
    edges_normalized = edges / 255.0
    depth_edges = depth_vis * edges_normalized
    # Optional: enhance contrast
    depth_edges = np.nan_to_num(depth_edges, nan=0.0)

    # --- Load ground truth ---
    disp_gt = load_ground_truth_disp(gt_path)

    # --- Evaluate ---
    rmse, bad_px = evaluate_disparity(disp_pred, disp_gt)
    print(f"{os.path.basename(left_path)} → RMSE={rmse:.2f} px | Bad-px={bad_px:.2f}%")

# --- Visualization ---
    # Left image
    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
    plt.title("Left Image")
    plt.axis("off")

    # Depth map (auto-contrast)
    plt.figure(figsize=(10,6))
    plt.imshow(depth_vis, cmap='plasma', vmin=vmin, vmax=vmax)
    plt.title("Depth Map (meters)")
    plt.colorbar(label='Depth [m]')
    plt.axis("off")

    # Edge-enhanced depth
    plt.figure(figsize=(10,6))
    plt.imshow(depth_edges, cmap='plasma', vmin=vmin, vmax=vmax)
    plt.title("Edge-Enhanced Depth")
    plt.colorbar(label='Depth [m]')
    plt.axis("off")
    plt.show()



# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import glob, os
# import cv2.ximgproc as xip

# # === CONFIG ===
# DATA_ROOT = "data_scene_flow/training"
# baseline_m = 0.54           # KITTI stereo baseline
# focal_length_px = 721.5377  # KITTI calibration
# bad_px_threshold = 3.0

# # === Load image paths ===
# left_dir  = os.path.join(DATA_ROOT, "image_2")
# right_dir = os.path.join(DATA_ROOT, "image_3")
# gt_dir    = os.path.join(DATA_ROOT, "disp_noc_0")

# left_images  = sorted(glob.glob(os.path.join(left_dir, "*.png")))
# right_images = sorted(glob.glob(os.path.join(right_dir, "*.png")))
# gt_files     = sorted(glob.glob(os.path.join(gt_dir, "*.png")))

# print(f"Found {len(left_images)} training pairs")

# # === Stereo matcher setup (tuned for close cars) ===
# window_size = 5
# numDisparities = 384  # multiple of 16, enough for near objects
# minDisparity = 0

# left_matcher = cv2.StereoSGBM_create(
#     minDisparity=minDisparity,
#     numDisparities=numDisparities,
#     blockSize=window_size,
#     P1=8 * 3 * window_size**2,
#     P2=32 * 3 * window_size**2,
#     disp12MaxDiff=1,
#     uniquenessRatio=5,
#     speckleWindowSize=100,
#     speckleRange=32
# )

# right_matcher = xip.createRightMatcher(left_matcher)

# # WLS filter
# wls_filter = xip.createDisparityWLSFilter(matcher_left=left_matcher)
# wls_filter.setLambda(8000.0)
# wls_filter.setSigmaColor(1.5)

# # --- Functions ---
# def load_ground_truth_disp(path):
#     disp_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
#     disp_gt[disp_gt <= 0] = np.nan
#     return disp_gt

# def evaluate_disparity(disp_pred, disp_gt):
#     mask = ~np.isnan(disp_gt)
#     valid_pred = disp_pred[mask]
#     valid_gt = disp_gt[mask]
#     error = valid_pred - valid_gt
#     rmse = np.sqrt(np.nanmean(error**2))
#     bad_px = np.mean(np.abs(error) > bad_px_threshold) * 100.0
#     return rmse, bad_px

# def disparity_to_depth(disp):
#     with np.errstate(divide='ignore'):
#         depth = focal_length_px * baseline_m / disp
#         depth[disp <= 0] = np.nan
#     return depth

# # --- Process first few sample images ---
# for left_path, right_path, gt_path in zip(left_images[:3], right_images[:3], gt_files[:3]):
#     imgL = cv2.imread(left_path)
#     imgR = cv2.imread(right_path)

#     grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
#     grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
#     grayL = cv2.equalizeHist(grayL)
#     grayR = cv2.equalizeHist(grayR)

#     # --- Compute disparities ---
#     disp_left = left_matcher.compute(grayL, grayR).astype(np.float32) / 16.0
#     disp_right = right_matcher.compute(grayR, grayL).astype(np.float32) / 16.0
#     filtered_disp = wls_filter.filter(disp_left, grayL, None, disp_right)
#     disp_pred = np.where(filtered_disp > 0, filtered_disp, np.nan)

#     # Mask sky
#     sky_cut = disp_pred.shape[0] // 4
#     disp_pred[:sky_cut, :] = np.nan

#     # Depth map
#     depth_map = disparity_to_depth(disp_pred)

#     # Load ground truth
#     disp_gt = load_ground_truth_disp(gt_path)

#     # Evaluate
#     rmse, bad_px = evaluate_disparity(disp_pred, disp_gt)
#     print(f"{os.path.basename(left_path)} → RMSE={rmse:.2f} px | Bad-px={bad_px:.2f}%")

#     # --- Visualization ---
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))

#     # Left image
#     axes[0].imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
#     axes[0].set_title("Left Image")
#     axes[0].axis("off")

#     # Depth map (clipped and with plasma colormap)
#     max_depth_vis = 50  # meters
#     im1 = axes[1].imshow(np.clip(depth_map, 0, max_depth_vis), cmap='plasma')
#     axes[1].set_title("Depth Map (meters, clipped)")
#     axes[1].axis("off")
#     plt.colorbar(im1, ax=axes[1], fraction=0.046, label='Depth [m]')

#     # Error map
#     overlay = np.abs(disp_pred - disp_gt)
#     im2 = axes[2].imshow(overlay, cmap='inferno', vmax=bad_px_threshold)
#     axes[2].set_title("Error Map (abs diff)")
#     axes[2].axis("off")
#     plt.colorbar(im2, ax=axes[2], fraction=0.046)

#     plt.suptitle(f"RMSE={rmse:.2f} px | Bad-px={bad_px:.2f}%", fontsize=12)
#     plt.tight_layout()
#     plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import glob, os
# import cv2.ximgproc as xip

# # === CONFIG ===
# DATA_ROOT = "data_scene_flow/training"
# baseline_m = 0.54           # KITTI stereo baseline in metres
# focal_length_px = 721.5377  # typical KITTI calibration
# bad_px_threshold = 3.0      # pixels

# # === Load image paths ===
# left_dir  = os.path.join(DATA_ROOT, "image_2")
# right_dir = os.path.join(DATA_ROOT, "image_3")
# gt_dir    = os.path.join(DATA_ROOT, "disp_noc_0")

# left_images  = sorted(glob.glob(os.path.join(left_dir, "*.png")))
# right_images = sorted(glob.glob(os.path.join(right_dir, "*.png")))
# gt_files     = sorted(glob.glob(os.path.join(gt_dir, "*.png")))

# print(f"Found {len(left_images)} training pairs")

# # === Stereo matchers (created once) ===
# window_size = 3
# numDisparities = 124  # must be multiple of 16
# minDisparity = 0

# left_matcher = cv2.StereoSGBM_create(
#     minDisparity=minDisparity,
#     numDisparities=numDisparities,
#     blockSize=window_size,
#     P1 = 8 * 3 * window_size** 2,
#     P2 = 24 * 3 * window_size** 2,
#     disp12MaxDiff=1,
#     uniquenessRatio=5,
#     speckleWindowSize=100,
#     speckleRange=32
# )

# right_matcher = xip.createRightMatcher(left_matcher)

# # === WLS filter setup ===
# wls_filter = xip.createDisparityWLSFilter(matcher_left=left_matcher)
# wls_filter.setLambda(8000.0)   # smoothness strength
# wls_filter.setSigmaColor(2)  # color sensitivity

# # === Functions ===
# def load_ground_truth_disp(path):
#     """KITTI ground truth disparity is stored as 16-bit PNG, scaled by 256."""
#     disp_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
#     disp_gt[disp_gt <= 0] = np.nan  # invalid pixels
#     return disp_gt

# def evaluate_disparity(disp_pred, disp_gt):
#     """Compute RMSE and bad-pixel rate."""
#     mask = ~np.isnan(disp_gt)
#     valid_pred = disp_pred[mask]
#     valid_gt = disp_gt[mask]
#     error = valid_pred - valid_gt
#     rmse = np.sqrt(np.nanmean(error**2))
#     bad_px = np.mean(np.abs(error) > bad_px_threshold) * 100.0
#     return rmse, bad_px

# # === Process a few sample pairs ===
# for i, (left_path, right_path, gt_path) in enumerate(zip(left_images[:3], right_images[:3], gt_files[:3])):
#     imgL = cv2.imread(left_path)
#     imgR = cv2.imread(right_path)
#     grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
#     grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
#     grayL = cv2.equalizeHist(grayL)
#     grayR = cv2.equalizeHist(grayR)

#     # --- Compute disparities ---
#     disp_left = left_matcher.compute(grayL, grayR).astype(np.float32) / 16.0
#     disp_right = right_matcher.compute(grayR, grayL).astype(np.float32) / 16.0

#     # --- Apply WLS filter ---
#     filtered_disp = wls_filter.filter(disp_left, grayL, None, disp_right)
#     disp_pred = np.where(filtered_disp > 0, filtered_disp, np.nan)

#     # --- Mask the sky region (top 1/4 of image) ---
#     sky_cut = disp_pred.shape[0] // 4
#     disp_pred[:sky_cut, :] = np.nan

#     # --- Load and mask ground truth disparity ---
#     disp_gt = load_ground_truth_disp(gt_path)

#     # --- Evaluate disparity ---
#     rmse, bad_px = evaluate_disparity(disp_pred, disp_gt)

#     print(f"{os.path.basename(left_path)} → RMSE={rmse:.2f} px, Bad-px={bad_px:.2f}%")

#     # --- Visual comparison ---
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     axes[0].imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
#     axes[0].set_title("Left Image")
#     axes[0].axis("off")

#     im1 = axes[1].imshow(disp_pred, cmap='plasma')
#     axes[1].set_title("Filtered Disparity (sky masked)")
#     axes[1].axis("off")
#     plt.colorbar(im1, ax=axes[1], fraction=0.046)

#     overlay = np.abs(disp_pred - disp_gt)
#     im2 = axes[2].imshow(overlay, cmap='inferno', vmax=bad_px_threshold)
#     axes[2].set_title("Error Map (abs diff)")
#     axes[2].axis("off")
#     plt.colorbar(im2, ax=axes[2], fraction=0.046)

#     plt.suptitle(f"RMSE = {rmse:.2f}px | Bad-pixel = {bad_px:.2f}%", fontsize=12)
#     plt.tight_layout()
#     plt.show()
