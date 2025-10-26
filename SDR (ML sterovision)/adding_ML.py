import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import cv2.ximgproc as xip
from ultralytics import YOLO

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
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
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
    with np.errstate(divide='ignore'):
        depth = focal_length_px * baseline_m / disp
        depth[disp <= 0] = np.nan
    return depth

# === Load YOLOv8 model ===
yolo_model = YOLO("yolov8x.pt")  # smallest YOLOv8 model for speed

# Class weights for edge visualization
class_weight = {'person': 1.0, 'pole': 1.0, 'car': 1.0, 'tree': 1.0}

# === Process first 3 image pairs ===
for left_path, right_path, gt_path in zip(left_images[4:6], right_images[4:6], gt_files[4:6]):
    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)

    # Grayscale + CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    grayL = clahe.apply(cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY))
    grayR = clahe.apply(cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY))

    # --- Compute disparities ---
    disp_left = left_matcher.compute(grayL, grayR).astype(np.float32) / 16.0
    disp_right = right_matcher.compute(grayR, grayL).astype(np.float32) / 16.0
    filtered_disp = wls_filter.filter(disp_left, grayL, None, disp_right)
    disp_pred = np.where(filtered_disp > 0, filtered_disp, np.nan)

    # Mask sky region (top 1/4)
    sky_cut = disp_pred.shape[0] // 4
    disp_pred[:sky_cut, :] = np.nan

    # --- Convert disparity to depth ---
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
    depth_edges = np.nan_to_num(depth_edges, nan=0.0)

    # --- Load ground truth and evaluate ---
    disp_gt = load_ground_truth_disp(gt_path)
    rmse, bad_px = evaluate_disparity(disp_pred, disp_gt)
    print(f"{os.path.basename(left_path)} → RMSE={rmse:.2f} px | Bad-px={bad_px:.2f}%")

    # --- YOLO object detection ---
    results = yolo_model.predict(source=cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
    detections = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]

    # Overlay edge-enhanced depth in bounding boxes
    overlay = imgL.copy()
    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box)
        label = results[0].names[int(results[0].boxes.cls[i])]
        bbox_depth = depth_map[y1:y2, x1:x2]
        avg_depth = np.nanmean(bbox_depth) if not np.isnan(bbox_depth).all() else np.nan

        bbox_edges = depth_edges[y1:y2, x1:x2]
        weight = class_weight.get(label, 0.5)
        for c in range(3):
            overlay[y1:y2, x1:x2, c] = np.maximum(
                overlay[y1:y2, x1:x2, c],
                (bbox_edges * 255 * weight).astype(np.uint8)
            )

        # Draw bounding box + label + depth
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(overlay, f"{label} {avg_depth:.1f}m" if not np.isnan(avg_depth) else label,
                    (x1, max(y1-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # --- Visualization: YOLO + edge-enhanced depth ---
    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Edge-Enhanced Depth + YOLO Objects ({os.path.basename(left_path)})")
    plt.axis("off")
    plt.show()

    # --- Visualization: edge-enhanced depth ---
    plt.figure(figsize=(10,6))
    plt.imshow(depth_edges, cmap='plasma', vmin=vmin, vmax=vmax)
    plt.title("Edge-Enhanced Depth")
    plt.colorbar(label='Depth [m]')
    plt.axis("off")
    plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from ultralytics import YOLO  # pip install ultralytics
# import glob, os

# # === CONFIG ===
# DATA_ROOT = "data_scene_flow/training"
# baseline_m = 0.54
# focal_length_px = 721.5377
# max_depth_clip = 30  # meters for visualization

# # === Load image paths ===
# left_dir = os.path.join(DATA_ROOT, "image_2")
# right_dir = os.path.join(DATA_ROOT, "image_3")
# left_images = sorted(glob.glob(os.path.join(left_dir, "*.png")))
# right_images = sorted(glob.glob(os.path.join(right_dir, "*.png")))

# # --- Stereo matcher setup (SGBM + WLS) ---
# import cv2.ximgproc as xip
# window_size = 5
# numDisparities = 384
# minDisparity = 0

# left_matcher = cv2.StereoSGBM_create(
#     minDisparity=minDisparity,
#     numDisparities=numDisparities,
#     blockSize=window_size,
#     P1=8*3*window_size**2,
#     P2=32*3*window_size**2,
#     disp12MaxDiff=1,
#     uniquenessRatio=5,
#     speckleWindowSize=100,
#     speckleRange=32
# )
# right_matcher = xip.createRightMatcher(left_matcher)
# wls_filter = xip.createDisparityWLSFilter(matcher_left=left_matcher)
# wls_filter.setLambda(8000.0)
# wls_filter.setSigmaColor(1.5)

# # --- Depth conversion ---
# def disparity_to_depth(disp):
#     with np.errstate(divide='ignore'):
#         depth = focal_length_px * baseline_m / disp
#         depth[disp <= 0] = np.nan
#     return depth

# # --- Load YOLOv8 pretrained model ---
# yolo_model = YOLO("yolov8n.pt")  # or yolov8s.pt for slightly bigger model

# # === Process first few image pairs ===
# for left_path, right_path in zip(left_images[:3], right_images[:3]):
#     # Load and preprocess
#     imgL = cv2.imread(left_path)
#     imgR = cv2.imread(right_path)
#     grayL = cv2.equalizeHist(cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY))
#     grayR = cv2.equalizeHist(cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY))

#     # Compute disparity
#     disp_left = left_matcher.compute(grayL, grayR).astype(np.float32)/16.0
#     disp_right = right_matcher.compute(grayR, grayL).astype(np.float32)/16.0
#     filtered_disp = wls_filter.filter(disp_left, grayL, None, disp_right)
#     disp_pred = np.where(filtered_disp > 0, filtered_disp, np.nan)

#     # Mask sky region
#     sky_cut = disp_pred.shape[0] // 4
#     disp_pred[:sky_cut, :] = np.nan

#     # Convert to depth map
#     depth_map = disparity_to_depth(disp_pred)
#     depth_vis = np.clip(depth_map, 0, max_depth_clip)

#     # --- Run YOLO object detection ---
#     results = yolo_model(imgL)  # returns a list of detections
#     detections = results[0].boxes.xyxy  # xmin, ymin, xmax, ymax

#     # --- Draw results and estimate distance ---
#     img_annot = imgL.copy()
#     for box in detections:
#         xmin, ymin, xmax, ymax = map(int, box)
#         # median depth inside the bounding box
#         obj_depth = depth_map[ymin:ymax, xmin:xmax]
#         median_distance = np.nanmedian(obj_depth)
#         if np.isnan(median_distance):
#             continue  # skip if no valid depth

#         # Draw rectangle + distance label
#         cv2.rectangle(img_annot, (xmin, ymin), (xmax, ymax), (0, 165, 255), 2)
#         cv2.putText(
#             img_annot, f"{median_distance:.1f}m",
#             (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1
#         )

#     # --- Show results ---
#     plt.figure(figsize=(12,6))
#     plt.imshow(cv2.cvtColor(img_annot, cv2.COLOR_BGR2RGB))
#     plt.title("YOLO Detection + Estimated Depth")
#     plt.axis("off")
#     plt.show()

#     plt.figure(figsize=(12,6))
#     plt.imshow(depth_vis, cmap='plasma', vmin=np.nanmin(depth_vis), vmax=np.nanmax(depth_vis))
#     plt.title("Depth Map (meters)")
#     plt.colorbar(label="Depth [m]")
#     plt.axis("off")
#     plt.show()
