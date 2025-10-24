# A2Funcs.py
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class StereoDepthSGBM:
    """
    Minimal stereo pipeline:
      - keep __init__ and read_img (as provided)
      - rectification from intrinsics/extrinsics (no .npz)
      - per-pixel disparity (SGBM) -> depth via Q
      - quick plotting utilities
    """

    # ──────────────────────────────────────────────────────────────────────────
    # KEEP: __init__ (unchanged from your snippet, only added minor assignments)
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(self, left_path, right_path, calib, pose):
        """
        Args:
            left_path, right_path: folders containing left/right images.
            calib: dict with Kl, Dl, Kr, Dr, R (left->right), t (left->right).
            pose:  dict with per-frame world->camera extrinsics and filenames.
        """
        self.left_dir = Path(left_path)
        self.right_dir = Path(right_path)

        # Pose (per-frame world->camera)
        self.Rc = np.asarray(pose["R"])                  # shape ~ (3, 3, 49)
        self.tc = np.asarray(pose["t"])                  # shape ~ (3, 49)
        self.filenames_left  = list(pose["filenames_left"])
        self.filenames_right = list(pose["filenames_right"])

        # Calibration (intrinsics & left→right extrinsics)
        self.Kl = np.asarray(calib["Kl"], dtype=np.float64)            # (3,3) left intrinsics
        self.Dl = np.asarray(calib["Dl"], dtype=np.float64).ravel()    # (5,)  left distortion
        self.Kr = np.asarray(calib["Kr"], dtype=np.float64)            # (3,3) right intrinsics
        self.Dr = np.asarray(calib["Dr"], dtype=np.float64).ravel()    # (5,)  right distortion
        self.R_lr  = np.asarray(calib["R"],  dtype=np.float64)         # (3,3) rotation left->right
        self.t_lr  = np.asarray(calib["t"],  dtype=np.float64).reshape(3,1)  # (3,1) translation left->right

        # Will be filled after read_img() + build_rectification()
        self.imgsL = None
        self.imgsR = None
        self.mapLx = self.mapLy = self.mapRx = self.mapRy = None
        self.Q = None

        # Default SGBM params (tunable)
        self.min_disp = 0
        self.num_disp = 128   # must be multiple of 16
        self.block_size = 5   # odd: 3..11 typical
        self.use_wls = True   # set False if ximgproc missing

        # Create matchers now (can be re-created if you later change params)
        self._create_matchers()

    # ──────────────────────────────────────────────────────────────────────────
    # KEEP: read_img (unchanged)
    # ──────────────────────────────────────────────────────────────────────────
    def read_img(self):
        """
        Load the left/right image stacks into memory.

        Raises:
            FileNotFoundError if any expected file is missing.
            AssertionError if the expected count (49) is not met.
        """
        # Read left images
        imgsL = []
        for fn in self.filenames_left:
            img = cv2.imread(str(self.left_dir / fn), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Missing: {self.left_dir / fn}")
            imgsL.append(img)
        assert len(imgsL) == 49, f"Expected 49, got {len(imgsL)}"
        self.imgsL = np.stack(imgsL, axis=0)   # (49,H,W,3) uint8

        # Read right images
        imgsR = []
        for fn in self.filenames_right:
            img = cv2.imread(str(self.right_dir / fn), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Missing: {self.right_dir / fn}")
            imgsR.append(img)
        assert len(imgsR) == 49, f"Expected 49, got {len(imgsR)}"
        self.imgsR = np.stack(imgsR, axis=0)   # (49,H,W,3) uint8

        return

    # ──────────────────────────────────────────────────────────────────────────
    # Rectification & Matchers
    # ──────────────────────────────────────────────────────────────────────────
    def build_rectification(self, alpha: float = 0.0):
        """
        Build rectification maps and Q using intrinsics and left->right extrinsics.
        Call this AFTER read_img(), since we need the image size.
        """
        assert self.imgsL is not None and self.imgsR is not None, "Call read_img() first."
        H, W = self.imgsL.shape[1:3]
        image_size = (W, H)

        RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(
            self.Kl, self.Dl, self.Kr, self.Dr, image_size,
            self.R_lr, self.t_lr, alpha=alpha, flags=cv2.CALIB_ZERO_DISPARITY
        )
        self.Q = Q

        self.mapLx, self.mapLy = cv2.initUndistortRectifyMap(
            self.Kl, self.Dl, RL, PL, image_size, cv2.CV_32FC1
        )
        self.mapRx, self.mapRy = cv2.initUndistortRectifyMap(
            self.Kr, self.Dr, RR, PR, image_size, cv2.CV_32FC1
        )

    def _create_matchers(self):
        """Create SGBM + optional WLS right matcher."""
        nd = int(self.num_disp // 16) * 16
        if nd <= 0: nd = 16
        self.num_disp = nd

        P1 = 8 * 3 * self.block_size**2
        P2 = 32 * 3 * self.block_size**2

        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=self.block_size,
            P1=P1, P2=P2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self.right_matcher = None
        self.wls = None
        if self.use_wls and hasattr(cv2, "ximgproc"):
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.wls = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
            self.wls.setLambda(8000.0)
            self.wls.setSigmaColor(1.2)
        elif self.use_wls:
            # ximgproc not available → fall back gracefully
            print("[StereoDepthSGBM] Warning: cv2.ximgproc not found. WLS disabled.")
            self.use_wls = False

    # ──────────────────────────────────────────────────────────────────────────
    # Core per-pixel depth
    # ──────────────────────────────────────────────────────────────────────────
    def _rectify_pair(self, left_bgr, right_bgr):
        assert self.mapLx is not None, "Call build_rectification() first."
        rL = cv2.remap(left_bgr,  self.mapLx, self.mapLy, interpolation=cv2.INTER_LINEAR)
        rR = cv2.remap(right_bgr, self.mapRx, self.mapRy, interpolation=cv2.INTER_LINEAR)
        return rL, rR

    def _compute_disparity(self, rectL, rectR):
        gL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        dispL = self.left_matcher.compute(gL, gR).astype(np.int16)

        if self.use_wls and self.right_matcher is not None and self.wls is not None:
            dispR = self.right_matcher.compute(gR, gL).astype(np.int16)
            disp = self.wls.filter(dispL, rectL, disparity_map_right=dispR)
        else:
            disp = dispL

        disp_f = disp.astype(np.float32) / 16.0
        disp_f[disp_f <= 0.0] = np.nan
        return disp_f

    def _disparity_to_depth(self, disp_f):
        assert self.Q is not None, "Q not set. Call build_rectification()."
        disp_safe = np.where(np.isfinite(disp_f), disp_f, 0.0).astype(np.float32)
        pts3d = cv2.reprojectImageTo3D(disp_safe, self.Q)  # (H,W,3)
        depth = pts3d[..., 2]
        depth[(~np.isfinite(disp_f)) | (depth <= 0)] = np.nan
        return depth

    def depth_for_index(self, i: int):
        """
        Compute per-pixel depth for frame index i.
        Returns: depth_m (H,W float32), disp_f (H,W float32), rectL, rectR (BGR)
        """
        left  = self.imgsL[i]
        right = self.imgsR[i]
        rectL, rectR = self._rectify_pair(left, right)
        disp_f = self._compute_disparity(rectL, rectR)
        depth_m = self._disparity_to_depth(disp_f)
        return depth_m, disp_f, rectL, rectR

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities (plot/save)
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def show_depth(depth_m, vmin=None, vmax=None, title="Depth (m)"):
        plt.figure(figsize=(8, 3.6))
        im = plt.imshow(depth_m, cmap="plasma", vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.axis("off")
        cbar = plt.colorbar(im, fraction=0.025, pad=0.02)
        cbar.set_label("meters")
        plt.show()

    @staticmethod
    def save_depth_mm(depth_m, out_path):
        depth_mm = (depth_m * 1000.0).astype(np.float32)
        depth_mm[~np.isfinite(depth_mm)] = 0.0
        depth_u16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(out_path), depth_u16)

    @staticmethod
    def colorize_disparity(disp_f):
        d = np.nan_to_num(disp_f, nan=0.0)
        dmin, dmax = float(np.min(d)), float(np.max(d))
        if dmax - dmin < 1e-6: dmax = dmin + 1.0
        norm = ((d - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)
