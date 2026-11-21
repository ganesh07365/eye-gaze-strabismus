"""
per_eye_tracker.py - Unified Per-Eye Tracking & Ocular Alignment Engine
========================================================================
BTP Project: Webcam-Based Strabismus Screening & Vision-Therapy Support System

This module provides the core Layer-1 tracking backend for the entire system:
  1. Anatomical scale conversion using the human iris diameter (11.7 mm) as a built-in ruler.
  2. Eyeball spherical kinematics (R = 12.0 mm) converting mm displacements into angular degrees.
  3. Physiological conjugate disparity (nasal-positive cancellation for side gazes).
  4. Inter-ocular misalignment in ophthalmic Prism Diopters (Delta PD) and degrees.
  5. Multi-cue Instantaneous Cover Detection (detects hand, card, or paddle occlusion with zero lag).
  6. Direct landmark reticle visualization on BOTH eyes simultaneously without false occlusion masking.
  7. Saccade-preserving RockSolidFilter to eliminate webcam jitter without lagging fast eye jumps.
"""

import os
import sys
import time
import math
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================================================================
# 1. ANATOMICAL CONSTANTS & CLINICAL GEOMETRY
# ==============================================================================
HUMAN_IRIS_DIAMETER_MM = 11.7   # Average adult horizontal corneal diameter
EYEBALL_RADIUS_MM = 12.0        # Average adult eyeball radius
DEFAULT_VIEWING_DIST_MM = 600.0 # Standard laptop viewing distance
DEFAULT_IPD_MM = 62.0           # Standard inter-pupillary distance

# MediaPipe 478 Landmark Indices (Subject's anatomical perspective):
# Right Eye (Subject's right, camera's left - OD):
RIGHT_OUTER  = 33   # Temporal corner (lateral)
RIGHT_INNER  = 133  # Nasal corner (medial)
RIGHT_TOP    = 159  # Upper eyelid
RIGHT_BOTTOM = 145  # Lower eyelid
RIGHT_IRIS_CENTER = 468
RIGHT_IRIS_BORDER = (469, 470, 471, 472)  # right, top, left, bottom

# Left Eye (Subject's left, camera's right - OS):
LEFT_INNER   = 362  # Nasal corner (medial)
LEFT_OUTER   = 263  # Temporal corner (lateral)
LEFT_TOP     = 386  # Upper eyelid
LEFT_BOTTOM  = 374  # Lower eyelid
LEFT_IRIS_CENTER = 473
LEFT_IRIS_BORDER = (474, 475, 476, 477)  # right, top, left, bottom

NOSE_TIP = 1
MOUTH_RIGHT = 61
MOUTH_LEFT = 291

# Clinical Unit Conversion: 1 degree ~ 1.7455 Prism Diopters
PD_PER_DEGREE = 1.7455


# ==============================================================================
# 2. DATA STRUCTURES
# ==============================================================================
@dataclass
class PerEyeFrameResult:
    timestamp_ms: int = 0
    od_angle: Tuple[float, float] = (0.0, 0.0)    # (yaw_deg, pitch_deg)
    os_angle: Tuple[float, float] = (0.0, 0.0)    # (yaw_deg, pitch_deg)
    delta_deg: Tuple[float, float] = (0.0, 0.0)   # (h_deg, v_deg)
    delta_pd: Tuple[float, float] = (0.0, 0.0)    # (h_pd, v_pd)
    screen_xy: Tuple[int, int] = (0, 0)
    eye_covered: str = "NONE"                     # "NONE", "OD", "OS", "BOTH"
    is_blink_od: bool = False
    is_blink_os: bool = False
    is_blink: bool = False
    data_quality: float = 0.0                     # 0.0 to 1.0
    head_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0) # (yaw, pitch, roll in deg)
    od_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    os_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    od_iris_px: Tuple[float, float] = (0.0, 0.0)
    os_iris_px: Tuple[float, float] = (0.0, 0.0)
    od_aperture: float = 0.0
    os_aperture: float = 0.0
    od_contrast: float = 0.0
    os_contrast: float = 0.0
    od_mm_per_px: float = 0.0
    os_mm_per_px: float = 0.0
    fps: float = 0.0
    face_detected: bool = False


# ==============================================================================
# 3. LOW-LATENCY SACCADE-PRESERVING SIGNAL FILTER
# ==============================================================================
class RockSolidFilter:
    """
    Fast-response hybrid median + adaptive EMA filter.
    Preserves rapid eye saccades while removing high-frequency landmark tremor.
    """
    def __init__(self, init_val=0.0, window_size=3, alpha_still=0.25, alpha_move=0.85, deadband=0.05, saccade_thresh=0.4):
        self.buf = deque([float(init_val)] * window_size, maxlen=window_size)
        self.smooth = float(init_val)
        self.alpha_still = float(alpha_still)
        self.alpha_move = float(alpha_move)
        self.deadband = float(deadband)
        self.saccade_thresh = float(saccade_thresh)
        self.initialized = False

    def reset(self, init_val=0.0):
        self.buf.clear()
        self.smooth = float(init_val)
        self.initialized = False

    def update(self, val: float) -> float:
        val = float(val)
        if not self.initialized:
            self.buf = deque([val] * self.buf.maxlen, maxlen=self.buf.maxlen)
            self.smooth = val
            self.initialized = True
            return self.smooth

        self.buf.append(val)
        med = float(np.median(self.buf))
        diff = med - self.smooth
        abs_diff = abs(diff)

        if abs_diff <= self.deadband:
            return self.smooth

        alpha = self.alpha_move if abs_diff > self.saccade_thresh else self.alpha_still
        self.smooth = self.smooth + alpha * diff
        return self.smooth


# ==============================================================================
# 4. UNIFIED PER-EYE TRACKER CLASS
# ==============================================================================
class PerEyeTracker:
    def __init__(
        self,
        model_asset_path: Optional[str] = None,
        screen_w: int = 1920,
        screen_h: int = 1080,
        mirror_mode: bool = True,
        running_mode: vision.RunningMode = vision.RunningMode.VIDEO,
        auto_pad_crop: bool = False
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.mirror_mode = mirror_mode
        self.running_mode = running_mode
        self.auto_pad_crop = auto_pad_crop

        if model_asset_path is None or not os.path.exists(model_asset_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(script_dir, "..", "models", "face_landmarker.task"),
                os.path.join(script_dir, "face_landmarker.task"),
                "models/face_landmarker.task",
                "face_landmarker.task"
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_asset_path = os.path.abspath(c)
                    break

        if not model_asset_path or not os.path.exists(model_asset_path):
            raise FileNotFoundError(f"MediaPipe face_landmarker.task not found at: {model_asset_path}")

        self.model_asset_path = model_asset_path

        base_options = python.BaseOptions(model_asset_path=self.model_asset_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=self.running_mode,
            num_faces=1,
            output_facial_transformation_matrixes=True
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.filter_od_yaw = RockSolidFilter()
        self.filter_od_pitch = RockSolidFilter()
        self.filter_os_yaw = RockSolidFilter()
        self.filter_os_pitch = RockSolidFilter()
        self.filter_screen_x = RockSolidFilter(init_val=screen_w / 2, alpha_still=0.30, alpha_move=0.90)
        self.filter_screen_y = RockSolidFilter(init_val=screen_h / 2, alpha_still=0.30, alpha_move=0.90)

        # Fast 3-frame rolling queue for instant cover transitions
        self.cover_history = deque(maxlen=3)

        self.start_time = time.time()
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

    def reset_filters(self):
        self.filter_od_yaw.reset()
        self.filter_od_pitch.reset()
        self.filter_os_yaw.reset()
        self.filter_os_pitch.reset()
        self.filter_screen_x.reset(self.screen_w / 2)
        self.filter_screen_y.reset(self.screen_h / 2)
        self.cover_history.clear()

    def _extract_head_pose(self, transform_matrix) -> Tuple[float, float, float]:
        if transform_matrix is None:
            return (0.0, 0.0, 0.0)
        M = np.array(transform_matrix)
        R = M[:3, :3]
        yaw = math.degrees(math.atan2(-R[0, 2], R[2, 2]))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[1, 2]))))
        roll = math.degrees(math.atan2(R[1, 0], R[1, 1]))
        return (yaw, pitch, roll)

    def _extract_single_eye(
        self,
        lms,
        gray: np.ndarray,
        w: int,
        h: int,
        idx_outer: int,
        idx_inner: int,
        idx_top: int,
        idx_bottom: int,
        idx_iris: int,
        idx_iris_borders: Tuple[int, int, int, int],
        is_right_eye: bool
    ) -> Dict[str, Any]:
        """
        Extracts anatomical physical eye rotation, aperture, and texture metrics.
        Always processes in the natural facial frame.
        """
        po = np.array([lms[idx_outer].x * w,  lms[idx_outer].y * h],  dtype=np.float64)
        pi = np.array([lms[idx_inner].x * w,  lms[idx_inner].y * h],  dtype=np.float64)
        pt = np.array([lms[idx_top].x * w,    lms[idx_top].y * h],    dtype=np.float64)
        pb = np.array([lms[idx_bottom].x * w, lms[idx_bottom].y * h], dtype=np.float64)
        p_iris = np.array([lms[idx_iris].x * w, lms[idx_iris].y * h], dtype=np.float64)

        # Iris horizontal diameter in pixels
        p_br = np.array([lms[idx_iris_borders[0]].x * w, lms[idx_iris_borders[0]].y * h])
        p_bl = np.array([lms[idx_iris_borders[2]].x * w, lms[idx_iris_borders[2]].y * h])
        iris_width_px = float(np.linalg.norm(p_br - p_bl))

        canthi_vec = pi - po
        canthi_dist = float(np.linalg.norm(canthi_vec))
        palp_h = float(np.linalg.norm(pt - pb))
        aperture = palp_h / max(canthi_dist, 1.0)

        # Scale factor: mm per pixel
        mm_per_px = HUMAN_IRIS_DIAMETER_MM / max(iris_width_px, 1.0)

        # Canthal unit vector pointing NASALLY (+X along eye axis)
        u_nasal = canthi_vec / max(canthi_dist, 1e-4)

        # Downward unit vector (+Y perpendicular)
        if is_right_eye:
            u_down = np.array([-u_nasal[1], u_nasal[0]], dtype=np.float64)
        else:
            u_down = np.array([u_nasal[1], -u_nasal[0]], dtype=np.float64)

        p_mid = 0.5 * (po + pi)
        d_iris_vec = p_iris - p_mid

        # Displacements in mm
        d_horiz_mm = float(np.dot(d_iris_vec, u_nasal)) * mm_per_px
        d_vert_mm  = float(np.dot(d_iris_vec, u_down)) * mm_per_px

        # Eyeball spherical kinematics: theta = arcsin(clamp(d / R))
        norm_h = max(-0.95, min(0.95, d_horiz_mm / EYEBALL_RADIUS_MM))
        norm_v = max(-0.95, min(0.95, d_vert_mm / EYEBALL_RADIUS_MM))

        yaw_deg = math.degrees(math.asin(norm_h))
        pitch_deg = math.degrees(math.asin(norm_v))

        # Bounding box
        min_x = int(max(0, min(po[0], pi[0]) - canthi_dist * 0.15))
        max_x = int(min(w - 1, max(po[0], pi[0]) + canthi_dist * 0.15))
        min_y = int(max(0, min(pt[1], pb[1]) - palp_h * 0.20))
        max_y = int(min(h - 1, max(pt[1], pb[1]) + palp_h * 0.20))
        box = (min_x, min_y, max_x, max_y)

        crop = gray[min_y:max_y, min_x:max_x]
        if crop.size > 16:
            p10 = float(np.percentile(crop, 10))
            p90 = float(np.percentile(crop, 90))
            contrast = max(0.0, p90 - p10)
            crop_std = float(np.std(crop))
            crop_mean = float(np.mean(crop))
        else:
            contrast = 0.0
            crop_std = 0.0
            crop_mean = 0.0

        # Physiological open eye criteria:
        # 1. Aperture is at least 13% of eye width (relaxed to handle all eye shapes)
        # 2. Eye crop has texture variance (std >= 8.0, eliminating flat occluding cards/paddles)
        # 3. Dynamic contrast >= 14.0 (dark pupil vs bright sclera/skin)
        # 4. Crop mean brightness >= 35.0 (eliminating dark opaque occluders)
        is_open = (aperture >= 0.13) and (crop_std >= 8.0) and (contrast >= 14.0) and (crop_mean >= 35.0)

        return {
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "aperture": aperture,
            "contrast": contrast,
            "crop_std": crop_std,
            "crop_mean": crop_mean,
            "is_open": is_open,
            "iris_width_px": iris_width_px,
            "mm_per_px": mm_per_px,
            "box": box,
            "iris_px": (float(p_iris[0]), float(p_iris[1]))
        }

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> PerEyeFrameResult:
        """
        Main per-frame processing method.
        Always runs MediaPipe on the unmirrored physical camera frame for maximum geometric accuracy.
        """
        h, w, _ = frame_bgr.shape
        if timestamp_ms is None:
            timestamp_ms = int((time.time() - self.start_time) * 1000)

        # FPS calculation
        self.frame_count += 1
        curr_time = time.time()
        if curr_time - self.last_fps_time >= 0.5:
            self.fps = self.frame_count / (curr_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = curr_time

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ROI brightness difference for paddle occluder detection
        l_roi = gray[int(h*0.2):int(h*0.8), :int(w*0.45)]
        r_roi = gray[int(h*0.2):int(h*0.8), int(w*0.55):]
        roi_diff = float(np.mean(r_roi) - np.mean(l_roi)) if l_roi.size > 0 and r_roi.size > 0 else 0.0

        if self.auto_pad_crop:
            pad_t = int(h * 0.45)
            pad_b = int(h * 0.45)
            pad_l = int(w * 0.25)
            pad_r = int(w * 0.25)
            padded = cv2.copyMakeBorder(rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[128, 128, 128])
            ph_h, ph_w = padded.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=padded)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.running_mode == vision.RunningMode.VIDEO:
            detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        else:
            detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks:
            raw_cover = "NONE"
            if roi_diff > 25: raw_cover = "OD"
            elif roi_diff < -25: raw_cover = "OS"
            self.cover_history.append(raw_cover)
            return PerEyeFrameResult(
                timestamp_ms=timestamp_ms,
                eye_covered=raw_cover,
                fps=self.fps,
                face_detected=False
            )

        raw_lms = detection_result.face_landmarks[0]
        if self.auto_pad_crop:
            class UnpadLandmark:
                def __init__(self, x, y, z=0.0):
                    self.x = x
                    self.y = y
                    self.z = z
            lms = [
                UnpadLandmark(
                    x=(lm.x * ph_w - pad_l) / w,
                    y=(lm.y * ph_h - pad_t) / h,
                    z=lm.z
                )
                for lm in raw_lms
            ]
        else:
            lms = raw_lms

        # Extract 3D head pose
        head_matrix = None
        if detection_result.facial_transformation_matrixes:
            head_matrix = detection_result.facial_transformation_matrixes[0]
        head_yaw, head_pitch, head_roll = self._extract_head_pose(head_matrix)

        # Extract Right Eye (OD: Subject's anatomical right eye)
        od = self._extract_single_eye(
            lms, gray, w, h,
            idx_outer=RIGHT_OUTER, idx_inner=RIGHT_INNER,
            idx_top=RIGHT_TOP, idx_bottom=RIGHT_BOTTOM,
            idx_iris=RIGHT_IRIS_CENTER, idx_iris_borders=RIGHT_IRIS_BORDER,
            is_right_eye=True
        )

        # Extract Left Eye (OS: Subject's anatomical left eye)
        os_eye = self._extract_single_eye(
            lms, gray, w, h,
            idx_outer=LEFT_OUTER, idx_inner=LEFT_INNER,
            idx_top=LEFT_TOP, idx_bottom=LEFT_BOTTOM,
            idx_iris=LEFT_IRIS_CENTER, idx_iris_borders=LEFT_IRIS_BORDER,
            is_right_eye=False
        )

        # Robust Multi-cue Occlusion Logic:
        od_open = od["is_open"]
        os_open = os_eye["is_open"]

        if roi_diff > 25:
            raw_cover = "OD"
        elif roi_diff < -25:
            raw_cover = "OS"
        elif od_open and not os_open:
            raw_cover = "OS"
        elif os_open and not od_open:
            raw_cover = "OD"
        elif not od_open and not os_open:
            raw_cover = "BOTH"
        else:
            raw_cover = "NONE"

        # Update 3-frame rolling queue (majority vote gives ~66ms zero-lag response)
        self.cover_history.append(raw_cover)

        od_count = sum(1 for c in self.cover_history if c == "OD")
        os_count = sum(1 for c in self.cover_history if c == "OS")
        both_count = sum(1 for c in self.cover_history if c == "BOTH")

        if od_count >= 2:
            eye_covered = "OD"
        elif os_count >= 2:
            eye_covered = "OS"
        elif both_count >= 2:
            eye_covered = "BOTH"
        else:
            eye_covered = "NONE"

        # Smooth eye angles
        smooth_od_yaw = self.filter_od_yaw.update(od["yaw_deg"])
        smooth_od_pitch = self.filter_od_pitch.update(od["pitch_deg"])
        smooth_os_yaw = self.filter_os_yaw.update(os_eye["yaw_deg"])
        smooth_os_pitch = self.filter_os_pitch.update(os_eye["pitch_deg"])

        # Misalignment Delta Formulation (Nasal-positive cancellation for side gazes):
        # In nasal coordinates: looking right causes OD temporal (-k) and OS nasal (+k).
        # Their sum cancels to 0 for normal conjugate gaze!
        # Inward convergence (Esotropia) -> positive H
        # Outward divergence (Exotropia) -> negative H
        h_deg = float(smooth_od_yaw + smooth_os_yaw)
        v_deg = float(smooth_os_pitch - smooth_od_pitch)

        # Convert to Ophthalmic Prism Diopters: Delta PD = Delta_Deg * 1.7455
        h_pd = float(h_deg * PD_PER_DEGREE)
        v_pd = float(v_deg * PD_PER_DEGREE)

        # Screen gaze coordinates (X, Y)
        if eye_covered == "OD":
            gaze_yaw, gaze_pitch = smooth_os_yaw, smooth_os_pitch
        elif eye_covered == "OS":
            gaze_yaw, gaze_pitch = smooth_od_yaw, smooth_od_pitch
        else:
            gaze_yaw = 0.5 * (smooth_od_yaw - smooth_os_yaw)
            gaze_pitch = 0.5 * (smooth_od_pitch + smooth_os_pitch)

        total_yaw = gaze_yaw + head_yaw * 0.35
        total_pitch = gaze_pitch + head_pitch * 0.35

        raw_screen_x = int(self.screen_w / 2 + math.tan(math.radians(total_yaw)) * (self.screen_w * 0.70))
        raw_screen_y = int(self.screen_h / 2 + math.tan(math.radians(total_pitch)) * (self.screen_h * 0.70))

        screen_x = max(0, min(self.screen_w - 1, int(self.filter_screen_x.update(raw_screen_x))))
        screen_y = max(0, min(self.screen_h - 1, int(self.filter_screen_y.update(raw_screen_y))))

        # Signal quality
        quality = 1.0
        if abs(head_yaw) > 25 or abs(head_pitch) > 25:
            quality *= 0.6
        if not od_open or not os_open:
            quality *= 0.5

        return PerEyeFrameResult(
            timestamp_ms=timestamp_ms,
            od_angle=(smooth_od_yaw, smooth_od_pitch),
            os_angle=(smooth_os_yaw, smooth_os_pitch),
            delta_deg=(h_deg, v_deg),
            delta_pd=(h_pd, v_pd),
            screen_xy=(screen_x, screen_y),
            eye_covered=eye_covered,
            is_blink_od=(not od_open),
            is_blink_os=(not os_open),
            is_blink=bool(not od_open and not os_open),
            data_quality=float(quality),
            head_pose=(head_yaw, head_pitch, head_roll),
            od_box=od["box"],
            os_box=os_eye["box"],
            od_iris_px=od["iris_px"],
            os_iris_px=os_eye["iris_px"],
            od_aperture=od["aperture"],
            os_aperture=os_eye["aperture"],
            od_contrast=od["contrast"],
            os_contrast=os_eye["contrast"],
            od_mm_per_px=od["mm_per_px"],
            os_mm_per_px=os_eye["mm_per_px"],
            fps=self.fps,
            face_detected=True
        )

    def draw_hud(
        self,
        frame: np.ndarray,
        result: PerEyeFrameResult,
        draw_pip: bool = True,
        draw_landmarks: bool = True,
        draw_header: bool = True
    ) -> np.ndarray:
        """
        Renders HUD with instantaneous visual feedback:
        - Top banner with FPS, COVER state, and Misalignment DELTA PD.
        - OD PiP card (User's Right Eye).
        - OS PiP card (User's Left Eye).
        - Direct reticle visualization on BOTH eyes simultaneously without false occlusion masking.
        """
        # In mirror mode, flip canvas for intuitive user viewing
        if self.mirror_mode:
            canvas = cv2.flip(frame, 1)
        else:
            canvas = frame.copy()

        h, w, _ = canvas.shape

        if not result.face_detected:
            cv2.putText(canvas, "NO FACE DETECTED", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
            return canvas

        # 1. Header Information Bar
        if draw_header:
            cv2.rectangle(canvas, (0, 0), (w, 50), (25, 25, 25), -1)
            cv2.putText(canvas, f"FPS: {result.fps:.1f}", (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Cover State Badge
            if result.eye_covered == "NONE":
                cover_color = (0, 255, 0)
            elif result.eye_covered in ("OD", "OS"):
                cover_color = (0, 140, 255)
            else:
                cover_color = (0, 0, 255)

            cv2.putText(canvas, f"COVER: {result.eye_covered}", (150, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cover_color, 2)

            # Misalignment Delta PD Badge
            h_pd, v_pd = result.delta_pd
            pd_color = (0, 255, 0) if abs(h_pd) < 6.0 and abs(v_pd) < 5.0 else (0, 0, 255)
            cv2.putText(canvas, f"DELTA: H={h_pd:+.1f} PD | V={v_pd:+.1f} PD", (380, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pd_color, 2)

            # Quality meter
            cv2.putText(canvas, f"Q: {int(result.data_quality * 100)}%", (w - 120, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Helper to transform coordinate for canvas
        def map_pt(pt_x, pt_y):
            if self.mirror_mode:
                return (w - 1 - int(pt_x), int(pt_y))
            return (int(pt_x), int(pt_y))

        def map_box(box):
            x1, y1, x2, y2 = box
            if self.mirror_mode:
                return (w - 1 - x2, y1, w - 1 - x1, y2)
            return (x1, y1, x2, y2)

        # 2. Dual-Eye Picture-in-Picture (PiP) Cards
        if draw_pip:
            pip_w, pip_h = 160, 100

            # OD PiP (User's Right Eye -> displayed on user's right in mirror mode)
            od_x1, od_y1, od_x2, od_y2 = map_box(result.od_box)
            if od_x2 > od_x1 and od_y2 > od_y1:
                crop_od = canvas[od_y1:od_y2, od_x1:od_x2]
                if crop_od.size > 0:
                    pip_od = cv2.resize(crop_od, (pip_w, pip_h))
                    pos_od_x = (w - pip_w - 15) if self.mirror_mode else 15
                    canvas[60:60 + pip_h, pos_od_x:pos_od_x + pip_w] = pip_od
                    od_border = (0, 0, 255) if result.eye_covered == "OD" else (0, 255, 0)
                    cv2.rectangle(canvas, (pos_od_x, 60), (pos_od_x + pip_w, 60 + pip_h), od_border, 2)
                    if result.eye_covered == "OD":
                        cv2.rectangle(canvas, (pos_od_x, 60 + pip_h - 26), (pos_od_x + pip_w, 60 + pip_h), (0, 0, 200), -1)
                        cv2.putText(canvas, "OD COVERED", (pos_od_x + 20, 60 + pip_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:
                        cv2.putText(canvas, f"OD: {result.od_angle[0]:+.1f} deg", (pos_od_x + 5, 60 + pip_h + 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # OS PiP (User's Left Eye -> displayed on user's left in mirror mode)
            os_x1, os_y1, os_x2, os_y2 = map_box(result.os_box)
            if os_x2 > os_x1 and os_y2 > os_y1:
                crop_os = canvas[os_y1:os_y2, os_x1:os_x2]
                if crop_os.size > 0:
                    pip_os = cv2.resize(crop_os, (pip_w, pip_h))
                    pos_os_x = 15 if self.mirror_mode else (w - pip_w - 15)
                    canvas[60:60 + pip_h, pos_os_x:pos_os_x + pip_w] = pip_os
                    os_border = (0, 0, 255) if result.eye_covered == "OS" else (0, 255, 0)
                    cv2.rectangle(canvas, (pos_os_x, 60), (pos_os_x + pip_w, 60 + pip_h), os_border, 2)
                    if result.eye_covered == "OS":
                        cv2.rectangle(canvas, (pos_os_x, 60 + pip_h - 26), (pos_os_x + pip_w, 60 + pip_h), (0, 0, 200), -1)
                        cv2.putText(canvas, "OS COVERED", (pos_os_x + 20, 60 + pip_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:
                        cv2.putText(canvas, f"OS: {result.os_angle[0]:+.1f} deg", (pos_os_x + 5, 60 + pip_h + 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 3. Direct Iris Reticles on Main Face (ALWAYS drawn on both eyes!)
        if draw_landmarks:
            # OD iris reticle (Red dot + yellow ring)
            if result.od_iris_px[0] > 0 and result.od_iris_px[1] > 0:
                od_cx, od_cy = map_pt(result.od_iris_px[0], result.od_iris_px[1])
                cv2.circle(canvas, (od_cx, od_cy), 4, (0, 0, 255), -1)
                cv2.circle(canvas, (od_cx, od_cy), 10, (0, 255, 255), 1)

            # OS iris reticle (Red dot + yellow ring)
            if result.os_iris_px[0] > 0 and result.os_iris_px[1] > 0:
                os_cx, os_cy = map_pt(result.os_iris_px[0], result.os_iris_px[1])
                cv2.circle(canvas, (os_cx, os_cy), 4, (0, 0, 255), -1)
                cv2.circle(canvas, (os_cx, os_cy), 10, (0, 255, 255), 1)

        return canvas


# ==============================================================================
# 5. STANDALONE CLI TEST RUNNER
# ==============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified Per-Eye Tracker Diagnostic Runner")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file for offline testing")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirror mode (auto-enabled for webcam)")
    args = parser.parse_args()

    use_mirror = (args.video is None) and (not args.no_mirror)
    print(f"[INITIALIZING] Starting PerEyeTracker (Mirror Mode: {use_mirror})...")
    tracker = PerEyeTracker(mirror_mode=use_mirror)
    print("[READY] PerEyeTracker initialized successfully.")

    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"[INPUT] Reading video: {args.video}")
    else:
        cap = cv2.VideoCapture(args.webcam)
        print(f"[INPUT] Reading webcam: {args.webcam}")

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        sys.exit(1)

    print("\n--- CONTROLS ---")
    print("  [q] Quit")
    print("  [r] Reset filters")
    print("  [m] Toggle mirror mode")
    print("----------------\n")

    start_t = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            if args.video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        timestamp_ms = int((time.time() - start_t) * 1000)
        result = tracker.process_frame(frame, timestamp_ms=timestamp_ms)
        hud_frame = tracker.draw_hud(frame, result)

        cv2.imshow("PerEyeTracker Live Engine (Step 1)", hud_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset_filters()
            print("[INFO] Filters reset.")
        elif key == ord('m'):
            tracker.mirror_mode = not tracker.mirror_mode
            print(f"[INFO] Mirror mode set to: {tracker.mirror_mode}")

    cap.release()
    cv2.destroyAllWindows()
