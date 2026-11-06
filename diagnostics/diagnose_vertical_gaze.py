"""
diagnose_vertical_gaze.py  -  VERSION 2 (corrected FACE_MODEL)
================================================================
Same experiment as v1 but now:
  - Version A: old FACE_MODEL (mouth Y=30mm) - inline, unchanged baseline
  - Version B: fixed FACE_MODEL (mouth Y=65mm) - imported from core/normalize.py

Both versions run on EVERY live frame using the same CNN checkpoint.
Saves everything to diagnostics/results/vertical_gaze_diag_v2/.
"""

import sys
import os
import time
import ctypes
import csv
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn

# Import the FIXED normalize (mouth Y=65mm)
from normalize import (
    normalize_single_eye,
    FACE_MODEL as FACE_MODEL_FIXED,
    rodrigues_to_matrix,
    get_eye_center_3d,
    NORM_DISTANCE,
    NORM_FOCAL_LEN,
    EYE_PATCH_SIZE,
)

# ── OLD face model (mouth Y=30mm) kept inline as Version A baseline ──────────
FACE_MODEL_OLD = np.array([
    [-46.5,  0.0,  0.0],
    [-16.5,  0.0,  0.0],
    [ 16.5,  0.0,  0.0],
    [ 46.5,  0.0,  0.0],
    [-25.0, 30.0,  0.0],   # mouth right — OLD wrong value
    [ 25.0, 30.0,  0.0],   # mouth left  — OLD wrong value
    [  0.0, 15.0, -30.0],
], dtype=np.float64)

OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'vertical_gaze_diag_v2')
os.makedirs(OUT_DIR, exist_ok=True)

user32 = ctypes.windll.user32
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)

NUM_TRIALS   = 3
RECORD_SECS  = 2.0
CALIB_MARGIN = 0.08

TARGETS = [
    ("TOP",    SCREEN_W // 2, int(CALIB_MARGIN * SCREEN_H)),
    ("MIDDLE", SCREEN_W // 2, SCREEN_H // 2),
    ("BOTTOM", SCREEN_W // 2, int((1.0 - CALIB_MARGIN) * SCREEN_H)),
]

FACE_LANDMARK_INDICES = [33, 133, 362, 263, 61, 291, 1]
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
RIGHT_IRIS      = 468
DIST_COEFFS     = np.zeros((4, 1))
EYE_TO_USE      = 'right'


# ── CNN ───────────────────────────────────────────────────────────────────────
class GazeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten_size = 50 * 6 * 12
        self.fc1 = nn.Linear(self.flatten_size, 500)
        self.fc2 = nn.Linear(500 + 2, 2)
        self.relu = nn.ReLU()

    def forward(self, img, head_pose):
        x = self.relu(self.conv1(img))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = torch.cat([x, head_pose], dim=1)
        return self.fc2(x)


# ── helpers ───────────────────────────────────────────────────────────────────
def get_camera_matrix(w, h):
    return np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)


def solve_head_pose_with_model(lms, w, h, camera_matrix, face_model, prev_rvec, prev_tvec):
    """solvePnP using the supplied face_model (old or new). Identical strategy to gaze_to_screen.py."""
    image_points = np.array(
        [[lms[i].x * w, lms[i].y * h] for i in FACE_LANDMARK_INDICES],
        dtype=np.float64
    )
    if prev_rvec is None:
        ok, ri, ti = cv2.solvePnP(face_model, image_points, camera_matrix,
                                   DIST_COEFFS, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return None, None
    else:
        ri, ti = prev_rvec.copy(), prev_tvec.copy()

    ok, rvec, tvec = cv2.solvePnP(face_model, image_points, camera_matrix,
                                   DIST_COEFFS, rvec=ri, tvec=ti,
                                   useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None

    if prev_rvec is not None:
        if np.linalg.norm(rvec - prev_rvec) > 0.5 or np.linalg.norm(tvec - prev_tvec) > 80:
            return prev_rvec, prev_tvec
    return rvec, tvec


def normalize_eye_with_model(image, camera_matrix, rvec, tvec, face_model):
    """
    Mirrors normalize_single_eye() but uses a supplied face_model instead of
    the module-level one, so we can run both old and new in the same script.
    """
    R = rodrigues_to_matrix(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)

    # eye center from the supplied face_model
    eye_pts   = face_model[[0, 1]]   # right eye indices
    eye_cam   = (R @ eye_pts.T).T + t
    eye_center = eye_cam.mean(axis=0)

    distance = np.linalg.norm(eye_center)
    if distance < 1e-6:
        raise ValueError("Degenerate eye center")

    z_axis = eye_center / distance
    y_axis = np.cross(z_axis, R[:, 0]); y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis); x_axis /= np.linalg.norm(x_axis)
    R_norm = np.stack([x_axis, y_axis, z_axis], axis=0)

    S        = np.diag([1.0, 1.0, NORM_DISTANCE / distance])
    fx = fy  = NORM_FOCAL_LEN
    cx, cy   = EYE_PATCH_SIZE[0] / 2.0, EYE_PATCH_SIZE[1] / 2.0
    cam_norm = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    W        = cam_norm @ S @ R_norm @ np.linalg.inv(camera_matrix)

    warped = cv2.warpPerspective(image, W, EYE_PATCH_SIZE)
    gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray   = np.fliplr(gray)   # right eye

    head_rot_norm = R_norm @ R
    head_pitch = np.arcsin(-head_rot_norm[1, 2])
    head_yaw   = np.arctan2(-head_rot_norm[0, 2], head_rot_norm[2, 2])
    head_yaw   = -head_yaw   # right eye flip

    return gray, head_pitch, head_yaw, W, R_norm, eye_center, distance


def project_to_patch(u_raw, v_raw, W, patch_w=60):
    p = W @ np.array([u_raw, v_raw, 1.0])
    u = (patch_w - 1) - (p[0] / p[2])
    v = p[1] / p[2]
    return float(u), float(v)


def run_cnn(model, gray_patch, head_pitch, head_yaw):
    t_img = torch.from_numpy(gray_patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    t_hp  = torch.tensor([[head_pitch, head_yaw]], dtype=torch.float32)
    with torch.no_grad():
        pred = model(t_img, t_hp)[0]
    return np.degrees(pred[0].item()), np.degrees(pred[1].item())


def annotate_frame(frame, lms, w, h, label):
    ann = frame.copy()
    for i in FACE_LANDMARK_INDICES:
        cv2.circle(ann, (int(lms[i].x*w), int(lms[i].y*h)), 3, (255,0,255), -1)
    p_out = (int(lms[RIGHT_EYE_OUTER].x*w), int(lms[RIGHT_EYE_OUTER].y*h))
    p_inn = (int(lms[RIGHT_EYE_INNER].x*w), int(lms[RIGHT_EYE_INNER].y*h))
    cv2.circle(ann, p_out, 5, (0,255,0), 2)
    cv2.circle(ann, p_inn, 5, (0,255,0), 2)
    cv2.line(ann, p_out, p_inn, (0,255,0), 1)
    if len(lms) > RIGHT_IRIS:
        cv2.circle(ann, (int(lms[RIGHT_IRIS].x*w), int(lms[RIGHT_IRIS].y*h)), 5, (0,0,255), 2)
    cv2.putText(ann, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)
    return ann


def make_patch_visual(gray, iris_u, iris_v, canthus_v, label, scale=6):
    """Return annotated, scaled-up patch with reference lines and iris dot."""
    col  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    big  = cv2.resize(col, (gray.shape[1]*scale, gray.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    ph, pw = big.shape[:2]
    # yellow: intended patch centre (row 18)
    cv2.line(big, (0, int(18*scale)), (pw, int(18*scale)), (0,255,255), 1)
    # green: actual canthus mean y
    if canthus_v is not None:
        cy_px = int(canthus_v * scale)
        if 0 <= cy_px < ph:
            cv2.line(big, (0, cy_px), (pw, cy_px), (0,255,0), 1)
    # red dot: iris
    if iris_u is not None and iris_v is not None:
        pu, pv = int(iris_u*scale), int(iris_v*scale)
        if 0 <= pu < pw and 0 <= pv < ph:
            cv2.circle(big, (pu, pv), 4, (0,0,255), -1)
    # label at bottom
    cv2.putText(big, label, (4, ph-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)
    return big


def find_camera():
    try:
        from pygrabber.dshow_graph import FilterGraph
        for i, name in enumerate(FilterGraph().get_input_devices()):
            if 'hp' in name.lower():
                return i, name
    except Exception:
        pass
    return 0, "Default"


def open_camera(index, w=1280, h=720):
    for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened(): cap.release(); continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        for _ in range(15):
            ret, fr = cap.read()
            if ret and fr is not None and fr.std() > 5:
                return cap
            time.sleep(0.03)
        cap.release()
    return None


try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═════════════════════════════════════════════════════════════════════════════
def main():
    log_lines = []
    def log(msg):
        print(msg); log_lines.append(msg)

    log("=" * 80)
    log("VERTICAL GAZE DIAGNOSTIC v2  -  A/B COMPARISON")
    log("  Version A: OLD FACE_MODEL (mouth Y=30mm) - baseline")
    log("  Version B: FIXED FACE_MODEL (mouth Y=65mm) - corrected")
    log("  Same webcam frame -> same CNN -> only face model differs")
    log("=" * 80)

    # verify fixed model was imported
    log(f"FACE_MODEL_FIXED mouth Y = {FACE_MODEL_FIXED[4, 1]:.1f} mm  (expected 65.0)")
    log(f"FACE_MODEL_OLD   mouth Y = {FACE_MODEL_OLD[4, 1]:.1f} mm  (expected 30.0)")

    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'gaze_model_p00.pth')
    task_path  = os.path.join(os.path.dirname(__file__), '..', 'models', 'face_landmarker.task')

    device = torch.device('cpu')
    model  = GazeCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    log(f"Loaded model: {model_path}")

    base_options = python.BaseOptions(model_asset_path=task_path)
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
    )
    log(f"MediaPipe loaded: {task_path}")

    cam_idx, cam_name = find_camera()
    log(f"Camera: {cam_idx} ('{cam_name}')")
    cap = open_camera(cam_idx)
    if cap is None:
        log("ERROR: webcam failed to open."); return

    cv2.namedWindow("Vertical Gaze Diag v2", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Vertical Gaze Diag v2", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    log("\n" + "=" * 80)
    log("INSTRUCTIONS:")
    log("  1. Sit at a FIXED distance (~50-55 cm).  Chin/forehead rested.")
    log("  2. Move EYES ONLY — head must stay still throughout.")
    log("  3. Look at the dot, press SPACE when gaze is steady.")
    log("  4. Hold for 2 seconds while 'RECORDING' is shown.")
    log("  5. Sequence: TOP > MIDDLE > BOTTOM  x  3 trials.")
    log("=" * 80 + "\n")

    # Instruction screen
    instr = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    for i, line in enumerate([
        "VERTICAL GAZE DIAGNOSTIC v2",
        "A/B comparison: old crop (30mm) vs fixed crop (65mm)",
        "",
        "Instructions:",
        "  1. Fixed distance ~50-55 cm from screen",
        "  2. Rest chin/forehead — ONLY move eyes",
        "  3. Look at dot, press SPACE when steady",
        "  4. Hold 2 seconds during RECORDING",
        "  5. TOP > MIDDLE > BOTTOM  x  3 trials",
        "",
        "Press SPACE to begin..."
    ]):
        cv2.putText(instr, line, (100, 100 + i*55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
    cv2.imshow("Vertical Gaze Diag v2", instr)
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == ord(' '): break
        if k == ord('q'): cap.release(); cv2.destroyAllWindows(); return

    all_rows   = []
    saved_best = {}   # (trial, target) -> dict
    start_time = time.time()

    prev_A, prev_B = (None, None), (None, None)   # (rvec, tvec) per version

    for trial in range(1, NUM_TRIALS + 1):
        log(f"\n>>> TRIAL {trial}/{NUM_TRIALS}")
        prev_A = (None, None)
        prev_B = (None, None)

        for target_name, tx, ty in TARGETS:
            log(f"  Target: {target_name}  screen=({tx},{ty})")

            # ── WAIT PHASE ────────────────────────────────────────────────────
            live_pitch_A = live_pitch_B = None
            while True:
                ret, frame = cap.read()
                if not ret: continue
                h, w, _ = frame.shape
                ts = int((time.time() - start_time) * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

                if res.face_landmarks:
                    lms     = res.face_landmarks[0]
                    cam_mat = get_camera_matrix(w, h)

                    rvA, tvA = solve_head_pose_with_model(lms, w, h, cam_mat, FACE_MODEL_OLD,   *prev_A)
                    rvB, tvB = solve_head_pose_with_model(lms, w, h, cam_mat, FACE_MODEL_FIXED, *prev_B)

                    if rvA is not None:
                        prev_A = (rvA, tvA)
                        try:
                            gA, hpA, hyA, *_ = normalize_eye_with_model(frame, cam_mat, rvA, tvA, FACE_MODEL_OLD)
                            live_pitch_A, _ = run_cnn(model, gA, hpA, hyA)
                        except Exception: pass
                    if rvB is not None:
                        prev_B = (rvB, tvB)
                        try:
                            gB, hpB, hyB, *_ = normalize_eye_with_model(frame, cam_mat, rvB, tvB, FACE_MODEL_FIXED)
                            live_pitch_B, _ = run_cnn(model, gB, hpB, hyB)
                        except Exception: pass

                canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
                cv2.circle(canvas, (tx, ty), 35, (0,255,0), 3)
                cv2.circle(canvas, (tx, ty), 5,  (255,255,255), -1)
                cv2.putText(canvas, f"Trial {trial}/{NUM_TRIALS}  |  {target_name}  — look at dot, press SPACE",
                            (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)
                cv2.putText(canvas, "Head STILL — eyes only",
                            (50, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 2)
                if live_pitch_A is not None:
                    cv2.putText(canvas, f"Live  A (old 30mm): {live_pitch_A:+.1f} deg",
                                (50, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,180,255), 2)
                if live_pitch_B is not None:
                    cv2.putText(canvas, f"Live  B (fix 65mm): {live_pitch_B:+.1f} deg",
                                (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,120), 2)
                cv2.imshow("Vertical Gaze Diag v2", canvas)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    cap.release(); cv2.destroyAllWindows()
                    _save_all(all_rows, saved_best, log_lines, log); return
                if k == ord(' '): break

            # ── RECORD PHASE ──────────────────────────────────────────────────
            rec_start   = time.time()
            frame_count = 0

            while time.time() - rec_start < RECORD_SECS:
                ret, frame = cap.read()
                if not ret: continue
                h, w, _ = frame.shape
                ts = int((time.time() - start_time) * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

                if not res.face_landmarks: continue
                lms     = res.face_landmarks[0]
                cam_mat = get_camera_matrix(w, h)

                # solve pose with both models
                rvA, tvA = solve_head_pose_with_model(lms, w, h, cam_mat, FACE_MODEL_OLD,   *prev_A)
                rvB, tvB = solve_head_pose_with_model(lms, w, h, cam_mat, FACE_MODEL_FIXED, *prev_B)

                if rvA is None or rvB is None: continue
                prev_A = (rvA, tvA)
                prev_B = (rvB, tvB)

                # raw landmarks
                u_out = lms[RIGHT_EYE_OUTER].x * w;  v_out = lms[RIGHT_EYE_OUTER].y * h
                u_inn = lms[RIGHT_EYE_INNER].x * w;  v_inn = lms[RIGHT_EYE_INNER].y * h
                has_iris = len(lms) > RIGHT_IRIS
                u_iri = lms[RIGHT_IRIS].x * w if has_iris else None
                v_iri = lms[RIGHT_IRIS].y * h if has_iris else None

                # iris ratio (raw)
                if has_iris:
                    cv = np.array([u_inn-u_out, v_inn-v_out])
                    ew = np.linalg.norm(cv) + 1e-9
                    u_unit = cv / ew
                    nd     = np.array([-u_unit[1], u_unit[0]])
                    pm     = np.array([(u_out+u_inn)/2, (v_out+v_inn)/2])
                    raw_off = float(np.dot(np.array([u_iri-pm[0], v_iri-pm[1]]), nd))
                    iris_ratio = raw_off / ew
                else:
                    ew = raw_off = iris_ratio = None

                try:
                    gA, hpA, hyA, WA, _, _, _ = normalize_eye_with_model(frame, cam_mat, rvA, tvA, FACE_MODEL_OLD)
                    gB, hpB, hyB, WB, _, _, _ = normalize_eye_with_model(frame, cam_mat, rvB, tvB, FACE_MODEL_FIXED)
                except Exception as exc:
                    log(f"    WARNING normalize failed: {exc}"); continue

                pitchA_deg, yawA_deg = run_cnn(model, gA, hpA, hyA)
                pitchB_deg, yawB_deg = run_cnn(model, gB, hpB, hyB)

                # patch iris positions
                iuA = ivA = iuB = ivB = cvA = cvB = None
                if has_iris:
                    iuA, ivA = project_to_patch(u_iri, v_iri, WA)
                    iuB, ivB = project_to_patch(u_iri, v_iri, WB)
                cvA = (project_to_patch(u_out, v_out, WA)[1] + project_to_patch(u_inn, v_inn, WA)[1]) / 2
                cvB = (project_to_patch(u_out, v_out, WB)[1] + project_to_patch(u_inn, v_inn, WB)[1]) / 2

                row = {
                    'trial': trial, 'target': target_name, 'frame': frame_count,
                    # raw
                    'raw_iris_v':      round(v_iri, 2)     if v_iri    is not None else '',
                    'raw_offset_px':   round(raw_off, 3)   if raw_off  is not None else '',
                    'iris_ratio':      round(iris_ratio, 4) if iris_ratio is not None else '',
                    'raw_canthus_v':   round(v_out, 2),
                    # Version A (old)
                    'A_pitch_deg':     round(pitchA_deg, 3),
                    'A_patch_iris_v':  round(ivA, 2)  if ivA is not None else '',
                    'A_canthus_v':     round(cvA, 2),
                    'A_head_pitch':    round(np.degrees(hpA), 3),
                    # Version B (fixed)
                    'B_pitch_deg':     round(pitchB_deg, 3),
                    'B_patch_iris_v':  round(ivB, 2)  if ivB is not None else '',
                    'B_canthus_v':     round(cvB, 2),
                    'B_head_pitch':    round(np.degrees(hpB), 3),
                }
                all_rows.append(row)

                key = (trial, target_name)
                if key not in saved_best:
                    saved_best[key] = {
                        'frame': frame.copy(), 'lms': lms,
                        'gA': gA.copy(), 'gB': gB.copy(),
                        'ivA': ivA, 'cvA': cvA, 'iuA': iuA,
                        'ivB': ivB, 'cvB': cvB, 'iuB': iuB,
                        'pitchA': pitchA_deg, 'pitchB': pitchB_deg,
                        'target': target_name, 'trial': trial,
                    }

                frame_count += 1

                # recording display
                canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
                cv2.circle(canvas, (tx, ty), 35, (0,165,255), 3)
                cv2.circle(canvas, (tx, ty), 5,  (255,255,255), -1)
                cv2.putText(canvas, f"RECORDING Trial {trial} | {target_name} | {frame_count} frames",
                            (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,165,255), 2)
                cv2.putText(canvas, f"A={pitchA_deg:+.1f}  B={pitchB_deg:+.1f} deg",
                            (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
                cv2.imshow("Vertical Gaze Diag v2", canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release(); cv2.destroyAllWindows()
                    _save_all(all_rows, saved_best, log_lines, log); return

            log(f"    {frame_count} usable frames recorded.")

    cap.release()
    cv2.destroyAllWindows()
    _save_all(all_rows, saved_best, log_lines, log)


# ═════════════════════════════════════════════════════════════════════════════
def _save_all(all_rows, saved_best, log_lines, log):

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, 'measurements.csv')
    if all_rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
        log(f"Saved CSV: {csv_path}  ({len(all_rows)} rows)")

    # ── patch images ──────────────────────────────────────────────────────────
    for (trial, tgt), d in saved_best.items():
        w_fr, h_fr = d['frame'].shape[1], d['frame'].shape[0]

        # raw frame annotated
        ann = annotate_frame(d['frame'], d['lms'], w_fr, h_fr, f"T{trial} {tgt}")
        cv2.imwrite(os.path.join(OUT_DIR, f"frame_t{trial}_{tgt}.png"), ann)

        # clean raw frame (for re-analysis)
        cv2.imwrite(os.path.join(OUT_DIR, f"raw_t{trial}_{tgt}.png"), d['frame'])

        # patch A
        pA = make_patch_visual(d['gA'], d['iuA'], d['ivA'], d['cvA'],
                               f"A(30mm) {d['pitchA']:+.1f}deg")
        # patch B
        pB = make_patch_visual(d['gB'], d['iuB'], d['ivB'], d['cvB'],
                               f"B(65mm) {d['pitchB']:+.1f}deg")

        # side by side A|B
        if pA.shape[0] != pB.shape[0]:
            mh = max(pA.shape[0], pB.shape[0])
            pA = np.vstack([pA, np.zeros((mh-pA.shape[0], pA.shape[1], 3), np.uint8)])
            pB = np.vstack([pB, np.zeros((mh-pB.shape[0], pB.shape[1], 3), np.uint8)])
        sep = np.ones((pA.shape[0], 4, 3), dtype=np.uint8) * 180   # separator
        ab  = np.hstack([pA, sep, pB])
        cv2.imwrite(os.path.join(OUT_DIR, f"patch_ab_t{trial}_{tgt}.png"), ab)

    log(f"Saved per-target patches and frames to {OUT_DIR}")

    # ── side-by-side per trial (all 3 targets) ────────────────────────────────
    for trial in range(1, 4):
        panels = []
        for tgt in ['TOP', 'MIDDLE', 'BOTTOM']:
            key = (trial, tgt)
            if key not in saved_best: continue
            d = saved_best[key]
            pA = make_patch_visual(d['gA'], d['iuA'], d['ivA'], d['cvA'],
                                   f"{tgt} A {d['pitchA']:+.1f}d", scale=5)
            pB = make_patch_visual(d['gB'], d['iuB'], d['ivB'], d['cvB'],
                                   f"{tgt} B {d['pitchB']:+.1f}d", scale=5)
            sep = np.ones((pA.shape[0], 3, 3), np.uint8) * 150
            panels.append(np.hstack([pA, sep, pB]))
        if panels:
            mh = max(p.shape[0] for p in panels)
            sep_v = np.ones((mh, 6, 3), np.uint8) * 80
            out_img = panels[0]
            for p in panels[1:]:
                if p.shape[0] < mh:
                    p = np.vstack([p, np.zeros((mh-p.shape[0], p.shape[1], 3), np.uint8)])
                out_img = np.hstack([out_img, sep_v, p])
            cv2.imwrite(os.path.join(OUT_DIR, f"comparison_t{trial}.png"), out_img)
    log("Saved comparison images.")

    # ── statistics ────────────────────────────────────────────────────────────
    stats = {}
    for trial in range(1, 4):
        for tgt in ['TOP', 'MIDDLE', 'BOTTOM']:
            rows = [r for r in all_rows if r['trial']==trial and r['target']==tgt]
            if not rows: continue
            pA   = [r['A_pitch_deg'] for r in rows]
            pB   = [r['B_pitch_deg'] for r in rows]
            ir   = [r['iris_ratio']  for r in rows if r['iris_ratio']  != '']
            ivA  = [r['A_patch_iris_v'] for r in rows if r['A_patch_iris_v'] != '']
            ivB  = [r['B_patch_iris_v'] for r in rows if r['B_patch_iris_v'] != '']
            cvA  = [r['A_canthus_v'] for r in rows]
            cvB  = [r['B_canthus_v'] for r in rows]
            stats[(trial, tgt)] = {
                'n':   len(rows),
                'A_pitch_mean': statistics.mean(pA),
                'A_pitch_std':  statistics.stdev(pA) if len(pA)>1 else 0,
                'B_pitch_mean': statistics.mean(pB),
                'B_pitch_std':  statistics.stdev(pB) if len(pB)>1 else 0,
                'iris_ratio_mean': statistics.mean(ir) if ir else None,
                'A_iris_v_mean':   statistics.mean(ivA) if ivA else None,
                'B_iris_v_mean':   statistics.mean(ivB) if ivB else None,
                'A_canthus_v':     statistics.mean(cvA),
                'B_canthus_v':     statistics.mean(cvB),
            }

    log("\nAVERAGE RESULTS:")
    log(f"{'Tri':<4} {'Tgt':<8} {'N':<5} {'A pitch':<16} {'B pitch':<16} {'IrisRatio':<12} {'A iris_v':<10} {'B iris_v':<10} {'A canth':<9} {'B canth'}")
    log("-" * 105)
    for trial in range(1,4):
        for tgt in ['TOP','MIDDLE','BOTTOM']:
            s = stats.get((trial, tgt))
            if not s: continue
            ir  = f"{s['iris_ratio_mean']:+.4f}" if s['iris_ratio_mean'] else 'N/A'
            ivA = f"{s['A_iris_v_mean']:.2f}"    if s['A_iris_v_mean']   else 'N/A'
            ivB = f"{s['B_iris_v_mean']:.2f}"    if s['B_iris_v_mean']   else 'N/A'
            log(f"{trial:<4} {tgt:<8} {s['n']:<5} {s['A_pitch_mean']:+.2f}+/-{s['A_pitch_std']:.2f}  "
                f"{s['B_pitch_mean']:+.2f}+/-{s['B_pitch_std']:.2f}  "
                f"{ir:<12} {ivA:<10} {ivB:<10} {s['A_canthus_v']:.1f}     {s['B_canthus_v']:.1f}")
        log("")

    log("TOP-to-BOTTOM DELTAS:")
    log(f"{'':4} {'Version A (30mm)':<25} {'Version B (65mm)'}")
    log("-" * 55)
    A_deltas, B_deltas = [], []
    for trial in range(1,4):
        sT = stats.get((trial,'TOP')); sB = stats.get((trial,'BOTTOM'))
        if sT and sB:
            dA = sB['A_pitch_mean'] - sT['A_pitch_mean']
            dB = sB['B_pitch_mean'] - sT['B_pitch_mean']
            A_deltas.append(dA); B_deltas.append(dB)
            log(f"T{trial}   {dA:+.2f} deg  ({sT['A_pitch_mean']:+.2f}->{sB['A_pitch_mean']:+.2f})    "
                f"{dB:+.2f} deg  ({sT['B_pitch_mean']:+.2f}->{sB['B_pitch_mean']:+.2f})")
    if A_deltas:
        log(f"Mean  {sum(A_deltas)/len(A_deltas):+.2f} deg                    {sum(B_deltas)/len(B_deltas):+.2f} deg")

    log("\nPATCH IRIS V CHANGE (TOP->BOT, patch center=18px):")
    for trial in range(1,4):
        sT = stats.get((trial,'TOP')); sB = stats.get((trial,'BOTTOM'))
        if sT and sB and sT['A_iris_v_mean'] and sB['A_iris_v_mean']:
            dA = sB['A_iris_v_mean'] - sT['A_iris_v_mean']
            dB = sB['B_iris_v_mean'] - sT['B_iris_v_mean'] if sT['B_iris_v_mean'] and sB['B_iris_v_mean'] else None
            log(f"T{trial}   A: {sT['A_iris_v_mean']:.2f}->{sB['A_iris_v_mean']:.2f} ({dA:+.2f}px)   "
                + (f"B: {sT['B_iris_v_mean']:.2f}->{sB['B_iris_v_mean']:.2f} ({dB:+.2f}px)" if dB is not None else ""))

    # ── pitch plot ────────────────────────────────────────────────────────────
    if HAS_MPL and stats:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        colors = ['tab:blue','tab:orange','tab:green']
        x_lbls = ['TOP','MIDDLE','BOTTOM']
        for ax, version, key_p, key_std, title in [
            (axes[0], 'A', 'A_pitch_mean', 'A_pitch_std', 'Version A  (old 30mm mouth)'),
            (axes[1], 'B', 'B_pitch_mean', 'B_pitch_std', 'Version B  (fixed 65mm mouth)'),
        ]:
            for ti, trial in enumerate([1,2,3]):
                ys   = [stats.get((trial,t), {}).get(key_p,  float('nan')) for t in x_lbls]
                errs = [stats.get((trial,t), {}).get(key_std, 0)            for t in x_lbls]
                ax.errorbar([0,1,2], ys, yerr=errs, marker='o', label=f'Trial {trial}',
                            color=colors[ti], capsize=5, linewidth=2)
            ax.set_title(title, fontsize=12)
            ax.set_xticks([0,1,2]); ax.set_xticklabels(x_lbls, fontsize=12)
            ax.set_ylabel("CNN pitch (deg)"); ax.legend(); ax.grid(True, alpha=0.3)
            ax.invert_yaxis()
        fig.suptitle("A/B Comparison: Old vs Fixed FACE_MODEL", fontsize=13)
        fig.tight_layout()
        plot_path = os.path.join(OUT_DIR, 'pitch_plot_AB.png')
        fig.savefig(plot_path, dpi=150); plt.close(fig)
        log(f"Saved: {plot_path}")

    # ── summary.txt ───────────────────────────────────────────────────────────
    summary = [
        "VERTICAL GAZE DIAGNOSTIC v2 — A/B SUMMARY",
        "=" * 60,
        f"Version A: FACE_MODEL mouth Y = 30.0 mm  (OLD, wrong)",
        f"Version B: FACE_MODEL mouth Y = 65.0 mm  (FIXED)",
        "",
        "Patch centre (intended iris row) = 18 px in 60x36 patch",
        "",
        "Mean TOP/MID/BOT pitch (Version A):",
    ]
    for tgt in ['TOP','MIDDLE','BOTTOM']:
        vals = [stats[(t,tgt)]['A_pitch_mean'] for t in [1,2,3] if (t,tgt) in stats]
        if vals:
            summary.append(f"  {tgt:<8}: {sum(vals)/len(vals):+.2f} deg")
    summary.append("\nMean TOP/MID/BOT pitch (Version B):")
    for tgt in ['TOP','MIDDLE','BOTTOM']:
        vals = [stats[(t,tgt)]['B_pitch_mean'] for t in [1,2,3] if (t,tgt) in stats]
        if vals:
            summary.append(f"  {tgt:<8}: {sum(vals)/len(vals):+.2f} deg")
    summary += ["", "Mean TOP->BOTTOM delta:",
                f"  Version A: {sum(A_deltas)/len(A_deltas):+.2f} deg" if A_deltas else "  N/A",
                f"  Version B: {sum(B_deltas)/len(B_deltas):+.2f} deg" if B_deltas else "  N/A"]

    with open(os.path.join(OUT_DIR, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    log(f"Saved: {os.path.join(OUT_DIR, 'summary.txt')}")

    # ── terminal log ──────────────────────────────────────────────────────────
    with open(os.path.join(OUT_DIR, 'terminal_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    log(f"Saved: {os.path.join(OUT_DIR, 'terminal_log.txt')}")
    log("\nAll results -> " + OUT_DIR)


if __name__ == '__main__':
    main()
