import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
import time
import ctypes

from normalize import normalize_single_eye, FACE_MODEL

# ============================================================
# SCREEN SIZE (Windows-specific; adjust if on another OS)
# ============================================================
user32 = ctypes.windll.user32
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)

# ============================================================
# CALIBRATION CONFIG
# ============================================================
MARGIN = 0.08  # keep calibration points slightly inset from the true screen edge
CALIB_POINTS_FRAC = [
    (MARGIN, MARGIN), (0.5, MARGIN), (1 - MARGIN, MARGIN),
    (MARGIN, 0.5),    (0.5, 0.5),    (1 - MARGIN, 0.5),
    (MARGIN, 1 - MARGIN), (0.5, 1 - MARGIN), (1 - MARGIN, 1 - MARGIN),
]
CALIB_POINTS = [(int(fx * SCREEN_W), int(fy * SCREEN_H)) for fx, fy in CALIB_POINTS_FRAC]
FRAMES_PER_CALIB_POINT = 25  # averaged, not single-frame - reduces noise from any one bad reading

SMOOTHING_ALPHA = 0.35  # exponential smoothing for the live circle (higher = more responsive, jitterier)

# Bigger, bullseye-style calibration target (per professor's suggestion) - a plain
# small filled dot leaves room for imprecise fixation, especially on the vertical
# axis where natural eye movement range is already small. A larger ring with a
# tiny precise center point gives a much clearer, more fixatable target.
CALIB_DOT_OUTER_RADIUS = 35   # was a plain filled circle of radius 20
CALIB_DOT_INNER_RADIUS = 5


def draw_calib_dot(canvas, px, py, color_outer, color_inner=(255, 255, 255)):
    cv2.circle(canvas, (px, py), CALIB_DOT_OUTER_RADIUS, color_outer, 3)   # ring
    cv2.circle(canvas, (px, py), CALIB_DOT_INNER_RADIUS, color_inner, -1)  # precise center dot


# ---- must exactly match training architecture ----
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


import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def find_asset(filename):
    for candidate in [
        os.path.join(PROJECT_ROOT, "models", filename),
        os.path.join(SCRIPT_DIR, filename),
        os.path.join(".", "models", filename),
        filename,
    ]:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return filename

device = torch.device("cpu")
model = GazeCNN().to(device)
model_path = find_asset("gaze_model_p00.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---- MediaPipe setup ----
task_path = find_asset("face_landmarker.task")
base_options = python.BaseOptions(model_asset_path=task_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_facial_transformation_matrixes=True
)
landmarker = vision.FaceLandmarker.create_from_options(options)

FACE_LANDMARK_INDICES = [33, 133, 362, 263, 61, 291, 1]
# index 1 = nose tip in MediaPipe's 468-point face mesh. This MUST be added last,
# in the same order as the 7th row (nose tip) added to FACE_MODEL in normalize.py -
# solvePnP matches image_points[i] to FACE_MODEL[i] by position, so the orders
# must line up exactly.
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 33, 133
RIGHT_EYE_IRIS = 468
EYE_TO_USE = 'right'
DIST_COEFFS = np.zeros((4, 1))


def compute_iris_ratios(lms, w, h):
    """
    Computes scale-invariant horizontal and vertical iris displacement relative to the eye corner axis:
    - canthi_vec: vector from outer canthus (33) to inner canthus (133).
    - u = canthi_vec / eye_width (nasal direction, +X along eye axis).
    - normal_down = [-u[1], u[0]] (perpendicular downward direction, +Y).
    - iris_h: signed distance along u from canthal midpoint, normalized by eye_width.
    - iris_v: signed distance along normal_down from canthal midpoint, normalized by eye_width.
    """
    if len(lms) <= RIGHT_EYE_IRIS:
        return 0.0, 0.0
    p_outer = np.array([lms[RIGHT_EYE_OUTER].x * w, lms[RIGHT_EYE_OUTER].y * h], dtype=np.float64)
    p_inner = np.array([lms[RIGHT_EYE_INNER].x * w, lms[RIGHT_EYE_INNER].y * h], dtype=np.float64)
    p_iris  = np.array([lms[RIGHT_EYE_IRIS].x * w,  lms[RIGHT_EYE_IRIS].y * h],  dtype=np.float64)

    canthi_vec = p_inner - p_outer
    eye_width = np.linalg.norm(canthi_vec)
    if eye_width < 1e-4:
        return 0.0, 0.0

    u = canthi_vec / eye_width
    normal_down = np.array([-u[1], u[0]], dtype=np.float64)
    p_mid = 0.5 * (p_outer + p_inner)
    d_iris = p_iris - p_mid

    horiz_offset_px = np.dot(d_iris, u)
    vert_offset_px = np.dot(d_iris, normal_down)

    iris_h = float(horiz_offset_px / eye_width)
    iris_v = float(vert_offset_px / eye_width)
    return iris_h, iris_v


def get_camera_matrix(w, h):
    focal_length = w
    return np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)


def solve_head_pose(lms, w, h, camera_matrix, prev_rvec, prev_tvec):
    image_points = np.array(
        [[lms[i].x * w, lms[i].y * h] for i in FACE_LANDMARK_INDICES],
        dtype=np.float64
    )
    use_guess = prev_rvec is not None and prev_tvec is not None

    if not use_guess:
        # NOTE: previously this branch used cv2.solvePnPGeneric(..., flags=SOLVEPNP_IPPE)
        # to resolve a pose ambiguity that only exists when all object points are
        # coplanar. IPPE *requires* a planar point set - it silently fails on
        # anything else, which is why it broke once the nose point was added.
        #
        # Plain SOLVEPNP_ITERATIVE with no seed was tried next, but it isn't
        # designed to run "cold" (no starting guess) - on some frames it locks
        # onto a bad pose, and the rot_jump/trans_jump safety check below then
        # rejects every good subsequent frame because they look "too different"
        # from that bad first estimate, freezing the whole point on garbage.
        #
        # SOLVEPNP_EPNP is a linear method built specifically to produce a solid
        # estimate with no seed and no planarity requirement - ideal for exactly
        # this "first frame of a fresh calibration point" situation.
        ok, rvec_init, tvec_init = cv2.solvePnP(
            FACE_MODEL, image_points, camera_matrix, DIST_COEFFS,
            flags=cv2.SOLVEPNP_EPNP
        )
        if not ok:
            return None, None
    else:
        rvec_init, tvec_init = prev_rvec.copy(), prev_tvec.copy()

    ok, rvec, tvec = cv2.solvePnP(
        FACE_MODEL, image_points, camera_matrix, DIST_COEFFS,
        rvec=rvec_init, tvec=tvec_init,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None, None

    if prev_rvec is not None:
        rot_jump = np.linalg.norm(rvec - prev_rvec)
        trans_jump = np.linalg.norm(tvec - prev_tvec)
        if rot_jump > 0.5 or trans_jump > 80:
            return prev_rvec, prev_tvec

    return rvec, tvec


def get_gaze_pitch_yaw(frame, w, h, prev_rvec, prev_tvec, timestamp_ms):
    """Runs the full pipeline on one frame. Returns (pitch, yaw, iris_h, iris_v, new_rvec, new_tvec) or (None, None, None, None, prev_rvec, prev_tvec)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.face_landmarks:
        return None, None, None, None, prev_rvec, prev_tvec

    lms = result.face_landmarks[0]
    iris_h, iris_v = compute_iris_ratios(lms, w, h)

    camera_matrix = get_camera_matrix(w, h)
    rvec, tvec = solve_head_pose(lms, w, h, camera_matrix, prev_rvec, prev_tvec)
    if rvec is None:
        return None, None, None, None, prev_rvec, prev_tvec

    try:
        gray, _, _, head_pitch, head_yaw = normalize_single_eye(
            frame, camera_matrix, DIST_COEFFS, rvec, tvec,
            gaze_target_3d=[0, 0, 600], eye_side=EYE_TO_USE
        )
    except Exception:
        return None, None, None, None, rvec, tvec

    img_tensor = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    head_pose_tensor = torch.tensor([[head_pitch, head_yaw]], dtype=torch.float32)
    with torch.no_grad():
        pred = model(img_tensor, head_pose_tensor)[0]
    return pred[0].item(), pred[1].item(), iris_h, iris_v, rvec, tvec


def features_x(pitch, yaw, iris_h, iris_v):
    # Complete dual-axis hybrid: CNN yaw + physical horizontal iris displacement
    return np.array([1.0, yaw, iris_h])


def features_y(pitch, yaw, iris_h, iris_v):
    # Complete dual-axis hybrid: CNN pitch + physical vertical iris displacement
    return np.array([1.0, pitch, iris_v])


from collections import deque

# ============================================================
# TWO-STAGE ADAPTIVE GAZE FILTER (Angle-Space Stabilization)
# ============================================================
class RockSolidFilter:
    """
    Two-stage gaze angle filter with sustained saccade detection:
    1. Median Window (5 frames): Eliminates single-frame spikes, blinks, and solvePnP noise.
    2. Adaptive Deadband with Sustained Movement Detection:
       - Fixation (diff <= deadband): Completely locked still (zero micro-jitter).
       - Smooth pursuit (deadband < diff <= saccade_thresh): Steady damping (alpha_still = 0.08).
       - Saccade (diff > saccade_thresh): Requires >= 2 consecutive frames of large movement
         before accelerating (alpha_move = 0.35). Single-frame noise/blinks CANNOT trigger a jump!
    """
    def __init__(self, init_deg=0.0, window_size=5, alpha_still=0.08, alpha_move=0.35, deadband=0.15, saccade_thresh=0.60):
        self.buf = deque([float(init_deg)] * window_size, maxlen=window_size)
        self.smooth = float(init_deg)
        self.alpha_still = float(alpha_still)
        self.alpha_move = float(alpha_move)
        self.deadband = float(deadband)
        self.saccade_thresh = float(saccade_thresh)
        self.consecutive_saccades = 0
        self.initialized = False

    def update(self, val_deg):
        val_deg = float(val_deg)
        if not self.initialized:
            self.buf = deque([val_deg] * self.buf.maxlen, maxlen=self.buf.maxlen)
            self.smooth = val_deg
            self.initialized = True
            return self.smooth

        self.buf.append(val_deg)
        med = float(np.median(self.buf))
        diff = med - self.smooth
        abs_diff = abs(diff)

        if abs_diff <= self.deadband:
            self.consecutive_saccades = 0
            return self.smooth

        if abs_diff > self.saccade_thresh:
            self.consecutive_saccades += 1
        else:
            self.consecutive_saccades = 0

        # Only accelerate if movement is sustained for >= 2 frames (prevents sudden jerks)
        alpha = self.alpha_move if self.consecutive_saccades >= 2 else self.alpha_still
        self.smooth = self.smooth + alpha * diff
        return self.smooth


def fit_mapping(calib_data):
    """calib_data: list of (pitch, yaw, iris_h, iris_v, screen_x, screen_y). Returns (coeffs_x, coeffs_y, rmse_x, rmse_y, rmse)."""
    Ax = np.array([features_x(p, y, ih, iv) for p, y, ih, iv, _, _ in calib_data])
    Ay = np.array([features_y(p, y, ih, iv) for p, y, ih, iv, _, _ in calib_data])
    xs = np.array([sx for _, _, _, _, sx, _ in calib_data])
    ys = np.array([sy for _, _, _, _, _, sy in calib_data])
    coeffs_x, _, _, _ = np.linalg.lstsq(Ax, xs, rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(Ay, ys, rcond=None)

    pred_xs = Ax @ coeffs_x
    pred_ys = Ay @ coeffs_y
    rmse_x = float(np.sqrt(np.mean((pred_xs - xs) ** 2)))
    rmse_y = float(np.sqrt(np.mean((pred_ys - ys) ** 2)))
    rmse = float(np.sqrt(rmse_x**2 + rmse_y**2))

    return coeffs_x, coeffs_y, rmse_x, rmse_y, rmse


def save_tracking_session_log(calib_data, coeffs_x, coeffs_y, rmse_x, rmse_y,
                              live_frames_count, avg_fps, raw_jumps, filt_jumps):
    log_dir = os.path.join(PROJECT_ROOT, "diagnostics", "results")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "live_tracking_log.txt")

    mean_raw_jump = float(np.mean(raw_jumps)) if raw_jumps else 0.0
    mean_filt_jump = float(np.mean(filt_jumps)) if filt_jumps else 0.0
    jitter_reduction = (1.0 - (mean_filt_jump / mean_raw_jump)) * 100.0 if mean_raw_jump > 0 else 0.0

    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 75,
        f"LIVE GAZE TRACKING SESSION LOG (COMPLETE DUAL-AXIS HYBRID) — {timestamp_str}",
        "=" * 75,
        f"Screen Resolution  : {SCREEN_W} x {SCREEN_H}",
        f"Calibration Points : {len(calib_data)}",
        "",
        "CALIBRATION TARGETS & RESIDUAL ERRORS:",
    ]
    for idx, (p, y, ih, iv, sx, sy) in enumerate(calib_data):
        pred_x = float(np.dot(features_x(p, y, ih, iv), coeffs_x))
        pred_y = float(np.dot(features_y(p, y, ih, iv), coeffs_y))
        err = np.sqrt((pred_x - sx)**2 + (pred_y - sy)**2)
        lines.append(f"  Point {idx+1:2d}: target=({sx:4d},{sy:4d}) | "
                     f"pitch={np.degrees(p):6.1f} deg, yaw={np.degrees(y):6.1f} deg, iris_h={ih:6.4f}, iris_v={iv:6.4f} | error={err:5.1f} px")

    lines.extend([
        "",
        "COMPLETE DUAL-AXIS HYBRID FORMULAS & SENSITIVITIES:",
        f"  Screen X = {coeffs_x[0]:.1f} + ({coeffs_x[1]:.1f} * yaw_rad) + ({coeffs_x[2]:.1f} * iris_h)",
        f"             -> CNN yaw sensitivity     = {coeffs_x[1] * np.pi / 180.0:6.1f} px/deg",
        f"             -> Iris horiz sensitivity = {coeffs_x[2]:6.1f} px/ratio",
        f"  Screen Y = {coeffs_y[0]:.1f} + ({coeffs_y[1]:.1f} * pitch_rad) + ({coeffs_y[2]:.1f} * iris_v)",
        f"             -> CNN pitch sensitivity  = {coeffs_y[1] * np.pi / 180.0:6.1f} px/deg",
        f"             -> Iris vert sensitivity  = {coeffs_y[2]:6.1f} px/ratio",
        f"  Calibration RMSE : X = {rmse_x:.1f} px, Y = {rmse_y:.1f} px, Total = {rmse:.1f} px",
        "",
        "ROCK-SOLID ANTI-JUMP FILTER PERFORMANCE:",
        f"  Total live frames evaluated : {live_frames_count}",
        f"  Average Tracking FPS        : {avg_fps:.1f}",
        f"  Raw frame-to-frame jump     : {mean_raw_jump:.1f} px",
        f"  Filtered frame jump         : {mean_filt_jump:.1f} px",
        f"  Measured Jitter Reduction   : {jitter_reduction:.1f}%",
        "=" * 75,
        "",
    ])

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[LOG] Saved tracking session results to: {log_file}")


def predict_screen_point(pitch, yaw, iris_h, iris_v, coeffs_x, coeffs_y):
    x = float(np.dot(features_x(pitch, yaw, iris_h, iris_v), coeffs_x))
    y = float(np.dot(features_y(pitch, yaw, iris_h, iris_v), coeffs_y))
    x = np.clip(x, 0, SCREEN_W - 1)
    y = np.clip(y, 0, SCREEN_H - 1)
    return int(x), int(y)


def find_camera_index(name_hint):
    from pygrabber.dshow_graph import FilterGraph
    devices = FilterGraph().get_input_devices()
    for i, name in enumerate(devices):
        if name_hint.lower() in name.lower():
            return i, name
    return None, None


def open_camera_robust(index, width=640, height=480, label=""):
    backend_candidates = [
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_ANY, "default"),
    ]
    for backend, name in backend_candidates:
        try:
            cap = cv2.VideoCapture(index, backend)
        except cv2.error:
            continue
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        valid = False
        for _ in range(15):
            ret, frame = cap.read()
            if ret and frame is not None and frame.std() > 5:
                valid = True
                break
            time.sleep(0.03)
        if valid:
            print(f"[{label}] camera {index}: opened successfully using backend '{name}'")
            return cap
        cap.release()
        print(f"[{label}] camera {index}: backend '{name}' opened but gave no valid frames, trying next...")
    print(f"[{label}] camera {index}: FAILED to get valid frames from any backend")
    return None


EYE_CAMERA_NAME_HINT = "HP"  # substring matching your laptop webcam's device name

# ============================================================
# MAIN
# ============================================================
cam_index, cam_name = find_camera_index(EYE_CAMERA_NAME_HINT)
if cam_index is None:
    print(f"Could not find a camera matching '{EYE_CAMERA_NAME_HINT}'. "
          f"Run pygrabber's FilterGraph().get_input_devices() to see actual device names.")
    raise SystemExit(1)
print(f"Camera: index {cam_index} ('{cam_name}')")

cap = open_camera_robust(cam_index, label="EYE")
if cap is None:
    print("Camera failed to open - stopping before calibration.")
    raise SystemExit(1)
start_time = time.time()
prev_rvec, prev_tvec = None, None

cv2.namedWindow("Gaze Screen", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gaze Screen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ---- CALIBRATION PHASE ----
calib_data = []
print(f"Calibration starting - {len(CALIB_POINTS)} points, "
      f"{FRAMES_PER_CALIB_POINT} frames averaged per point.")
print("Look at each dot as it appears and hold your gaze steady.")

import random

CAPTURE_DURATION_SEC = 1.2  # real time, not frame count - robust to variable FPS from heavy per-frame compute

shuffled_indexed = list(enumerate(CALIB_POINTS))  # keep the true grid index attached
random.shuffle(shuffled_indexed)  # decorrelates calibration order from any session-drift-over-time

for point_idx, (grid_idx, (px, py)) in enumerate(shuffled_indexed):
    # Reset head-pose tracking fresh for each point, instead of carrying forward the
    # previous point's rvec/tvec. Carrying it forward let any small solvePnP drift
    # accumulate steadily across the (several-minute) calibration session, which -
    # since points were visited in a fixed spatial order - looked like a real signal
    # to the regression and caused pitch/yaw crosstalk in the live tracking.
    prev_rvec, prev_tvec = None, None

    # ---- WAITING PHASE: dot is shown, live readout updates, user presses SPACE when ready ----
    last_pitch, last_yaw, last_ih, last_iv = None, None, None, None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w, _ = frame.shape
        timestamp_ms = int((time.time() - start_time) * 1000)
        pitch, yaw, iris_h, iris_v, prev_rvec, prev_tvec = get_gaze_pitch_yaw(frame, w, h, prev_rvec, prev_tvec, timestamp_ms)
        if pitch is not None:
            last_pitch, last_yaw, last_ih, last_iv = pitch, yaw, iris_h, iris_v

        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        draw_calib_dot(canvas, px, py, color_outer=(0, 255, 0))
        cv2.putText(canvas, f"Point {point_idx + 1}/{len(CALIB_POINTS)} (grid pos {grid_idx}) - "
                             f"look at the dot, press SPACE when steady", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if last_pitch is not None:
            cv2.putText(canvas, f"live: pitch={np.degrees(last_pitch):.1f} yaw={np.degrees(last_yaw):.1f} iris_h={last_ih:.3f} iris_v={last_iv:.3f}",
                        (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow("Gaze Screen", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit(0)
        if key == ord(' '):
            break  # move to capture phase

    # ---- CAPTURE PHASE: fixed real-time duration, not frame count ----
    samples = []
    capture_start = time.time()
    while time.time() - capture_start < CAPTURE_DURATION_SEC:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w, _ = frame.shape
        timestamp_ms = int((time.time() - start_time) * 1000)
        pitch, yaw, iris_h, iris_v, prev_rvec, prev_tvec = get_gaze_pitch_yaw(frame, w, h, prev_rvec, prev_tvec, timestamp_ms)

        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        draw_calib_dot(canvas, px, py, color_outer=(0, 200, 255))  # different color = "recording"
        cv2.putText(canvas, "recording...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        cv2.imshow("Gaze Screen", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit(0)

        if pitch is not None and iris_h is not None and iris_v is not None:
            samples.append((pitch, yaw, iris_h, iris_v))

    if len(samples) < 5:
        print(f"WARNING: point {point_idx + 1} got only {len(samples)} valid samples - "
              f"face may have been lost. Consider recalibrating if accuracy looks poor.")
    if samples:
        avg_pitch = float(np.mean([s[0] for s in samples]))
        avg_yaw = float(np.mean([s[1] for s in samples]))
        avg_ih = float(np.mean([s[2] for s in samples]))
        avg_iv = float(np.mean([s[3] for s in samples]))
        calib_data.append((avg_pitch, avg_yaw, avg_ih, avg_iv, px, py))
        print(f"Point {point_idx + 1} (grid pos {grid_idx}, screen=({px},{py})): "
              f"avg pitch={np.degrees(avg_pitch):.1f} yaw={np.degrees(avg_yaw):.1f} "
              f"iris_h={avg_ih:.4f} iris_v={avg_iv:.4f} ({len(samples)} samples)")

if len(calib_data) < 6:
    print("Not enough calibration points succeeded to fit a reliable mapping. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit(1)

coeffs_x, coeffs_y, rmse_x, rmse_y, rmse = fit_mapping(calib_data)
print("\n" + "=" * 60)
print(f"Calibration complete! Fit RMSE: X={rmse_x:.1f}px, Y={rmse_y:.1f}px, Total={rmse:.1f}px")
print(f"Horizontal sensitivity: CNN yaw = {coeffs_x[1] * np.pi / 180.0:.1f} px/deg, Iris = {coeffs_x[2]:.1f} px/ratio")
print(f"Vertical sensitivity:   CNN pitch = {coeffs_y[1] * np.pi / 180.0:.1f} px/deg, Iris = {coeffs_y[2]:.1f} px/ratio")
print("Starting live tracking:")
print("  - [q] Quit and save session log")
print("  - [t] Toggle raw noisy dot (yellow) vs smooth filtered dot (red)")
print("  - [r] Toggle HUD display")
print("=" * 60 + "\n")

# ---- LIVE TRACKING PHASE ----
pitch_filter  = RockSolidFilter(init_deg=0.0, window_size=5, alpha_still=0.08, alpha_move=0.35, deadband=0.15, saccade_thresh=0.60)
yaw_filter    = RockSolidFilter(init_deg=0.0, window_size=5, alpha_still=0.08, alpha_move=0.35, deadband=0.15, saccade_thresh=0.60)
iris_v_filter = RockSolidFilter(init_deg=-0.05, window_size=5, alpha_still=0.10, alpha_move=0.40, deadband=0.002, saccade_thresh=0.012)
iris_h_filter = RockSolidFilter(init_deg=0.0, window_size=5, alpha_still=0.10, alpha_move=0.40, deadband=0.002, saccade_thresh=0.012)

show_raw = False
show_hud = True
tracking_start = time.time()
live_frames_count = 0
raw_jumps = []
filt_jumps = []
prev_raw_pt = None
prev_filt_pt = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    timestamp_ms = int((time.time() - start_time) * 1000)
    pitch, yaw, iris_h, iris_v, prev_rvec, prev_tvec = get_gaze_pitch_yaw(frame, w, h, prev_rvec, prev_tvec, timestamp_ms)

    canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

    if pitch is not None and iris_h is not None and iris_v is not None:
        # Raw screen coordinate prediction
        raw_x, raw_y = predict_screen_point(pitch, yaw, iris_h, iris_v, coeffs_x, coeffs_y)

        # Filter in angle and iris space
        p_deg = np.degrees(pitch)
        y_deg = np.degrees(yaw)
        filt_p_deg = pitch_filter.update(p_deg)
        filt_y_deg = yaw_filter.update(y_deg)
        filt_ih    = iris_h_filter.update(iris_h)
        filt_iv    = iris_v_filter.update(iris_v)

        # Map stabilized signals to screen coordinates
        filt_x, filt_y = predict_screen_point(np.radians(filt_p_deg), np.radians(filt_y_deg), filt_ih, filt_iv, coeffs_x, coeffs_y)
        filt_x = max(0, min(SCREEN_W - 1, filt_x))
        filt_y = max(0, min(SCREEN_H - 1, filt_y))

        if prev_raw_pt is not None:
            raw_jumps.append(float(np.hypot(raw_x - prev_raw_pt[0], raw_y - prev_raw_pt[1])))
            filt_jumps.append(float(np.hypot(filt_x - prev_filt_pt[0], filt_y - prev_filt_pt[1])))
        prev_raw_pt = (raw_x, raw_y)
        prev_filt_pt = (filt_x, filt_y)
        live_frames_count += 1

        # Show raw noisy dot in yellow if toggled
        if show_raw:
            cv2.circle(canvas, (raw_x, raw_y), 9, (0, 255, 255), -1)
            cv2.putText(canvas, "RAW", (raw_x + 12, raw_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Smooth, stable gaze target (bullseye: red ring with bright center dot)
        cv2.circle(canvas, (filt_x, filt_y), 24, (0, 0, 255), 3)
        cv2.circle(canvas, (filt_x, filt_y), 5, (255, 255, 255), -1)

        if show_hud:
            fps_val = live_frames_count / max(1e-3, (time.time() - tracking_start))
            cv2.putText(canvas, f"pitch: {p_deg:.1f} | yaw: {y_deg:.1f} | iris_h: {filt_ih:.3f} | iris_v: {filt_iv:.3f}",
                        (30, SCREEN_H - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            raw_status = "ON" if show_raw else "OFF"
            cv2.putText(canvas, f"Dual-Axis Hybrid [ACTIVE] | FPS: {fps_val:.1f} | Raw overlay [t]: {raw_status} | Quit [q]",
                        (30, SCREEN_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)
    else:
        cv2.putText(canvas, "face not detected", (30, SCREEN_H - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Gaze Screen", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        show_raw = not show_raw
    elif key == ord('r'):
        show_hud = not show_hud

avg_fps = live_frames_count / max(1e-3, (time.time() - tracking_start))
save_tracking_session_log(calib_data, coeffs_x, coeffs_y, rmse_x, rmse_y,
                          live_frames_count, avg_fps, raw_jumps, filt_jumps)

cap.release()
cv2.destroyAllWindows()