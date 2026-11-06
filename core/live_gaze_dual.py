import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
import time

from normalize import normalize_single_eye, FACE_MODEL

# ============================================================
# CONFIG - cameras are found by NAME, not hardcoded index, since DroidCam's
# virtual camera index can shift between sessions depending on connection order.
# Adjust these substrings if your device names differ (check by running:
#   from pygrabber.dshow_graph import FilterGraph
#   print(FilterGraph().get_input_devices())
# ============================================================
EYE_CAMERA_NAME_HINT = "HP"          # substring matching your laptop webcam's name

USE_IP_WEBCAM_FOR_SCENE = True       # True = use the phone's IP Webcam URL below
                                      # False = fall back to DroidCam virtual device by name
SCENE_CAMERA_URL = "http://10.0.241.58:8080/video"
SCENE_CAMERA_NAME_HINT = "DroidCam"  # only used if USE_IP_WEBCAM_FOR_SCENE is False


def find_camera_index(name_hint):
    from pygrabber.dshow_graph import FilterGraph
    devices = FilterGraph().get_input_devices()
    for i, name in enumerate(devices):
        if name_hint.lower() in name.lower():
            return i, name
    return None, None


def open_network_camera(url, label=""):
    """Opens a network video stream (e.g. IP Webcam) and verifies it's giving real frames."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"[{label}] Failed to open stream at {url}")
        return None

    valid = False
    for _ in range(15):
        ret, frame = cap.read()
        if ret and frame is not None and frame.std() > 5:
            valid = True
            break
        time.sleep(0.05)

    if valid:
        print(f"[{label}] opened successfully via {url}")
        return cap

    cap.release()
    print(f"[{label}] connected but got no valid frames from {url} - "
          f"check the URL is reachable in a browser and the app is actively streaming.")
    return None


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

# Landmark indices, IN ORDER, matching FACE_MODEL rows 0-5 in normalize.py:
#   0: right eye outer corner   -> MediaPipe 33
#   1: right eye inner corner   -> MediaPipe 133
#   2: left eye inner corner    -> MediaPipe 362
#   3: left eye outer corner    -> MediaPipe 263
#   4: mouth right corner       -> MediaPipe 61
#   5: mouth left corner        -> MediaPipe 291
FACE_LANDMARK_INDICES = [33, 133, 362, 263, 61, 291]
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 33, 133

DEBUG_SHOW_MODEL_POINTS = True
EYE_TO_USE = 'right'

DIST_COEFFS = np.zeros((4, 1))


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
    rvec_init = prev_rvec.copy() if use_guess else np.zeros((3, 1))
    tvec_init = prev_tvec.copy() if use_guess else np.array([[0.0], [0.0], [600.0]])

    ok, rvec, tvec = cv2.solvePnP(
        FACE_MODEL, image_points, camera_matrix, DIST_COEFFS,
        rvec=rvec_init, tvec=tvec_init,
        useExtrinsicGuess=use_guess,
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


def open_camera_robust(index, width=640, height=480, label=""):
    """
    Tries multiple backends in order and picks the first one that produces
    real, varying video (not a blank/solid-color placeholder frame).
    This automates the manual DSHOW/MSMF trial-and-error from earlier debugging.
    """
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
        for _ in range(15):  # warm up + check several frames
            ret, frame = cap.read()
            if ret and frame is not None:
                std = frame.std()  # near-zero std = solid/blank frame (e.g. green placeholder)
                if std > 5:
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


EYE_CAMERA_INDEX, eye_name = find_camera_index(EYE_CAMERA_NAME_HINT)

if EYE_CAMERA_INDEX is None:
    print(f"Could not find a camera matching '{EYE_CAMERA_NAME_HINT}'. "
          f"Check EYE_CAMERA_NAME_HINT against your actual device list.")
    raise SystemExit(1)

print(f"Eye camera:   index {EYE_CAMERA_INDEX} ('{eye_name}')")
eye_cap = open_camera_robust(EYE_CAMERA_INDEX, label="EYE")

if USE_IP_WEBCAM_FOR_SCENE:
    print(f"Scene camera: IP Webcam at {SCENE_CAMERA_URL}")
    scene_cap = open_network_camera(SCENE_CAMERA_URL, label="SCENE")
else:
    SCENE_CAMERA_INDEX, scene_name = find_camera_index(SCENE_CAMERA_NAME_HINT)
    if SCENE_CAMERA_INDEX is None:
        print(f"Could not find a camera matching '{SCENE_CAMERA_NAME_HINT}'. "
              f"Is DroidCam Client connected?")
        raise SystemExit(1)
    print(f"Scene camera: index {SCENE_CAMERA_INDEX} ('{scene_name}')")
    scene_cap = open_camera_robust(SCENE_CAMERA_INDEX, label="SCENE")

if eye_cap is None or scene_cap is None:
    print("One or both cameras failed to open. Check indices/connections and rerun.")
    raise SystemExit(1)

start_time = time.time()
prev_rvec, prev_tvec = None, None

# simple FPS tracking to spot lag from the WiFi DroidCam connection
frame_count = 0
fps_timer = time.time()
eye_fps, scene_fps = 0.0, 0.0

while True:
    ret_eye, frame = eye_cap.read()
    ret_scene, scene_frame = scene_cap.read()

    if not ret_eye or not ret_scene:
        print("Lost a frame from one of the cameras - stopping.")
        break

    # ---- EYE CAMERA: full gaze pipeline (unchanged from live_gaze.py) ----
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    eye_patch = None
    display_frame = cv2.flip(frame, 1)

    def mirror_x(x):
        return w - x

    if result.face_landmarks:
        lms = result.face_landmarks[0]
        camera_matrix = get_camera_matrix(w, h)

        if DEBUG_SHOW_MODEL_POINTS:
            for idx_num, i in enumerate(FACE_LANDMARK_INDICES):
                pt = (mirror_x(int(lms[i].x * w)), int(lms[i].y * h))
                cv2.circle(display_frame, pt, 3, (255, 0, 255), -1)
                cv2.putText(display_frame, str(idx_num), (pt[0] + 4, pt[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        rvec, tvec = solve_head_pose(lms, w, h, camera_matrix, prev_rvec, prev_tvec)

        if rvec is not None:
            prev_rvec, prev_tvec = rvec, tvec
            try:
                gray, _, _, head_pitch, head_yaw = normalize_single_eye(
                    frame, camera_matrix, DIST_COEFFS, rvec, tvec,
                    gaze_target_3d=[0, 0, 600],
                    eye_side=EYE_TO_USE
                )
                eye_patch = gray
            except Exception:
                eye_patch = None

        if eye_patch is not None:
            img_tensor = torch.from_numpy(eye_patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            head_pose_tensor = torch.tensor([[head_pitch, head_yaw]], dtype=torch.float32)

            with torch.no_grad():
                pred = model(img_tensor, head_pose_tensor)[0]
            pred_pitch, pred_yaw = pred[0].item(), pred[1].item()

            cx, cy = int(lms[RIGHT_EYE_INNER].x * w + 50), int(lms[RIGHT_EYE_INNER].y * h)
            dx = int(-100 * np.sin(pred_yaw))
            dy = int(-100 * np.sin(pred_pitch))
            start_pt = (mirror_x(cx), cy)
            end_pt = (mirror_x(cx + dx), cy + dy)
            cv2.arrowedLine(display_frame, start_pt, end_pt, (0, 0, 255), 2)

            cv2.putText(display_frame, f"Gaze pitch:{np.degrees(pred_pitch):.1f} yaw:{np.degrees(pred_yaw):.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            eye_display = cv2.resize(eye_patch, (180, 108))
            display_frame[h - 118:h - 10, 10:190] = cv2.cvtColor(eye_display, cv2.COLOR_GRAY2BGR)

    cv2.putText(display_frame, f"eye cam FPS: {eye_fps:.1f}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.imshow('Eye Camera', display_frame)

    # ---- SCENE CAMERA: just display for now, no processing yet (that's Phase 5 Step 2) ----
    sh, sw, _ = scene_frame.shape
    cv2.putText(scene_frame, f"scene cam FPS: {scene_fps:.1f}", (10, sh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.imshow('Scene Camera', scene_frame)

    # ---- FPS bookkeeping ----
    frame_count += 1
    elapsed = time.time() - fps_timer
    if elapsed >= 1.0:
        eye_fps = frame_count / elapsed
        scene_fps = eye_fps  # both read in lockstep in this loop
        frame_count = 0
        fps_timer = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

eye_cap.release()
scene_cap.release()
cv2.destroyAllWindows()