import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
import time

from normalize import normalize_single_eye, FACE_MODEL

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
# VERIFY THIS VISUALLY via the magenta dots (DEBUG_SHOW_MODEL_POINTS) before trusting results.
FACE_LANDMARK_INDICES = [33, 133, 362, 263, 61, 291]
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 33, 133

DEBUG_SHOW_MODEL_POINTS = True   # draws the 6 solvePnP correspondence points each frame
EYE_TO_USE = 'right'             # must match RIGHT_EYE_OUTER/INNER above and normalize_single_eye's eye_side arg

DIST_COEFFS = np.zeros((4, 1))   # assume no lens distortion - reasonable for most webcams


def get_camera_matrix(w, h):
    # Approximate intrinsics for a generic webcam (not calibrated like MPIIFaceGaze's per-person
    # Camera.mat files were). This is a known extra source of error vs. offline-trained accuracy -
    # worth noting as a limitation in the report.
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

    # Plausibility check: reject a solution that jumped too far from the last frame
    # (guards against the flip/instability issue seen earlier in test_headpose.py).
    # These thresholds are a starting point - tune by watching debug output.
    if prev_rvec is not None:
        rot_jump = np.linalg.norm(rvec - prev_rvec)
        trans_jump = np.linalg.norm(tvec - prev_tvec)
        if rot_jump > 0.5 or trans_jump > 80:
            return prev_rvec, prev_tvec

    return rvec, tvec


cap = cv2.VideoCapture(0)
start_time = time.time()
prev_rvec, prev_tvec = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # NOTE: no mirror flip here. All detection/solvePnP/normalization below runs on
    # the raw, unmirrored frame to match normalize.py's coordinate convention (which
    # assumes a real, unmirrored camera image - mirroring is a reflection, not a
    # rigid transform, and breaks solvePnP's geometry if applied before detection).
    # A mirrored copy is created only for on-screen display, at the very end.
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    eye_patch = None
    # All overlays are drawn on this mirrored copy, using mirrored coordinates -
    # NOT by flipping the frame after drawing (that also mirrors text, making it
    # unreadable, as seen in testing).
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

            # gaze_target_3d is only used by normalize_single_eye to compute the pitch/yaw
            # LABEL for training data. At live inference time we don't know where the person
            # is looking (that's what the model is predicting) - so this is a placeholder.
            # The eye-patch crop and head_pitch/head_yaw outputs don't depend on it, only the
            # returned pitch/yaw do, and we discard those two below.
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

            cv2.imwrite('debug_eye_patch.png', eye_patch)

    cv2.imshow('Live Gaze', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()