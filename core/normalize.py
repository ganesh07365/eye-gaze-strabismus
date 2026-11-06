# writefile normalize.py
import numpy as np
import cv2

# 3-D generic face model in millimetres (X right, Y down, Z toward camera).
# Coordinates match the MediaPipe landmark order used in FACE_LANDMARK_INDICES:
#   [33, 133, 362, 263, 61, 291, 1]  (right-outer, right-inner, left-inner,
#    left-outer, mouth-right, mouth-left, nose-tip)
#
# FIX (2024): mouth Y corrected from 30.0 mm to 65.0 mm.
#   The old value (30 mm) placed the mouth only one-third of the way down the
#   face, far too close to the eyes.  Human anatomy puts the mouth corners
#   ~65-70 mm below the inter-canthal plane.  The under-sized Y caused
#   solvePnP to project the 3-D eye centre ~15 px too low in camera space,
#   which jammed the eye into rows 4-5 of the 60x36 normalised patch instead
#   of the intended row 18.  Correcting to 65.0 mm restores proper centering.
FACE_MODEL = np.array([
    [-46.5,  0.0,  0.0],   # right eye outer corner
    [-16.5,  0.0,  0.0],   # right eye inner corner
    [ 16.5,  0.0,  0.0],   # left eye inner corner
    [ 46.5,  0.0,  0.0],   # left eye outer corner
    [-25.0, 65.0,  0.0],   # mouth right corner  (was 30.0 mm — anatomically wrong)
    [ 25.0, 65.0,  0.0],   # mouth left corner   (was 30.0 mm — anatomically wrong)
    [  0.0, 15.0, -30.0],  # nose tip — the ONLY non-planar point (protrudes toward camera).
                           # This single off-plane point removes the flat-face pose ambiguity.
], dtype=np.float64)

RIGHT_EYE_IDX = [0, 1]
LEFT_EYE_IDX = [2, 3]
NORM_DISTANCE = 600.0
NORM_FOCAL_LEN = 960.0
EYE_PATCH_SIZE = (60, 36)

def rodrigues_to_matrix(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R

def get_eye_center_3d(R, t, eye_indices):
    pts = FACE_MODEL[eye_indices]
    pts_cam = (R @ pts.T).T + t.reshape(1, 3)
    return pts_cam.mean(axis=0)

def build_normalized_camera_matrix():
    fx = fy = NORM_FOCAL_LEN
    cx, cy = EYE_PATCH_SIZE[0] / 2.0, EYE_PATCH_SIZE[1] / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

def normalize_single_eye(image, camera_matrix, dist_coeffs, rvec, tvec,
                          gaze_target_3d, eye_side):
    R = rodrigues_to_matrix(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)

    eye_idx = RIGHT_EYE_IDX if eye_side == 'right' else LEFT_EYE_IDX
    eye_center = get_eye_center_3d(R, t, eye_idx)

    distance = np.linalg.norm(eye_center)
    if distance < 1e-6:
        raise ValueError("Degenerate eye center")

    z_axis = eye_center / distance
    head_x_axis = R[:, 0]
    y_axis = np.cross(z_axis, head_x_axis); y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis); x_axis /= np.linalg.norm(x_axis)
    R_norm = np.stack([x_axis, y_axis, z_axis], axis=0)

    S = np.diag([1.0, 1.0, NORM_DISTANCE / distance])
    cam_norm = build_normalized_camera_matrix()
    W = cam_norm @ S @ R_norm @ np.linalg.inv(camera_matrix)

    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    warped = cv2.warpPerspective(undistorted, W, EYE_PATCH_SIZE)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    gaze_vec = np.asarray(gaze_target_3d, dtype=np.float64) - eye_center
    gaze_vec = gaze_vec / np.linalg.norm(gaze_vec)
    gaze_norm = R_norm @ gaze_vec

    pitch = np.arcsin(-gaze_norm[1])
    yaw = np.arctan2(-gaze_norm[0], -gaze_norm[2])

    # also return normalized head pose (needed as auxiliary CNN input)
    head_rot_norm = R_norm @ R
    hr, _ = cv2.Rodrigues(head_rot_norm)
    head_pitch = np.arcsin(-head_rot_norm[1, 2])
    head_yaw = np.arctan2(-head_rot_norm[0, 2], head_rot_norm[2, 2])

    if eye_side == 'right':
        gray = np.fliplr(gray)
        yaw = -yaw
        head_yaw = -head_yaw

    return gray, pitch, yaw, head_pitch, head_yaw