"""
test_crop_alignment.py
Controlled offline A/B experiment to investigate the crop-alignment hypothesis for vertical gaze estimation.

Version A: Standard 3D normalization (cy = 18.0 px in 60x36 patch).
Version B: Centered 3D normalization (cy = 18.0 + 12.5 = 30.5 px).

Tests the hypothesis:
Does shifting the eye crop vertically into the center of the 60x36 patch restore vertical pitch sensitivity in the CNN?
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from normalize import (
    FACE_MODEL,
    rodrigues_to_matrix,
    get_eye_center_3d,
    NORM_DISTANCE,
    NORM_FOCAL_LEN,
    EYE_PATCH_SIZE,
)

# GazeCNN architecture matching live pipeline
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

# MediaPipe landmark indices
RIGHT_OUTER = 33
RIGHT_INNER = 133
RIGHT_IRIS = 468
FACE_LANDMARK_INDICES = [33, 133, 362, 263, 61, 291, 1]
DIST_COEFFS = np.zeros((4, 1))

def get_camera_matrix(w, h):
    focal_length = w
    return np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]], dtype=np.float64)

def solve_head_pose(lms, w, h, camera_matrix):
    image_points = np.array([[lms[i].x * w, lms[i].y * h] for i in FACE_LANDMARK_INDICES], dtype=np.float64)
    ok, rvec_init, tvec_init = cv2.solvePnP(FACE_MODEL, image_points, camera_matrix, DIST_COEFFS, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None, None
    ok, rvec, tvec = cv2.solvePnP(FACE_MODEL, image_points, camera_matrix, DIST_COEFFS,
                                 rvec=rvec_init, tvec=tvec_init, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    return (rvec, tvec) if ok else (None, None)

def project_point_to_patch(u_raw, v_raw, W, patch_w=60):
    """
    Applies homography W and right-eye horizontal flip to map raw image coordinates to patch coordinates.
    """
    p = W @ np.array([u_raw, v_raw, 1.0], dtype=np.float64)
    u_unflip = p[0] / p[2]
    v_unflip = p[1] / p[2]
    u_flip = (patch_w - 1) - u_unflip
    v_flip = v_unflip
    return u_flip, v_flip

def normalize_eye_versioned(image, camera_matrix, rvec, tvec, cy_offset=0.0):
    """
    Normalizes right eye patch using perspective warp.
    cy_offset: vertical shift in pixels added to normalized principal point cy.
    """
    R = rodrigues_to_matrix(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    eye_center = get_eye_center_3d(R, t, [0, 1])

    distance = np.linalg.norm(eye_center)
    if distance < 1e-6:
        raise ValueError("Degenerate eye center")

    z_axis = eye_center / distance
    head_x_axis = R[:, 0]
    y_axis = np.cross(z_axis, head_x_axis); y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis); x_axis /= np.linalg.norm(x_axis)
    R_norm = np.stack([x_axis, y_axis, z_axis], axis=0)

    S = np.diag([1.0, 1.0, NORM_DISTANCE / distance])
    
    # Normalized camera matrix with selectable cy
    fx = fy = NORM_FOCAL_LEN
    cx = EYE_PATCH_SIZE[0] / 2.0
    cy = (EYE_PATCH_SIZE[1] / 2.0) + cy_offset
    cam_norm = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    W = cam_norm @ S @ R_norm @ np.linalg.inv(camera_matrix)

    warped = cv2.warpPerspective(image, W, EYE_PATCH_SIZE)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = np.fliplr(gray)

    # Normalized head pose
    head_rot_norm = R_norm @ R
    head_pitch = np.arcsin(-head_rot_norm[1, 2])
    head_yaw = np.arctan2(-head_rot_norm[0, 2], head_rot_norm[2, 2])
    head_yaw = -head_yaw  # right eye flip

    return gray, head_pitch, head_yaw, W

def main():
    print("=" * 95)
    print("OFFLINE A/B EXPERIMENT: EYE-CROP ALIGNMENT HYPOTHESIS TEST")
    print("=" * 95)

    # 1. Verify files exist
    model_path = "gaze_model_p00.pth"
    task_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print(f"ERROR: Model checkpoint '{model_path}' not found.")
        return
    if not os.path.exists(task_path):
        print(f"ERROR: MediaPipe task file '{task_path}' not found.")
        return

    # 2. Load model
    device = torch.device("cpu")
    model = GazeCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Setup MediaPipe
    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    trials = [1, 2, 3]
    targets = ["TOP-CENTER", "MID-CENTER", "BOT-CENTER"]

    results = []
    comparisons = []

    print(f"\nEvaluating Version A (cy = 18.0 px) vs Version B (cy = 30.5 px, +12.5 px shift)...")

    for trial in trials:
        trial_res = {}
        for target in targets:
            fname = f"iris_raw_trial{trial}_{target}.png"
            if not os.path.exists(fname):
                print(f"WARNING: File {fname} not found, skipping.")
                continue

            img = cv2.imread(fname)
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection = landmarker.detect(mp_img)

            if not detection.face_landmarks:
                print(f"WARNING: No face landmarks detected in {fname}")
                continue

            lms = detection.face_landmarks[0]
            cam_mat = get_camera_matrix(w, h)
            rvec, tvec = solve_head_pose(lms, w, h, cam_mat)
            if rvec is None:
                print(f"WARNING: solvePnP failed for {fname}")
                continue

            u_out, v_out = lms[RIGHT_OUTER].x * w, lms[RIGHT_OUTER].y * h
            u_inn, v_inn = lms[RIGHT_INNER].x * w, lms[RIGHT_INNER].y * h
            u_iri, v_iri = lms[RIGHT_IRIS].x * w,  lms[RIGHT_IRIS].y * h

            # Version A (Standard cy = 18.0)
            patch_A, hp_A, hy_A, W_A = normalize_eye_versioned(img, cam_mat, rvec, tvec, cy_offset=0.0)
            t_img_A = torch.from_numpy(patch_A.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            t_hp_A = torch.tensor([[hp_A, hy_A]], dtype=torch.float32)
            with torch.no_grad():
                pred_A = model(t_img_A, t_hp_A)[0]
            pitch_A = np.degrees(pred_A[0].item())
            yaw_A = np.degrees(pred_A[1].item())

            # Version B (Centered cy = 18.0 + 12.5 = 30.5)
            patch_B, hp_B, hy_B, W_B = normalize_eye_versioned(img, cam_mat, rvec, tvec, cy_offset=12.5)
            t_img_B = torch.from_numpy(patch_B.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            t_hp_B = torch.tensor([[hp_B, hy_B]], dtype=torch.float32)
            with torch.no_grad():
                pred_B = model(t_img_B, t_hp_B)[0]
            pitch_B = np.degrees(pred_B[0].item())
            yaw_B = np.degrees(pred_B[1].item())

            # Landmark positions in patch
            iri_A = project_point_to_patch(u_iri, v_iri, W_A)
            c33_A = project_point_to_patch(u_out, v_out, W_A)
            c133_A = project_point_to_patch(u_inn, v_inn, W_A)

            iri_B = project_point_to_patch(u_iri, v_iri, W_B)
            c33_B = project_point_to_patch(u_out, v_out, W_B)
            c133_B = project_point_to_patch(u_inn, v_inn, W_B)

            # Save patches
            path_A = f"crop_trial{trial}_{target}_verA.png"
            path_B = f"crop_trial{trial}_{target}_verB.png"
            cv2.imwrite(path_A, patch_A)
            cv2.imwrite(path_B, patch_B)

            row = {
                "trial": trial,
                "target": target,
                "iri_A": iri_A,
                "c33_A": c33_A,
                "c133_A": c133_A,
                "mean_canthus_v_A": (c33_A[1] + c133_A[1]) / 2.0,
                "pitch_A": pitch_A,
                "yaw_A": yaw_A,
                "iri_B": iri_B,
                "c33_B": c33_B,
                "c133_B": c133_B,
                "mean_canthus_v_B": (c33_B[1] + c133_B[1]) / 2.0,
                "pitch_B": pitch_B,
                "yaw_B": yaw_B,
                "patch_A": patch_A,
                "patch_B": patch_B,
            }
            results.append(row)
            trial_res[target] = row

    # Print Landmark Alignment Verification Table
    print("\n" + "=" * 105)
    print("TABLE 1: LANDMARK POSITIONS IN 60x36 PATCH (VERIFYING VERTICAL EYE CENTERING)")
    print("=" * 105)
    print(f"{'Trial / Target':<20} | {'Ver A Iris (u, v)':<18} | {'Ver A Canthus Y':<15} | {'Ver B Iris (u, v)':<18} | {'Ver B Canthus Y':<15}")
    print("-" * 105)
    for r in results:
        t_label = f"T{r['trial']} {r['target']}"
        iri_A_str = f"({r['iri_A'][0]:4.1f}, {r['iri_A'][1]:4.1f})"
        iri_B_str = f"({r['iri_B'][0]:4.1f}, {r['iri_B'][1]:4.1f})"
        print(f"{t_label:<20} | {iri_A_str:<18} | y = {r['mean_canthus_v_A']:4.1f} px    | {iri_B_str:<18} | y = {r['mean_canthus_v_B']:4.1f} px")
    print("=" * 105)

    # Print CNN Predictions Table
    print("\n" + "=" * 105)
    print("TABLE 2: CNN PREDICTIONS (VERSION A vs VERSION B)")
    print("=" * 105)
    print(f"{'Trial':<6} | {'Target':<12} | {'Ver A Pitch':<13} | {'Ver A Yaw':<12} | {'Ver B Pitch':<13} | {'Ver B Yaw':<12}")
    print("-" * 105)
    for r in results:
        print(f"{r['trial']:<6} | {r['target']:<12} | {r['pitch_A']:+6.2f} deg    | {r['yaw_A']:+6.2f} deg  | {r['pitch_B']:+6.2f} deg    | {r['yaw_B']:+6.2f} deg")
    print("=" * 105)

    # Trial-by-Trial TOP to BOT Deltas
    print("\n" + "=" * 105)
    print("TABLE 3: TRIAL-BY-TRIAL TOP-TO-BOT PITCH DIFFERENCES")
    print("=" * 105)
    print(f"{'Trial':<6} | {'Ver A: TOP -> BOT':<28} | {'Ver A Delta':<14} | {'Ver B: TOP -> BOT':<28} | {'Ver B Delta':<14}")
    print("-" * 105)
    for trial in trials:
        t_rows = [r for r in results if r['trial'] == trial]
        top = [r for r in t_rows if r['target'] == 'TOP-CENTER'][0]
        mid = [r for r in t_rows if r['target'] == 'MID-CENTER'][0]
        bot = [r for r in t_rows if r['target'] == 'BOT-CENTER'][0]

        delta_A = bot['pitch_A'] - top['pitch_A']
        delta_B = bot['pitch_B'] - top['pitch_B']

        a_str = f"{top['pitch_A']:+.2f} deg -> {mid['pitch_A']:+.2f} deg -> {bot['pitch_A']:+.2f} deg"
        b_str = f"{top['pitch_B']:+.2f} deg -> {mid['pitch_B']:+.2f} deg -> {bot['pitch_B']:+.2f} deg"

        print(f"{trial:<6} | {a_str:<28} | {delta_A:+6.2f} deg   | {b_str:<28} | {delta_B:+6.2f} deg")
    print("-" * 105)

    # 3-Trial Averages
    top_A = np.mean([r['pitch_A'] for r in results if r['target'] == 'TOP-CENTER'])
    mid_A = np.mean([r['pitch_A'] for r in results if r['target'] == 'MID-CENTER'])
    bot_A = np.mean([r['pitch_A'] for r in results if r['target'] == 'BOT-CENTER'])

    top_B = np.mean([r['pitch_B'] for r in results if r['target'] == 'TOP-CENTER'])
    mid_B = np.mean([r['pitch_B'] for r in results if r['target'] == 'MID-CENTER'])
    bot_B = np.mean([r['pitch_B'] for r in results if r['target'] == 'BOT-CENTER'])

    avg_a_str = f"{top_A:+.2f} deg -> {mid_A:+.2f} deg -> {bot_A:+.2f} deg"
    avg_b_str = f"{top_B:+.2f} deg -> {mid_B:+.2f} deg -> {bot_B:+.2f} deg"
    print(f"{'MEAN':<6} | {avg_a_str:<28} | {(bot_A - top_A):+6.2f} deg   | {avg_b_str:<28} | {(bot_B - top_B):+6.2f} deg")
    print("=" * 105)

    # Create composite visual comparison montage
    # 3 trials (rows), 3 targets (cols), each cell showing Ver A on top, Ver B on bottom
    scale = 3  # enlarge patches for visual clarity (60x36 -> 180x108)
    cell_w, cell_h = 60 * scale, 36 * scale
    montage = np.zeros((cell_h * 2 * 3 + 120, cell_w * 3 + 100, 3), dtype=np.uint8)
    montage.fill(40)

    for trial_idx, trial in enumerate(trials):
        for target_idx, target in enumerate(targets):
            row_data = [r for r in results if r['trial'] == trial and r['target'] == target][0]
            patch_A_large = cv2.resize(row_data['patch_A'], (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
            patch_B_large = cv2.resize(row_data['patch_B'], (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)

            pA_bgr = cv2.cvtColor(patch_A_large, cv2.COLOR_GRAY2BGR)
            pB_bgr = cv2.cvtColor(patch_B_large, cv2.COLOR_GRAY2BGR)

            # Label on images
            cv2.putText(pA_bgr, f"Ver A: {row_data['pitch_A']:+.1f} deg", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(pB_bgr, f"Ver B: {row_data['pitch_B']:+.1f} deg", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            x_offset = 80 + target_idx * (cell_w + 10)
            y_offset = 40 + trial_idx * (cell_h * 2 + 30)

            montage[y_offset : y_offset + cell_h, x_offset : x_offset + cell_w] = pA_bgr
            montage[y_offset + cell_h + 2 : y_offset + cell_h * 2 + 2, x_offset : x_offset + cell_w] = pB_bgr

    # Headers
    for target_idx, target in enumerate(targets):
        x_offset = 80 + target_idx * (cell_w + 10) + 30
        cv2.putText(montage, target, (x_offset, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for trial_idx, trial in enumerate(trials):
        y_offset = 40 + trial_idx * (cell_h * 2 + 30) + 40
        cv2.putText(montage, f"Trial {trial}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(montage, "A (top)", (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(montage, "B (bot)", (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    montage_path = "crop_ab_comparison_montage.png"
    cv2.imwrite(montage_path, montage)
    print(f"\nVisual comparison montage saved to '{montage_path}'.")

if __name__ == "__main__":
    main()
