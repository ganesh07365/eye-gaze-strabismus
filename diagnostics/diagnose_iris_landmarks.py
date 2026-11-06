import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
import time
import ctypes
import os

from normalize import normalize_single_eye, FACE_MODEL

# Screen metrics (Windows)
user32 = ctypes.windll.user32
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)

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
# Right eye corners and pupil:
# Landmark 33: Right eye outer corner (subject's right)
# Landmark 133: Right eye inner corner (subject's right)
# Landmark 468: Right iris center / pupil
RIGHT_OUTER = 33
RIGHT_INNER = 133
RIGHT_IRIS = 468

FACE_LANDMARK_INDICES = [33, 133, 362, 263, 61, 291, 1]
DIST_COEFFS = np.zeros((4, 1))

def get_camera_matrix(w, h):
    focal_length = w
    return np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]], dtype=np.float64)

def solve_head_pose(lms, w, h, camera_matrix, prev_rvec, prev_tvec):
    image_points = np.array([[lms[i].x * w, lms[i].y * h] for i in FACE_LANDMARK_INDICES], dtype=np.float64)
    use_guess = prev_rvec is not None and prev_tvec is not None
    if not use_guess:
        ok, rvec_init, tvec_init = cv2.solvePnP(FACE_MODEL, image_points, camera_matrix, DIST_COEFFS, flags=cv2.SOLVEPNP_EPNP)
        if not ok: return None, None
    else:
        rvec_init, tvec_init = prev_rvec.copy(), prev_tvec.copy()
    ok, rvec, tvec = cv2.solvePnP(FACE_MODEL, image_points, camera_matrix, DIST_COEFFS,
                                 rvec=rvec_init, tvec=tvec_init, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    return (rvec, tvec) if ok else (None, None)

def compute_normalized_iris_ratio(lms, w, h):
    """
    Computes scale-invariant vertical iris displacement relative to the eye corner axis:
    - Measures perpendicular distance from iris center (468) to the line connecting eye corners (33 and 133).
    - Normalizes by inter-canthal distance (eye width) ||P_inner - P_outer||.
    - Positive = iris is lower (downward gaze). Negative = iris is higher (upward gaze).
    """
    if len(lms) <= RIGHT_IRIS:
        return None, None, None

    p_outer = np.array([lms[RIGHT_OUTER].x * w, lms[RIGHT_OUTER].y * h], dtype=np.float64)
    p_inner = np.array([lms[RIGHT_INNER].x * w, lms[RIGHT_INNER].y * h], dtype=np.float64)
    p_iris  = np.array([lms[RIGHT_IRIS].x * w,  lms[RIGHT_IRIS].y * h],  dtype=np.float64)

    canthi_vec = p_inner - p_outer
    eye_width = np.linalg.norm(canthi_vec)
    if eye_width < 1e-4:
        return None, None, None

    # Unit vector along eye axis
    u = canthi_vec / eye_width
    # Perpendicular vector pointing downward (+Y in image)
    normal_down = np.array([-u[1], u[0]], dtype=np.float64)

    # Midpoint of canthi
    p_mid = 0.5 * (p_outer + p_inner)

    # Signed perpendicular distance from canthal axis (in pixels)
    vert_offset_px = np.dot(p_iris - p_mid, normal_down)

    # Scale-invariant normalized ratio (unitless)
    ratio = vert_offset_px / eye_width

    return ratio, vert_offset_px, eye_width

def find_camera_index(name_hint):
    try:
        from pygrabber.dshow_graph import FilterGraph
        devices = FilterGraph().get_input_devices()
        for i, name in enumerate(devices):
            if name_hint.lower() in name.lower():
                return i, name
    except Exception:
        pass
    return 0, "Default Camera"

def open_camera_robust(index, width=640, height=480):
    for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None and frame.std() > 5:
                return cap
            time.sleep(0.03)
        cap.release()
    return None

def main():
    # 1. Load model
    device = torch.device("cpu")
    model = GazeCNN().to(device)
    if os.path.exists("gaze_model_p00.pth"):
        model.load_state_dict(torch.load("gaze_model_p00.pth", map_location=device))
        model.eval()
    else:
        print("ERROR: gaze_model_p00.pth not found.")
        return

    # 2. Setup MediaPipe FaceLandmarker
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # 3. Connect to camera
    cam_index, cam_name = find_camera_index("HP")
    print(f"Opening camera {cam_index} ('{cam_name}')...")
    cap = open_camera_robust(cam_index)
    if cap is None:
        print("ERROR: Failed to open webcam.")
        return

    cv2.namedWindow("Iris Landmark Diagnostic", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Iris Landmark Diagnostic", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    NUM_TRIALS = 3
    test_targets = [
        ("TOP-CENTER", SCREEN_W // 2, int(0.08 * SCREEN_H)),
        ("MID-CENTER", SCREEN_W // 2, int(0.50 * SCREEN_H)),
        ("BOT-CENTER", SCREEN_W // 2, int(0.92 * SCREEN_H)),
    ]

    print("\n" + "="*85)
    print("MEDIA PIPE RAW IRIS LANDMARK MULTI-TRIAL EXPERIMENT (3 COMPLETE TRIALS)")
    print("="*85)
    print("CRITICAL INSTRUCTIONS FOR REPEATABILITY:")
    print("1. Maintain a FIXED sitting distance (~50-55 cm from screen).")
    print("2. Rest your chin firmly on your palm/hand with elbow on desk to PREVENT head nods.")
    print("3. Move ONLY your eyes to look directly at the dot.")
    print("4. Sequence: 3 complete trials of [TOP -> MID -> BOT] (9 target recordings total).")
    print("5. For each dot: look at it steadily, press SPACEBAR, and hold for 2.0s.")
    print("="*85 + "\n")

    start_time = time.time()
    trial_records = []
    prev_rvec, prev_tvec = None, None

    for trial_idx in range(1, NUM_TRIALS + 1):
        print(f"\n>>> STARTING TRIAL {trial_idx} OF {NUM_TRIALS} <<<")
        for target_idx, (name, tx, ty) in enumerate(test_targets):
            # Waiting phase
            live_ratio = None
            live_width = None
            while True:
                ret, frame = cap.read()
                if not ret: continue
                h, w, _ = frame.shape
                ts = int((time.time() - start_time) * 1000)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = landmarker.detect_for_video(mp_img, ts)
                if res.face_landmarks:
                    lms = res.face_landmarks[0]
                    ratio, offset_px, eye_w = compute_normalized_iris_ratio(lms, w, h)
                    if ratio is not None:
                        live_ratio = ratio
                        live_width = eye_w

                canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
                cv2.circle(canvas, (tx, ty), 35, (0, 255, 0), 3)
                cv2.circle(canvas, (tx, ty), 5, (255, 255, 255), -1)
                cv2.putText(canvas, f"Trial {trial_idx}/{NUM_TRIALS} - Target {target_idx+1}/3: {name}", (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
                cv2.putText(canvas, "Keep head still at constant distance. Look at dot, press SPACEBAR when steady (or 'q' to quit)",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
                if live_ratio is not None and live_width is not None:
                    cv2.putText(canvas, f"Live Iris Ratio: {live_ratio:+.4f} | Eye Width: {live_width:.1f}px",
                                (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Iris Landmark Diagnostic", canvas)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    cap.release(); cv2.destroyAllWindows(); return
                if k == ord(' '):
                    break

            # Recording phase (2.0 seconds)
            rec_start = time.time()
            samples = []
            saved_raw_frame = None
            saved_norm_patch = None

            while time.time() - rec_start < 2.0:
                ret, frame = cap.read()
                if not ret: continue
                h, w, _ = frame.shape
                ts = int((time.time() - start_time) * 1000)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = landmarker.detect_for_video(mp_img, ts)
                if not res.face_landmarks:
                    continue

                lms = res.face_landmarks[0]
                ratio, offset_px, eye_w = compute_normalized_iris_ratio(lms, w, h)
                if ratio is None:
                    continue

                # Solve head pose and normalized eye crop
                cam_mat = get_camera_matrix(w, h)
                rvec, tvec = solve_head_pose(lms, w, h, cam_mat, prev_rvec, prev_tvec)
                norm_patch = None
                pred_pitch, pred_yaw = None, None
                if rvec is not None:
                    prev_rvec, prev_tvec = rvec, tvec
                    try:
                        gray, _, _, hp, hy = normalize_single_eye(frame, cam_mat, DIST_COEFFS, rvec, tvec, [0, 0, 600], 'right')
                        norm_patch = gray
                        t_img = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                        t_hp = torch.tensor([[hp, hy]], dtype=torch.float32)
                        with torch.no_grad():
                            pred = model(t_img, t_hp)[0]
                        pred_pitch = np.degrees(pred[0].item())
                        pred_yaw = np.degrees(pred[1].item())
                    except Exception:
                        pass

                if saved_raw_frame is None:
                    annotated = frame.copy()
                    p_out = (int(lms[RIGHT_OUTER].x * w), int(lms[RIGHT_OUTER].y * h))
                    p_inn = (int(lms[RIGHT_INNER].x * w), int(lms[RIGHT_INNER].y * h))
                    p_iri = (int(lms[RIGHT_IRIS].x * w),  int(lms[RIGHT_IRIS].y * h))
                    cv2.line(annotated, p_out, p_inn, (0, 255, 0), 1)
                    cv2.circle(annotated, p_out, 3, (0, 255, 0), -1)
                    cv2.circle(annotated, p_inn, 3, (0, 255, 0), -1)
                    cv2.circle(annotated, p_iri, 3, (0, 0, 255), -1)
                    saved_raw_frame = annotated

                if saved_norm_patch is None and norm_patch is not None:
                    saved_norm_patch = norm_patch.copy()

                samples.append({
                    "ratio": ratio,
                    "offset_px": offset_px,
                    "eye_w": eye_w,
                    "pred_pitch": pred_pitch,
                    "pred_yaw": pred_yaw,
                })

                canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
                cv2.circle(canvas, (tx, ty), 35, (0, 165, 255), 3)
                cv2.circle(canvas, (tx, ty), 5, (255, 255, 255), -1)
                cv2.putText(canvas, f"Recording Trial {trial_idx}/3 - {name}... ({len(samples)} frames)",
                            (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255), 2)
                cv2.imshow("Iris Landmark Diagnostic", canvas)
                cv2.waitKey(1)

            # Save representative images
            raw_path = f"iris_raw_trial{trial_idx}_{name}.png"
            patch_path = f"iris_patch_trial{trial_idx}_{name}.png"
            if saved_raw_frame is not None: cv2.imwrite(raw_path, saved_raw_frame)
            if saved_norm_patch is not None: cv2.imwrite(patch_path, saved_norm_patch)

            # Reliability check
            valid_cnt = len(samples)
            if valid_cnt >= 15:
                status = "OK"
                avg_ratio = float(np.mean([s["ratio"] for s in samples]))
                std_ratio = float(np.std([s["ratio"] for s in samples]))
                avg_offset = float(np.mean([s["offset_px"] for s in samples]))
                std_offset = float(np.std([s["offset_px"] for s in samples]))
                avg_width = float(np.mean([s["eye_w"] for s in samples]))
                pitches = [s["pred_pitch"] for s in samples if s["pred_pitch"] is not None]
                yaws = [s["pred_yaw"] for s in samples if s["pred_yaw"] is not None]
                avg_pitch = float(np.mean(pitches)) if pitches else np.nan
                avg_yaw = float(np.mean(yaws)) if yaws else np.nan
            elif valid_cnt > 0:
                status = "UNRELIABLE (few frames)"
                avg_ratio = float(np.mean([s["ratio"] for s in samples]))
                std_ratio = float(np.std([s["ratio"] for s in samples]))
                avg_offset = float(np.mean([s["offset_px"] for s in samples]))
                std_offset = float(np.std([s["offset_px"] for s in samples]))
                avg_width = float(np.mean([s["eye_w"] for s in samples]))
                avg_pitch = np.nan
                avg_yaw = np.nan
            else:
                status = "FAILED (no face detected)"
                avg_ratio, std_ratio = np.nan, np.nan
                avg_offset, std_offset = np.nan, np.nan
                avg_width, avg_pitch, avg_yaw = np.nan, np.nan, np.nan

            trial_records.append({
                "trial": trial_idx,
                "target": name,
                "norm_ratio_mean": avg_ratio,
                "norm_ratio_std": std_ratio,
                "offset_px_mean": avg_offset,
                "offset_px_std": std_offset,
                "eye_width_px_mean": avg_width,
                "cnn_pitch_mean": avg_pitch,
                "cnn_yaw_mean": avg_yaw,
                "valid_frames": valid_cnt,
                "status": status,
            })
            print(f"  [Trial {trial_idx}] {name:<11} | Ratio={avg_ratio:+.4f} | RawOffset={avg_offset:+5.2f}px | EyeWidth={avg_width:4.1f}px | Pitch={avg_pitch:+5.2f}° | Status={status}")

    cap.release()
    cv2.destroyAllWindows()

    # 1. Save results to CSV
    csv_filename = "iris_diagnostic_trials.csv"
    with open(csv_filename, "w") as f:
        f.write("trial,target,norm_ratio_mean,norm_ratio_std,offset_px_mean,offset_px_std,eye_width_px_mean,cnn_pitch_mean,cnn_yaw_mean,valid_frames,status\n")
        for r in trial_records:
            f.write(f"{r['trial']},{r['target']},{r['norm_ratio_mean']:.6f},{r['norm_ratio_std']:.6f},"
                    f"{r['offset_px_mean']:.4f},{r['offset_px_std']:.4f},{r['eye_width_px_mean']:.2f},"
                    f"{r['cnn_pitch_mean']:.2f},{r['cnn_yaw_mean']:.2f},{r['valid_frames']},{r['status']}\n")
    print(f"\nDetailed per-trial measurements written to: {csv_filename}")

    # 2. Print All Individual Trials Table
    print("\n" + "="*105)
    print("INDIVIDUAL TRIAL MEASUREMENTS")
    print("="*105)
    print(f"{'Trial':<6} | {'Target':<11} | {'Norm Ratio':<18} | {'Raw Offset (px)':<18} | {'Eye Width':<10} | {'CNN Pitch':<11} | {'Frames':<7} | {'Status'}")
    print("-"*105)
    for r in trial_records:
        if r["status"] == "OK":
            ratio_str = f"{r['norm_ratio_mean']:+7.4f} ± {r['norm_ratio_std']:.4f}"
            offset_str = f"{r['offset_px_mean']:+5.2f} ± {r['offset_px_std']:.2f} px"
            pitch_str = f"{r['cnn_pitch_mean']:+5.2f}°"
            width_str = f"{r['eye_width_px_mean']:4.1f} px"
        else:
            ratio_str = "N/A"
            offset_str = "N/A"
            pitch_str = "N/A"
            width_str = f"{r['eye_width_px_mean']:4.1f} px" if not np.isnan(r['eye_width_px_mean']) else "N/A"
        print(f"#{r['trial']:<5} | {r['target']:<11} | {ratio_str:<18} | {offset_str:<18} | {width_str:<10} | {pitch_str:<11} | {r['valid_frames']:<7} | {r['status']}")
    print("="*105)

    # 3. Print Grouped Summary Across the 3 Trials
    print("\n" + "="*105)
    print("GROUPED SUMMARY ACROSS ALL 3 TRIALS (MEAN ± STD ACROSS TRIALS)")
    print("="*105)
    print(f"{'Target':<12} | {'Norm Ratio (Mean ± Std)':<25} | {'Raw Offset (Mean ± Std)':<25} | {'Eye Width':<12} | {'CNN Pitch':<12}")
    print("-"*105)
    grouped = {}
    for t_name in ["TOP-CENTER", "MID-CENTER", "BOT-CENTER"]:
        recs = [r for r in trial_records if r["target"] == t_name and r["status"] == "OK"]
        if recs:
            ratios = [r["norm_ratio_mean"] for r in recs]
            offsets = [r["offset_px_mean"] for r in recs]
            widths = [r["eye_width_px_mean"] for r in recs]
            pitches = [r["cnn_pitch_mean"] for r in recs if not np.isnan(r["cnn_pitch_mean"])]

            mean_r, std_r = np.mean(ratios), np.std(ratios)
            mean_o, std_o = np.mean(offsets), np.std(offsets)
            mean_w, std_w = np.mean(widths), np.std(widths)
            mean_p, std_p = (np.mean(pitches), np.std(pitches)) if pitches else (np.nan, np.nan)

            grouped[t_name] = {"ratio": mean_r, "offset": mean_o, "pitch": mean_p}
            print(f"{t_name:<12} | {mean_r:+7.4f} ± {std_r:.4f}             | {mean_o:+5.2f} ± {std_o:.2f} px            | {mean_w:4.1f} ± {std_w:.1f}px | {mean_p:+5.2f} ± {std_p:.2f}°")
        else:
            print(f"{t_name:<12} | NO RELIABLE DATA (All trials failed or had <15 frames)")
    print("="*105)

    if "TOP-CENTER" in grouped and "BOT-CENTER" in grouped:
        delta_ratio = grouped["BOT-CENTER"]["ratio"] - grouped["TOP-CENTER"]["ratio"]
        delta_offset = grouped["BOT-CENTER"]["offset"] - grouped["TOP-CENTER"]["offset"]
        delta_pitch = grouped["BOT-CENTER"]["pitch"] - grouped["TOP-CENTER"]["pitch"]
        print(f"\nAVERAGE TOP-TO-BOTTOM DELTA ACROSS TRIALS:")
        print(f"  - Delta Normalized Iris Ratio: {delta_ratio:+.4f}")
        print(f"  - Delta Raw Pupil Offset:      {delta_offset:+.2f} pixels in raw frame")
        print(f"  - Delta CNN Predicted Pitch:   {delta_pitch:+.2f}°")
        print("="*105)

if __name__ == "__main__":
    main()
