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

device = torch.device("cpu")
model = GazeCNN().to(device)
model.load_state_dict(torch.load("gaze_model_p00.pth", map_location=device))
model.eval()

# MediaPipe setup
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
landmarker = vision.FaceLandmarker.create_from_options(options)

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
    cam_index, cam_name = find_camera_index("HP")
    print(f"Connecting to camera {cam_index} ('{cam_name}')...")
    cap = open_camera_robust(cam_index)
    if cap is None:
        print("ERROR: Failed to open webcam.")
        return

    cv2.namedWindow("Vertical Gaze Diagnostic", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Vertical Gaze Diagnostic", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    test_targets = [
        ("TOP-CENTER", SCREEN_W // 2, int(0.08 * SCREEN_H)),
        ("MID-CENTER", SCREEN_W // 2, int(0.50 * SCREEN_H)),
        ("BOT-CENTER", SCREEN_W // 2, int(0.92 * SCREEN_H)),
    ]

    print("\n" + "="*75)
    print("VERTICAL GAZE DIAGNOSTIC EXPERIMENT")
    print("="*75)
    print("INSTRUCTIONS:")
    print("1. Rest your chin on your hand or keep your head as still as possible.")
    print("2. A green circle will appear at the Top, Middle, and Bottom of your screen.")
    print("3. Look directly at the center of the dot with your eyes (head still).")
    print("4. When your gaze is steady, press the SPACEBAR.")
    print("5. Keep looking at the dot for ~1.5 seconds while it turns orange ('Recording').")
    print("6. Repeat for all 3 vertical points.")
    print("="*75 + "\n")

    start_time = time.time()
    results = []
    prev_rvec, prev_tvec = None, None

    for target_idx, (name, tx, ty) in enumerate(test_targets):
        # Waiting phase
        live_p, live_y = None, None
        while True:
            ret, frame = cap.read()
            if not ret: continue
            h, w, _ = frame.shape
            ts = int((time.time() - start_time) * 1000)

            # run detection to show live values
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_img, ts)
            if res.face_landmarks:
                lms = res.face_landmarks[0]
                cam_mat = get_camera_matrix(w, h)
                rvec, tvec = solve_head_pose(lms, w, h, cam_mat, prev_rvec, prev_tvec)
                if rvec is not None:
                    prev_rvec, prev_tvec = rvec, tvec
                    try:
                        gray, _, _, hp, hy = normalize_single_eye(frame, cam_mat, DIST_COEFFS, rvec, tvec, [0, 0, 600], 'right')
                        t_img = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                        t_hp = torch.tensor([[hp, hy]], dtype=torch.float32)
                        with torch.no_grad():
                            pred = model(t_img, t_hp)[0]
                        live_p, live_y = np.degrees(pred[0].item()), np.degrees(pred[1].item())
                    except Exception:
                        pass

            canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.circle(canvas, (tx, ty), 35, (0, 255, 0), 3)
            cv2.circle(canvas, (tx, ty), 5, (255, 255, 255), -1)
            cv2.putText(canvas, f"Point {target_idx+1}/3: {name}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(canvas, "Keep head still, look at the dot, and press SPACE when steady (or 'q' to quit)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            if live_p is not None:
                cv2.putText(canvas, f"Live Model Output: Pitch = {live_p:+.1f} deg,  Yaw = {live_y:+.1f} deg", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Vertical Gaze Diagnostic", canvas)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                cap.release(); cv2.destroyAllWindows(); return
            if k == ord(' '):
                break

        # Recording phase (1.5 seconds)
        rec_start = time.time()
        samples = []
        saved_patch = None
        while time.time() - rec_start < 1.5:
            ret, frame = cap.read()
            if not ret: continue
            h, w, _ = frame.shape
            ts = int((time.time() - start_time) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_img, ts)
            if not res.face_landmarks: continue
            lms = res.face_landmarks[0]
            cam_mat = get_camera_matrix(w, h)
            rvec, tvec = solve_head_pose(lms, w, h, cam_mat, prev_rvec, prev_tvec)
            if rvec is None: continue
            prev_rvec, prev_tvec = rvec, tvec

            try:
                gray, _, _, hp, hy = normalize_single_eye(frame, cam_mat, DIST_COEFFS, rvec, tvec, [0, 0, 600], 'right')
            except Exception:
                continue

            if saved_patch is None:
                saved_patch = gray.copy()

            t_img = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            t_hp = torch.tensor([[hp, hy]], dtype=torch.float32)
            with torch.no_grad():
                pred = model(t_img, t_hp)[0]

            samples.append({
                "pitch": np.degrees(pred[0].item()),
                "yaw": np.degrees(pred[1].item()),
                "head_pitch": np.degrees(hp),
                "head_yaw": np.degrees(hy),
                "tvec_y": tvec[1, 0] if tvec.ndim == 2 else tvec[1],
            })

            canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.circle(canvas, (tx, ty), 35, (0, 165, 255), 3)
            cv2.circle(canvas, (tx, ty), 5, (255, 255, 255), -1)
            cv2.putText(canvas, f"Recording {name}... ({len(samples)} frames)", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            cv2.imshow("Vertical Gaze Diagnostic", canvas)
            cv2.waitKey(1)

        patch_filename = f"diagnostic_patch_{name}.png"
        if saved_patch is not None:
            cv2.imwrite(patch_filename, saved_patch)

        if samples:
            avg_p = np.mean([s["pitch"] for s in samples])
            std_p = np.std([s["pitch"] for s in samples])
            avg_y = np.mean([s["yaw"] for s in samples])
            avg_hp = np.mean([s["head_pitch"] for s in samples])
            avg_ty = np.mean([s["tvec_y"] for s in samples])
            results.append((name, avg_p, std_p, avg_y, avg_hp, avg_ty, len(samples), patch_filename))

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*82)
    print(f"{'Target Position':<14} | {'Pred Pitch':<16} | {'Pred Yaw':<10} | {'Head Pitch':<12} | {'Head Y (mm)':<12}")
    print("="*82)
    for r in results:
        print(f"{r[0]:<14} | {r[1]:+6.2f}° ± {r[2]:4.2f}° | {r[3]:+6.2f}°   | {r[4]:+6.2f}°     | {r[5]:6.1f} mm")
    print("="*82)
    print("\nSaved eye patches:")
    for r in results:
        print(f"  - {r[7]}")
    print("\nDiagnostic complete! Please copy and share the table above.")

if __name__ == "__main__":
    main()
