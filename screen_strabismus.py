"""
screen_strabismus.py - Real-Time Bilateral Eye Alignment & Strabismus Screening
================================================================================
BTP Project: Webcam-Based Eye Tracking & Visual Training for Strabismus Screening

Features:
  1. Live Dual-Eye Picture-in-Picture (PiP) cards for Left and Right eyes.
  2. Sub-pixel scale- and rotation-invariant physical iris position tracking:
     - Horizontal Nasal/Temporal ratio (along canthal vector)
     - Vertical displacement ratio (perpendicular to canthal vector)
  3. Absolute (Target-anchored) and Relative (Inter-Ocular Disparity) alignment.
  4. Real-time diagnostic classification:
     - Orthophoria (Normal parallel alignment)
     - Unilateral Esotropia / Exotropia (Inward / Outward deviation)
     - Bilateral Esotropia / Exotropia (Both eyes deviated)
     - Hypertropia / Hypotropia (Vertical misalignment)
  5. Deviation measurement in clinical units:
     - Angular Deviation (Degrees: deg)
     - Ophthalmic Prism Diopters (Delta PD, where 1 deg ~ 1.75 PD)
  6. 9-Cardinal Gaze Clinical Diagnostic Protocol (Press 's' to run):
     - Tests all 9 diagnostic positions of gaze
     - Auto-saves full clinical screening report to diagnostics/results/
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import ctypes
from collections import deque

# ============================================================
# DISPLAY & ASSET SETUP
# ============================================================
user32 = ctypes.windll.user32
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def find_asset(filename):
    candidates = [
        os.path.join(PROJECT_ROOT, "models", filename),
        os.path.join(SCRIPT_DIR, filename),
        os.path.join(".", "models", filename),
        filename,
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return filename

# MediaPipe Face Landmarker
task_path = find_asset("face_landmarker.task")
base_options = python.BaseOptions(model_asset_path=task_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_facial_transformation_matrixes=True
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# ============================================================
# ANATOMICAL LANDMARK INDICES (478-mesh)
# ============================================================
# Right Eye (Subject's right, camera's left):
RIGHT_OUTER  = 33   # Temporal corner
RIGHT_INNER  = 133  # Nasal corner
RIGHT_TOP    = 159  # Upper eyelid
RIGHT_BOTTOM = 145  # Lower eyelid
RIGHT_IRIS   = 468  # Pupil / iris center

# Left Eye (Subject's left, camera's right):
LEFT_INNER   = 362  # Nasal corner
LEFT_OUTER   = 263  # Temporal corner
LEFT_TOP     = 386  # Upper eyelid
LEFT_BOTTOM  = 374  # Lower eyelid
LEFT_IRIS    = 473  # Pupil / iris center

# Clinical conversion factor: 1 degree ~ 1.75 Prism Diopters (PD)
# (Definition: 1 PD deflects a ray 1 cm at 1 m. At 1 m, tan(1°) ≈ 0.01746, so 1° ≈ 1.746 PD)
PD_PER_DEGREE = 1.75
# Scale factor converting normalised canthal ratio to angular degrees.
# Derivation: adult palpebral width ≈ 28 mm, corneal radius ≈ 12 mm.
# A ratio shift of 1.0 (full width, 28 mm) corresponds to arcsin(28/12) which is
# beyond physical range; over the clinically relevant ±0.15 range the relationship
# is approximately linear with slope ≈ arctan(28/12) × (180/π) ≈ 134°/ratio.
# NOTE: This is an uncalibrated geometric estimate. The live module should be
# regarded as a qualitative prototype until per-subject calibration is performed.
RATIO_TO_DEGREES = 134.0

# Diagnostic thresholds (in Prism Diopters):
NORMAL_PD_THRESH = 7.0        # < 7 PD is considered normal physiological alignment
STRABISMUS_PD_THRESH = 10.0   # > 10 PD is clinical strabismus

# ============================================================
# TWO-STAGE ADAPTIVE SIGNAL FILTER
# ============================================================
class RockSolidFilter:
    def __init__(self, init_val=0.0, window_size=5, alpha_still=0.10, alpha_move=0.45, deadband=0.002, saccade_thresh=0.012):
        self.buf = deque([float(init_val)] * window_size, maxlen=window_size)
        self.smooth = float(init_val)
        self.alpha_still = float(alpha_still)
        self.alpha_move = float(alpha_move)
        self.deadband = float(deadband)
        self.saccade_thresh = float(saccade_thresh)
        self.consecutive_moves = 0
        self.initialized = False

    def update(self, val):
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
            self.consecutive_moves = 0
            return self.smooth

        if abs_diff > self.saccade_thresh:
            self.consecutive_moves += 1
        else:
            self.consecutive_moves = 0

        alpha = self.alpha_move if self.consecutive_moves >= 2 else self.alpha_still
        self.smooth = self.smooth + alpha * diff
        return self.smooth


# ============================================================
# EYE METRIC EXTRACTION
# ============================================================
def extract_single_eye_metrics(lms, w, h, idx_outer, idx_inner, idx_top, idx_bottom, idx_iris):
    """
    Extracts scale- and rotation-invariant physical iris metrics:
    - horiz_ratio: displacement along canthal vector pointing NASALLY (+ = inward/nasal, - = outward/temporal)
    - vert_ratio: displacement along perpendicular vector pointing DOWNWARD (+ = downward, - = upward)
    - is_blink: True if palpebral fissure is closed/occluded
    - crop_box: (x1, y1, x2, y2) bounding box around eye for visual PiP card
    """
    p_outer  = np.array([lms[idx_outer].x * w,  lms[idx_outer].y * h],  dtype=np.float64)
    p_inner  = np.array([lms[idx_inner].x * w,  lms[idx_inner].y * h],  dtype=np.float64)
    p_top    = np.array([lms[idx_top].x * w,    lms[idx_top].y * h],    dtype=np.float64)
    p_bottom = np.array([lms[idx_bottom].x * w, lms[idx_bottom].y * h], dtype=np.float64)
    p_iris   = np.array([lms[idx_iris].x * w,   lms[idx_iris].y * h],   dtype=np.float64)

    # Canthal vector pointing NASALLY for both eyes
    canthi_vec = p_inner - p_outer
    eye_width = np.linalg.norm(canthi_vec)
    palpebral_height = np.linalg.norm(p_top - p_bottom)

    if eye_width < 1e-4:
        return 0.0, 0.0, True, None

    # Blink detection: palpebral aperture < 14% of width
    is_blink = (palpebral_height / eye_width) < 0.14

    # Horizontal unit vector pointing NASALLY (+ = nasal / inward turn)
    u_nasal = canthi_vec / eye_width
    # Vertical unit vector pointing DOWNWARD (+ = downward turn)
    # If eye is horizontal, u_nasal is (+1, 0) or (-1, 0) depending on side.
    # Normal down: rotated +90 deg towards +Y in image space
    if idx_outer == RIGHT_OUTER:
        # Right eye: outer is right-temple (smaller x), inner is nose (larger x) -> u_nasal points +x
        normal_down = np.array([-u_nasal[1], u_nasal[0]], dtype=np.float64)
    else:
        # Left eye: inner is nose (smaller x), outer is left-temple (larger x) -> inner-outer points -x
        normal_down = np.array([u_nasal[1], -u_nasal[0]], dtype=np.float64)

    p_mid = 0.5 * (p_outer + p_inner)
    d_iris = p_iris - p_mid

    horiz_offset = np.dot(d_iris, u_nasal)
    vert_offset  = np.dot(d_iris, normal_down)

    horiz_ratio = float(horiz_offset / eye_width)
    vert_ratio  = float(vert_offset / eye_width)

    # Eye bounding box for PiP card
    margin_x = int(eye_width * 0.40)
    margin_y = int(eye_width * 0.35)
    center_x = int(p_mid[0])
    center_y = int(p_mid[1])
    box = (
        max(0, center_x - margin_x),
        max(0, center_y - margin_y),
        min(w - 1, center_x + margin_x),
        min(h - 1, center_y + margin_y)
    )

    return horiz_ratio, vert_ratio, is_blink, box, (p_outer, p_inner, p_iris, p_mid)


def compute_strabismus_diagnosis(rh, rv, lh, lv, baseline_dh=0.0, baseline_dv=0.0):
    """
    Computes ocular alignment metrics and diagnosis.
    - rh, lh: nasal horizontal ratios (+ = inward/nasal, - = outward/temporal)
    - rv, lv: downward vertical ratios (+ = downward, - = upward)
    """
    # Convert ratios to degrees
    r_deg_h = rh * RATIO_TO_DEGREES
    l_deg_h = lh * RATIO_TO_DEGREES
    r_deg_v = rv * RATIO_TO_DEGREES
    l_deg_v = lv * RATIO_TO_DEGREES

    # Net inter-ocular disparity:
    # HORIZONTAL: Both eyes use nasal-positive horiz_ratio (u_nasal points toward nose
    # for each eye independently). When a person looks straight right, the right eye goes
    # temporal (rh < 0) and the left eye goes nasal (lh > 0), so lh - rh doubles instead
    # of cancelling — wrongly flagging a conjugate side-gaze as strabismus.
    # The correct misalignment signal is lh + rh:
    #   Normal conjugate gaze: lh ≈ +k, rh ≈ -k  →  lh + rh ≈ 0  (no misalignment)
    #   Esotropia (both eyes inward): lh > 0, rh > 0  →  lh + rh > 0
    #   Exotropia (both eyes outward): lh < 0, rh < 0  →  lh + rh < 0
    # VERTICAL: lv - rv is correct (both downward-positive, difference gives height mismatch).
    delta_deg_h = (lh + rh) * RATIO_TO_DEGREES - baseline_dh
    delta_deg_v = (lv - rv) * RATIO_TO_DEGREES - baseline_dv

    delta_pd_h = delta_deg_h * PD_PER_DEGREE
    delta_pd_v = delta_deg_v * PD_PER_DEGREE

    # Classification logic — uses corrected lh+rh disparity signal:
    # Positive delta_pd_h → net inward convergence (Esotropia)
    # Negative delta_pd_h → net outward divergence (Exotropia)

    # 1. Bilateral Esotropia — both individual nasal ratios are inward:
    if r_deg_h > 5.5 and l_deg_h > 5.5:
        category = "BILATERAL ESOTROPIA"
        severity = "Both eyes deviated inward (convergent squint)"
        color = (0, 0, 255)
    # 2. Bilateral Exotropia — both individual nasal ratios are outward:
    elif r_deg_h < -5.5 and l_deg_h < -5.5:
        category = "BILATERAL EXOTROPIA"
        severity = "Both eyes deviated outward (divergent squint)"
        color = (0, 140, 255)
    # 3. Esotropia — net convergent disparity above strabismus threshold:
    elif delta_pd_h > STRABISMUS_PD_THRESH:
        who = "Left" if l_deg_h > r_deg_h else "Right"
        category = "ESOTROPIA"
        severity = f"{who} Eye dominant inward deviation (+{delta_pd_h:.1f} PD)"
        color = (0, 80, 255)
    # 4. Exotropia — net divergent disparity above strabismus threshold:
    elif delta_pd_h < -STRABISMUS_PD_THRESH:
        who = "Right" if r_deg_h < l_deg_h else "Left"
        category = "EXOTROPIA"
        severity = f"{who} Eye dominant outward deviation ({abs(delta_pd_h):.1f} PD)"
        color = (0, 165, 255)
    # 5. Hypertropia — right eye sits higher than left (delta_v > 0 means R up / L down):
    elif delta_pd_v > STRABISMUS_PD_THRESH:
        category = "HYPERTROPIA"
        severity = f"Right Eye elevation / Left Eye depression ({abs(delta_pd_v):.1f} PD)"
        color = (0, 215, 255)
    # 6. Hypotropia — right eye sits lower than left:
    elif delta_pd_v < -STRABISMUS_PD_THRESH:
        category = "HYPOTROPIA"
        severity = f"Left Eye elevation / Right Eye depression ({abs(delta_pd_v):.1f} PD)"
        color = (0, 200, 200)
    else:
        category = "ORTHOPHORIA (NORMAL)"
        severity = "Binocular alignment within normal physiological limits"
        color = (0, 255, 0)

    return {
        "category": category,
        "severity": severity,
        "color": color,
        "delta_pd_h": delta_pd_h,
        "delta_pd_v": delta_pd_v,
        "delta_deg_h": delta_deg_h,
        "delta_deg_v": delta_deg_v,
        "r_deg_h": r_deg_h,
        "l_deg_h": l_deg_h,
        "r_deg_v": r_deg_v,
        "l_deg_v": l_deg_v,
    }


# ============================================================
# 9-CARDINAL GAZE CLINICAL SCREENING PROTOCOL
# ============================================================
CARDINAL_POSITIONS = [
    ("Primary (Center)",       0.50, 0.50),
    ("Dextroversion (Right)",   0.88, 0.50),
    ("Levoversion (Left)",     0.12, 0.50),
    ("Sursumversion (Up)",      0.50, 0.12),
    ("Deorsumversion (Down)",   0.50, 0.88),
    ("Dextroelevation (Up-R)",  0.88, 0.12),
    ("Levoelevation (Up-L)",    0.12, 0.12),
    ("Dextrodepression (Dn-R)", 0.88, 0.88),
    ("Levodepression (Dn-L)",   0.12, 0.88),
]


def run_cardinal_screening(cap, landmarker, start_time, r_filt_h, r_filt_v, l_filt_h, l_filt_v):
    """Runs the 9 cardinal diagnostic positions of gaze test and saves a full report."""
    results = []
    print("\n" + "=" * 70)
    print("STARTING 9-CARDINAL GAZE STRABISMUS SCREENING PROTOCOL")
    print("Look at each dot steadily with your eyes for 2 seconds. Keep head still.")
    print("=" * 70)

    for idx, (name, fx, fy) in enumerate(CARDINAL_POSITIONS):
        tx = int(fx * SCREEN_W)
        ty = int(fy * SCREEN_H)

        # Transition / Ready phase (1.0 sec)
        t_ready = time.time()
        while time.time() - t_ready < 1.0:
            ret, frame = cap.read()
            if not ret: continue
            canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.circle(canvas, (tx, ty), 32, (0, 255, 255), 3)
            cv2.circle(canvas, (tx, ty), 6, (255, 255, 255), -1)
            cv2.putText(canvas, f"Target {idx+1}/9: {name}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(canvas, "Get ready... fixate on the target", (50, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
            cv2.imshow("Strabismus Screening", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None

        # Capture phase (2.0 sec)
        samples_rh, samples_rv, samples_lh, samples_lv = [], [], [], []
        t_cap = time.time()
        while time.time() - t_cap < 2.0:
            ret, frame = cap.read()
            if not ret: continue
            h_f, w_f, _ = frame.shape
            ts = int((time.time() - start_time) * 1000)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_img, ts)

            canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.circle(canvas, (tx, ty), 32, (0, 200, 255), -1)
            cv2.circle(canvas, (tx, ty), 6, (255, 255, 255), -1)
            cv2.putText(canvas, f"Target {idx+1}/9: {name} [RECORDING]", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

            if res.face_landmarks:
                lms = res.face_landmarks[0]
                rh, rv, r_blk, _, _ = extract_single_eye_metrics(lms, w_f, h_f, RIGHT_OUTER, RIGHT_INNER, RIGHT_TOP, RIGHT_BOTTOM, RIGHT_IRIS)
                lh, lv, l_blk, _, _ = extract_single_eye_metrics(lms, w_f, h_f, LEFT_INNER, LEFT_OUTER, LEFT_TOP, LEFT_BOTTOM, LEFT_IRIS)

                if not r_blk and not l_blk:
                    samples_rh.append(rh)
                    samples_rv.append(rv)
                    samples_lh.append(lh)
                    samples_lv.append(lv)

            cv2.imshow("Strabismus Screening", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None

        if len(samples_rh) > 5:
            m_rh = float(np.mean(samples_rh))
            m_rv = float(np.mean(samples_rv))
            m_lh = float(np.mean(samples_lh))
            m_lv = float(np.mean(samples_lv))
            diag = compute_strabismus_diagnosis(m_rh, m_rv, m_lh, m_lv)
            diag["target_name"] = name
            diag["samples"] = len(samples_rh)
            results.append(diag)
            print(f"  [{idx+1}/9] {name:25s} -> {diag['category']:20s} | Delta H={diag['delta_pd_h']:+5.1f} PD, V={diag['delta_pd_v']:+5.1f} PD")

    # Save comprehensive screening report
    save_cardinal_screening_report(results)
    return results


def save_cardinal_screening_report(results):
    if not results: return
    log_dir = os.path.join(PROJECT_ROOT, "diagnostics", "results")
    os.makedirs(log_dir, exist_ok=True)
    report_file = os.path.join(log_dir, "strabismus_screening_report.txt")

    t_str = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 80,
        f"9-CARDINAL GAZE STRABISMUS CLINICAL SCREENING REPORT — {t_str}",
        "=" * 80,
        f"Screen Resolution  : {SCREEN_W} x {SCREEN_H}",
        f"Positions Evaluated: {len(results)} of 9",
        "",
        "DIAGNOSTIC POSITIONS SUMMARY:",
        "-" * 80,
        f"{'Position':<26} {'Diagnosis':<22} {'Delta H (PD)':<14} {'Delta V (PD)':<14} {'Findings'}",
        "-" * 80,
    ]

    abnormal_count = 0
    for r in results:
        is_abnormal = "ORTHOPHORIA" not in r["category"]
        if is_abnormal: abnormal_count += 1
        lines.append(
            f"{r['target_name']:<26} {r['category']:<22} {r['delta_pd_h']:+6.1f} PD       {r['delta_pd_v']:+6.1f} PD       {r['severity']}"
        )

    lines.extend([
        "-" * 80,
        "",
        "OVERALL CLINICAL CONCLUSION:",
    ])

    if abnormal_count == 0:
        conclusion = "NORMAL (ORTHOPHORIA) — Both eyes show conjugate motility across all 9 cardinal gaze fields."
    elif abnormal_count >= 6:
        conclusion = "CONCOMITANT STRABISMUS — Misalignment is present across majority of gaze fields."
    else:
        conclusion = "INCOMITANT / PARETIC STRABISMUS — Misalignment varies significantly with gaze direction (indicates specific muscle weakness)."

    lines.append(f"  Assessment: {conclusion}")
    lines.append("=" * 80 + "\n")

    with open(report_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[REPORT] Saved full clinical screening report to:\n  {report_file}")


# ============================================================
# MAIN INTERACTIVE PIPELINE
# ============================================================
def main():
    print("=" * 70)
    print("STRABISMUS SCREENING & BILATERAL OCULAR ALIGNMENT MODULE")
    print("Controls:")
    print("  [s] Start 9-Cardinal Gaze Diagnostic Screening Test")
    print("  [c] Tare baseline (zero current position as orthophoric baseline)")
    print("  [q] Quit")
    print("=" * 70)

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Independent adaptive filters for both eyes
    r_filt_h = RockSolidFilter(0.0)
    r_filt_v = RockSolidFilter(0.0)
    l_filt_h = RockSolidFilter(0.0)
    l_filt_v = RockSolidFilter(0.0)

    baseline_dh = 0.0
    baseline_dv = 0.0

    cv2.namedWindow("Strabismus Screening", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Strabismus Screening", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        ts = int((time.time() - start_time) * 1000)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(mp_img, ts)

        # Full-screen canvas
        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

        # Draw main camera feed scaled in center-bottom
        feed_h = int(SCREEN_H * 0.62)
        feed_w = int(feed_h * (w / h))
        feed_x = (SCREEN_W - feed_w) // 2
        feed_y = SCREEN_H - feed_h - 20
        frame_resized = cv2.resize(frame, (feed_w, feed_h))
        canvas[feed_y:feed_y+feed_h, feed_x:feed_x+feed_w] = frame_resized

        if res.face_landmarks:
            lms = res.face_landmarks[0]

            # Right eye
            rh, rv, r_blk, r_box, r_pts = extract_single_eye_metrics(
                lms, w, h, RIGHT_OUTER, RIGHT_INNER, RIGHT_TOP, RIGHT_BOTTOM, RIGHT_IRIS
            )
            # Left eye
            lh, lv, l_blk, l_box, l_pts = extract_single_eye_metrics(
                lms, w, h, LEFT_INNER, LEFT_OUTER, LEFT_TOP, LEFT_BOTTOM, LEFT_IRIS
            )

            # Filtered metrics
            rh_f = r_filt_h.update(rh)
            rv_f = r_filt_v.update(rv)
            lh_f = l_filt_h.update(lh)
            lv_f = l_filt_v.update(lv)

            # Compute clinical diagnosis
            diag = compute_strabismus_diagnosis(rh_f, rv_f, lh_f, lv_f, baseline_dh, baseline_dv)

            # Draw PiP Card for RIGHT EYE (Top-Left)
            if r_box and not r_blk:
                rx1, ry1, rx2, ry2 = r_box
                r_crop = frame[ry1:ry2, rx1:rx2].copy()
                if r_crop.size > 0:
                    r_card = cv2.resize(r_crop, (280, 160))
                    canvas[80:240, 60:340] = r_card
                    cv2.rectangle(canvas, (60, 80), (340, 240), (0, 200, 255), 2)
            cv2.putText(canvas, "RIGHT EYE (OD)", (60, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(canvas, f"H: {diag['r_deg_h']:+4.1f} deg | V: {diag['r_deg_v']:+4.1f} deg",
                        (60, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # Draw PiP Card for LEFT EYE (Top-Right)
            if l_box and not l_blk:
                lx1, ly1, lx2, ly2 = l_box
                l_crop = frame[ly1:ly2, lx1:lx2].copy()
                if l_crop.size > 0:
                    l_card = cv2.resize(l_crop, (280, 160))
                    canvas[80:240, SCREEN_W-340:SCREEN_W-60] = l_card
                    cv2.rectangle(canvas, (SCREEN_W-340, 80), (SCREEN_W-60, 240), (0, 200, 255), 2)
            cv2.putText(canvas, "LEFT EYE (OS)", (SCREEN_W-340, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(canvas, f"H: {diag['l_deg_h']:+4.1f} deg | V: {diag['l_deg_v']:+4.1f} deg",
                        (SCREEN_W-340, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # Center Diagnosis Banner
            badge_color = diag["color"]
            cv2.rectangle(canvas, (SCREEN_W//2 - 320, 45), (SCREEN_W//2 + 320, 140), (25, 25, 30), -1)
            cv2.rectangle(canvas, (SCREEN_W//2 - 320, 45), (SCREEN_W//2 + 320, 140), badge_color, 2)
            cv2.putText(canvas, diag["category"], (SCREEN_W//2 - 300, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, badge_color, 2)
            cv2.putText(canvas, diag["severity"], (SCREEN_W//2 - 300, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

            # Center Disparity Gauges
            disparity_txt = f"Inter-Ocular Disparity: Delta H = {diag['delta_pd_h']:+5.1f} PD ({diag['delta_deg_h']:+4.1f} deg) | Delta V = {diag['delta_pd_v']:+5.1f} PD"
            cv2.putText(canvas, disparity_txt, (SCREEN_W//2 - 380, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            if r_blk or l_blk:
                cv2.putText(canvas, "[BLINK DETECTED]", (feed_x + 20, feed_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        else:
            cv2.putText(canvas, "FACE NOT DETECTED - Please align your face in frame",
                        (SCREEN_W//2 - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Bottom Instructions & Controls HUD
        hud_bar_y = SCREEN_H - 45
        cv2.rectangle(canvas, (0, hud_bar_y - 15), (SCREEN_W, SCREEN_H), (15, 15, 18), -1)
        controls_txt = "[s] Start 9-Cardinal Gaze Diagnostic Screening   |   [c] Tare / Zero Baseline   |   [q] Quit"
        cv2.putText(canvas, controls_txt, (60, hud_bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)

        cv2.imshow("Strabismus Screening", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Tare baseline to current disparity
            if res.face_landmarks:
                baseline_dh = (l_filt_h.smooth - r_filt_h.smooth) * RATIO_TO_DEGREES
                baseline_dv = (l_filt_v.smooth - r_filt_v.smooth) * RATIO_TO_DEGREES
                print(f"[TARE] Baseline zeroed: DH={baseline_dh:+.2f} deg, DV={baseline_dv:+.2f} deg")
        elif key == ord('s'):
            run_cardinal_screening(cap, landmarker, start_time, r_filt_h, r_filt_v, l_filt_h, l_filt_v)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
