"""
cover_test.py - Clinical Automated Cover-Uncover Test for Strabismus & Phoria Detection
========================================================================================
BTP Project: Webcam-Based Eye Tracking & Visual Training for Strabismus Screening

CLINICAL FOUNDATION:
The Cover-Uncover Test is the gold-standard ophthalmic examination to detect and
differentiate:
  1. MANIFEST STRABISMUS (Tropia):
     Present under normal binocular viewing. When the fixating eye is covered,
     the deviating (uncovered) eye MUST make a refixation saccade to take up
     central fixation.
       - Temporal (outward) refixation  -> ESOTROPIA (was turned inward)
       - Nasal (inward) refixation     -> EXOTROPIA (was turned outward)
       - Downward refixation           -> HYPERTROPIA (was elevated)
       - Upward refixation             -> HYPOTROPIA (was depressed)

  2. LATENT STRABISMUS (Phoria):
     Maintained in alignment by binocular fusion. Misalignment only occurs when
     fusion is interrupted (under cover). When the cover is removed, the newly
     uncovered eye makes a fusional recovery movement to regain binocular alignment.
       - Inward recovery  -> EXOPHORIA (drifted out behind cover)
       - Outward recovery -> ESOPHORIA (drifted in behind cover)
       - Downward recovery -> HYPERPHORIA

  3. ORTHOPHORIA:
     Normal physiological alignment. Neither eye moves significantly during
     covering or uncovering (refixation <= 6.0 Delta PD).

MEASUREMENT UNITS:
  - Angular Deviation: Degrees (deg)
  - Ophthalmic Prism Diopters: Delta PD (1 deg ~ 1.75 PD)

CONTROLS & SHORTCUTS:
  - [SPACE] : Manually advance to next test phase
  - [P]     : Pause / Resume playback (video mode)
  - [R]     : Reset & Restart cover test protocol
  - [1]     : Jump to 0s  (Normal Segment)
  - [2]     : Jump to 31s (Esotropia Segment)
  - [3]     : Jump to 51s (Exotropia Segment)
  - [4]     : Jump to 79s (Hypertropia Segment)
  - [A]/[D] : Rewind / Fast-forward 5 seconds
  - [Q]     : Quit application

USAGE:
  Live Webcam Mode:
    python core/cover_test.py
    python core/cover_test.py --webcam 0

  Pre-Recorded Video Mode:
    python core/cover_test.py --video "strabismus-video/vidssave.com Cover Test 480P.mp4"
    python core/cover_test.py --video "strabismus-video/vidssave.com Cover Test 480P.mp4" --manual
"""

import os
import sys
import time
import argparse
from datetime import datetime
from collections import deque

# Prevent matplotlib temp-dir warnings on Windows
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "."), "mpl_cache")

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# PATHS & ASSET RESOLUTION
# ============================================================
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

# ============================================================
# ANATOMICAL LANDMARKS & CALIBRATED CLINICAL CONSTANTS
# ============================================================
# MediaPipe 478-landmark indices
RIGHT_OUTER  = 33   # Temporal corner (Right Eye / OD)
RIGHT_INNER  = 133  # Nasal corner
RIGHT_TOP    = 159  # Upper eyelid
RIGHT_BOTTOM = 145  # Lower eyelid
RIGHT_IRIS   = 468  # Iris / pupil center

LEFT_INNER   = 362  # Nasal corner (Left Eye / OS)
LEFT_OUTER   = 263  # Temporal corner
LEFT_TOP     = 386  # Upper eyelid
LEFT_BOTTOM  = 374  # Lower eyelid
LEFT_IRIS    = 473  # Iris / pupil center

# Ophthalmic conversion factors
PD_PER_DEGREE = 1.75
RATIO_TO_DEGREES = 134.0

# Calibrated clinical thresholds:
# Clinically, physiological fixation jitter/flutter is up to 2.5 deg (4.4 PD).
# True clinical strabismus requires saccadic displacement > 3.5 deg (~6.1 PD).
SACCADE_THRESH_DEG = 3.5     # > 3.5 deg (~6.1 Delta PD) indicates manifest refixation saccade
PHORIA_THRESH_DEG  = 3.5     # > 3.5 deg (~6.1 Delta PD) recovery indicates latent phoria
NORMAL_PD_LIMIT    = 7.0     # <= 7.0 PD is normal physiological alignment

# ============================================================
# SIGNAL FILTER
# ============================================================
class RockSolidFilter:
    """Two-stage median + adaptive EMA filter preserving saccades while damping jitter."""
    def __init__(self, init_val=0.0, window_size=5, alpha_still=0.12, alpha_move=0.55, deadband=0.0015, saccade_thresh=0.010):
        self.buf = deque([float(init_val)] * window_size, maxlen=window_size)
        self.smooth = float(init_val)
        self.alpha_still = float(alpha_still)
        self.alpha_move = float(alpha_move)
        self.deadband = float(deadband)
        self.saccade_thresh = float(saccade_thresh)
        self.consecutive_moves = 0
        self.initialized = False

    def reset(self, init_val=0.0):
        self.buf.clear()
        self.smooth = float(init_val)
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
# EYE METRICS & OCCLUSION EXTRACTION
# ============================================================
def extract_single_eye_metrics(lms, w, h, idx_outer, idx_inner, idx_top, idx_bottom, idx_iris):
    """
    Extracts scale- and rotation-invariant physical iris position metrics.
    Returns:
      horiz_ratio: displacement along canthal vector pointing NASALLY (+ = inward, - = outward)
      vert_ratio: displacement along perpendicular vector pointing DOWNWARD (+ = downward, - = upward)
      is_blink: True if palpebral fissure closed
      box: (x1, y1, x2, y2) bounding box
      landmarks: (p_outer, p_inner, p_iris, p_mid) in pixel coordinates
    """
    p_outer  = np.array([lms[idx_outer].x * w,  lms[idx_outer].y * h],  dtype=np.float64)
    p_inner  = np.array([lms[idx_inner].x * w,  lms[idx_inner].y * h],  dtype=np.float64)
    p_top    = np.array([lms[idx_top].x * w,    lms[idx_top].y * h],    dtype=np.float64)
    p_bottom = np.array([lms[idx_bottom].x * w, lms[idx_bottom].y * h], dtype=np.float64)
    p_iris   = np.array([lms[idx_iris].x * w,   lms[idx_iris].y * h],   dtype=np.float64)

    canthi_vec = p_inner - p_outer
    eye_width = np.linalg.norm(canthi_vec)
    palpebral_height = np.linalg.norm(p_top - p_bottom)

    if eye_width < 1e-4:
        return 0.0, 0.0, True, (0, 0, 0, 0), (p_outer, p_inner, p_iris, p_outer)

    # Blink / eye closure check
    aperture_ratio = palpebral_height / eye_width
    is_blink = aperture_ratio < 0.12

    # Unit vector pointing NASALLY
    u_nasal = canthi_vec / eye_width

    # Unit vector pointing DOWNWARD (+Y in image coordinate frame)
    if idx_outer == RIGHT_OUTER:
        normal_down = np.array([-u_nasal[1], u_nasal[0]], dtype=np.float64)
    else:
        normal_down = np.array([u_nasal[1], -u_nasal[0]], dtype=np.float64)

    p_mid = 0.5 * (p_outer + p_inner)
    d_iris = p_iris - p_mid

    horiz_offset = np.dot(d_iris, u_nasal)
    vert_offset  = np.dot(d_iris, normal_down)

    horiz_ratio = float(horiz_offset / eye_width)
    vert_ratio  = float(vert_offset / eye_width)

    # Eye bounding box with margin
    margin_x = int(eye_width * 0.40)
    margin_y = int(eye_width * 0.35)
    cx, cy = int(p_mid[0]), int(p_mid[1])
    box = (
        max(0, cx - margin_x),
        max(0, cy - margin_y),
        min(w - 1, cx + margin_x),
        min(h - 1, cy + margin_y)
    )

    return horiz_ratio, vert_ratio, is_blink, box, (p_outer, p_inner, p_iris, p_mid)


def detect_eye_occlusion(frame, box, is_blink):
    """
    Detects if an eye is covered by an occluder, hand, or paddle.
    Combines blink status and crop pixel standard deviation (occluders have low variance).
    """
    if is_blink:
        return True
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return True
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return True
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    std_val = float(np.std(gray))
    return std_val < 14.0

# ============================================================
# COVER TEST CLINICAL PROTOCOL PHASES
# ============================================================
class CoverTestPhase:
    IDLE        = 0  # Waiting for test start
    BASELINE    = 1  # Both eyes open: fixate central target
    COVER_OD    = 2  # Right Eye covered -> Track Left Eye (OS) refixation
    UNCOVER_OD  = 3  # Right Eye uncovered -> Measure OD recovery (Phoria)
    COVER_OS    = 4  # Left Eye covered -> Track Right Eye (OD) refixation
    UNCOVER_OS  = 5  # Left Eye uncovered -> Measure OS recovery (Phoria)
    COMPLETE    = 6  # Test finished, reports generated

PHASE_CONFIG = {
    CoverTestPhase.IDLE:       ("STANDBY", "Press [SPACE] to start Cover Test | [1-4] Jump to video segments", 0.0, (180, 180, 180)),
    CoverTestPhase.BASELINE:   ("BASELINE FIXATION", "Look directly at the center target with BOTH eyes", 3.0, (255, 180, 0)),
    CoverTestPhase.COVER_OD:   ("COVER RIGHT EYE (OD)", "Keep looking at target with LEFT eye! Tracking refixation...", 3.0, (0, 140, 255)),
    CoverTestPhase.UNCOVER_OD: ("UNCOVER RIGHT EYE", "Keep looking at target with both eyes. Measuring fusion recovery...", 2.5, (0, 230, 255)),
    CoverTestPhase.COVER_OS:   ("COVER LEFT EYE (OS)", "Keep looking at target with RIGHT eye! Tracking refixation...", 3.0, (0, 140, 255)),
    CoverTestPhase.UNCOVER_OS: ("UNCOVER LEFT EYE", "Keep looking at target with both eyes. Measuring fusion recovery...", 2.5, (0, 230, 255)),
    CoverTestPhase.COMPLETE:   ("TEST COMPLETE", "Clinical analysis finished. Report saved. Press [R] to re-test.", 0.0, (0, 255, 0)),
}

# ============================================================
# MAIN COVER TEST ENGINE
# ============================================================
class CoverTestEngine:
    def __init__(self, mode_auto=True, phase_duration=3.0):
        self.mode_auto = mode_auto
        self.phase_duration = phase_duration
        self.phase = CoverTestPhase.IDLE
        self.phase_start_time = 0.0

        # Filters for left and right eye signals
        self.r_filt_h = RockSolidFilter()
        self.r_filt_v = RockSolidFilter()
        self.l_filt_h = RockSolidFilter()
        self.l_filt_v = RockSolidFilter()

        # Phase data logs
        self.log_time = []
        self.log_phase = []
        self.log_rh, self.log_rv = [], []
        self.log_lh, self.log_lv = [], []
        self.log_r_deg_h, self.log_l_deg_h = [], []
        self.log_r_deg_v, self.log_l_deg_v = [], []

        # Baseline measurements
        self.baseline_rh = 0.0
        self.baseline_rv = 0.0
        self.baseline_lh = 0.0
        self.baseline_lv = 0.0

        # Saccade / refixation findings
        self.os_refix_h = 0.0  # Left eye shift when OD covered (deg)
        self.os_refix_v = 0.0
        self.od_refix_h = 0.0  # Right eye shift when OS covered (deg)
        self.od_refix_v = 0.0

        # Fusion recovery findings (Phoria)
        self.od_recovery_h = 0.0 # Right eye recovery upon uncover (deg)
        self.os_recovery_h = 0.0 # Left eye recovery upon uncover (deg)

        self.diagnosis = {}
        self.report_generated = False

    def reset(self):
        """Resets engine for a fresh run."""
        self.phase = CoverTestPhase.IDLE
        self.phase_start_time = 0.0
        self.r_filt_h.reset()
        self.r_filt_v.reset()
        self.l_filt_h.reset()
        self.l_filt_v.reset()
        self.log_time.clear()
        self.log_phase.clear()
        self.log_rh.clear(); self.log_rv.clear()
        self.log_lh.clear(); self.log_lv.clear()
        self.log_r_deg_h.clear(); self.log_l_deg_h.clear()
        self.log_r_deg_v.clear(); self.log_l_deg_v.clear()
        self.baseline_rh = 0.0; self.baseline_rv = 0.0
        self.baseline_lh = 0.0; self.baseline_lv = 0.0
        self.os_refix_h = 0.0; self.os_refix_v = 0.0
        self.od_refix_h = 0.0; self.od_refix_v = 0.0
        self.od_recovery_h = 0.0; self.os_recovery_h = 0.0
        self.diagnosis.clear()
        self.report_generated = False
        print("[INFO] Cover Test Engine reset to IDLE.")

    def start(self, current_time=None):
        """Transitions from IDLE to BASELINE."""
        self.phase = CoverTestPhase.BASELINE
        self.phase_start_time = time.time() if current_time is None else current_time
        print("\n" + "=" * 65)
        print(">>> STARTING AUTOMATED COVER-UNCOVER CLINICAL PROTOCOL <<<")
        print("=" * 65)

    def next_phase(self, current_time=None):
        """Advances to the next protocol phase."""
        t_ref = time.time() if current_time is None else current_time
        if self.phase == CoverTestPhase.IDLE:
            self.start(t_ref)
            return

        self._on_phase_finish(self.phase)

        if self.phase == CoverTestPhase.BASELINE:
            self.phase = CoverTestPhase.COVER_OD
        elif self.phase == CoverTestPhase.COVER_OD:
            self.phase = CoverTestPhase.UNCOVER_OD
        elif self.phase == CoverTestPhase.UNCOVER_OD:
            self.phase = CoverTestPhase.COVER_OS
        elif self.phase == CoverTestPhase.COVER_OS:
            self.phase = CoverTestPhase.UNCOVER_OS
        elif self.phase == CoverTestPhase.UNCOVER_OS:
            self.phase = CoverTestPhase.COMPLETE
            self.analyze_results()

        self.phase_start_time = t_ref

    def update_frame(self, t_now, rh_f, rv_f, lh_f, lv_f, r_occluded, l_occluded):
        """Records data and manages phase transitions."""
        r_deg_h = rh_f * RATIO_TO_DEGREES
        l_deg_h = lh_f * RATIO_TO_DEGREES
        r_deg_v = rv_f * RATIO_TO_DEGREES
        l_deg_v = lv_f * RATIO_TO_DEGREES

        if self.phase != CoverTestPhase.IDLE and self.phase != CoverTestPhase.COMPLETE:
            self.log_time.append(t_now)
            self.log_phase.append(self.phase)
            self.log_rh.append(rh_f)
            self.log_rv.append(rv_f)
            self.log_lh.append(lh_f)
            self.log_lv.append(lv_f)
            self.log_r_deg_h.append(r_deg_h)
            self.log_l_deg_h.append(l_deg_h)
            self.log_r_deg_v.append(r_deg_v)
            self.log_l_deg_v.append(l_deg_v)

        # Timed auto-transition logic
        if self.mode_auto and self.phase not in [CoverTestPhase.IDLE, CoverTestPhase.COMPLETE]:
            duration = self.phase_duration if self.phase != CoverTestPhase.UNCOVER_OD and self.phase != CoverTestPhase.UNCOVER_OS else (self.phase_duration * 0.8)
            elapsed = t_now - self.phase_start_time
            if elapsed >= duration:
                self.next_phase(t_now)

    def _on_phase_finish(self, finished_phase):
        """Extracts key measurements at phase completion."""
        phase_mask = [p == finished_phase for p in self.log_phase]
        if not any(phase_mask):
            return

        idx_phase = [i for i, m in enumerate(phase_mask) if m][-12:]

        if finished_phase == CoverTestPhase.BASELINE:
            self.baseline_rh = float(np.median([self.log_rh[i] for i in idx_phase]))
            self.baseline_rv = float(np.median([self.log_rv[i] for i in idx_phase]))
            self.baseline_lh = float(np.median([self.log_lh[i] for i in idx_phase]))
            self.baseline_lv = float(np.median([self.log_lv[i] for i in idx_phase]))

        elif finished_phase == CoverTestPhase.COVER_OD:
            # Right eye covered -> measure Left Eye refixation saccade
            all_idx = [i for i, m in enumerate(phase_mask) if m]
            if all_idx:
                diffs_h = [(self.log_lh[i] - self.baseline_lh) * RATIO_TO_DEGREES for i in all_idx]
                diffs_v = [(self.log_lv[i] - self.baseline_lv) * RATIO_TO_DEGREES for i in all_idx]
                self.os_refix_h = float(max(diffs_h, key=abs))
                self.os_refix_v = float(max(diffs_v, key=abs))

        elif finished_phase == CoverTestPhase.UNCOVER_OD:
            all_od_uncover_idx = [i for i, m in enumerate(phase_mask) if m]
            if len(all_od_uncover_idx) >= 6:
                early_rh = float(np.median([self.log_rh[i] for i in all_od_uncover_idx[:4]]))
                late_rh  = float(np.median([self.log_rh[i] for i in all_od_uncover_idx[-6:]]))
                self.od_recovery_h = (late_rh - early_rh) * RATIO_TO_DEGREES

        elif finished_phase == CoverTestPhase.COVER_OS:
            # Left eye covered -> measure Right Eye refixation saccade
            all_idx = [i for i, m in enumerate(phase_mask) if m]
            if all_idx:
                diffs_h = [(self.log_rh[i] - self.baseline_rh) * RATIO_TO_DEGREES for i in all_idx]
                diffs_v = [(self.log_rv[i] - self.baseline_rv) * RATIO_TO_DEGREES for i in all_idx]
                self.od_refix_h = float(max(diffs_h, key=abs))
                self.od_refix_v = float(max(diffs_v, key=abs))

        elif finished_phase == CoverTestPhase.UNCOVER_OS:
            all_os_uncover_idx = [i for i, m in enumerate(phase_mask) if m]
            if len(all_os_uncover_idx) >= 6:
                early_lh = float(np.median([self.log_lh[i] for i in all_os_uncover_idx[:4]]))
                late_lh  = float(np.median([self.log_lh[i] for i in all_os_uncover_idx[-6:]]))
                self.os_recovery_h = (late_lh - early_lh) * RATIO_TO_DEGREES

    def analyze_results(self):
        """Clinical decision tree evaluating Tropia vs Phoria vs Orthophoria."""
        if self.report_generated:
            return

        # Prism diopter conversions
        os_refix_pd_h = self.os_refix_h * PD_PER_DEGREE
        os_refix_pd_v = self.os_refix_v * PD_PER_DEGREE
        od_refix_pd_h = self.od_refix_h * PD_PER_DEGREE
        od_refix_pd_v = self.od_refix_v * PD_PER_DEGREE

        od_recov_pd_h = self.od_recovery_h * PD_PER_DEGREE
        os_recov_pd_h = self.os_recovery_h * PD_PER_DEGREE

        manifest_findings = []
        h_tropia = None
        v_tropia = None
        tropia_detail = "Binocular alignment within normal physiological limits (<= 6.0 Delta PD)."

        # Check Left Eye Manifest Deviation (Horizontal)
        if self.os_refix_h < -SACCADE_THRESH_DEG:
            manifest_findings.append(f"Left Eye Esotropia ({abs(os_refix_pd_h):.1f} Delta PD, outward refixation)")
            h_tropia = "LEFT ESOTROPIA"
        elif self.os_refix_h > SACCADE_THRESH_DEG:
            manifest_findings.append(f"Left Eye Exotropia ({abs(os_refix_pd_h):.1f} Delta PD, inward refixation)")
            h_tropia = "LEFT EXOTROPIA"

        # Check Right Eye Manifest Deviation (Horizontal)
        if self.od_refix_h < -SACCADE_THRESH_DEG:
            manifest_findings.append(f"Right Eye Esotropia ({abs(od_refix_pd_h):.1f} Delta PD, outward refixation)")
            h_tropia = "RIGHT ESOTROPIA" if h_tropia is None else "BILATERAL ESOTROPIA"
        elif self.od_refix_h > SACCADE_THRESH_DEG:
            manifest_findings.append(f"Right Eye Exotropia ({abs(od_refix_pd_h):.1f} Delta PD, inward refixation)")
            h_tropia = "RIGHT EXOTROPIA" if h_tropia is None else "BILATERAL EXOTROPIA"

        # Vertical Manifest Check
        if abs(self.os_refix_v * PD_PER_DEGREE) > NORMAL_PD_LIMIT:
            v_dir = "Hypertropia" if self.os_refix_v < 0 else "Hypotropia"
            manifest_findings.append(f"Left Eye {v_dir} ({abs(os_refix_pd_v):.1f} Delta PD)")
            v_tropia = f"LEFT {v_dir.upper()}"

        if abs(self.od_refix_v * PD_PER_DEGREE) > NORMAL_PD_LIMIT:
            v_dir = "Hypertropia" if self.od_refix_v < 0 else "Hypotropia"
            manifest_findings.append(f"Right Eye {v_dir} ({abs(od_refix_pd_v):.1f} Delta PD)")
            v_tropia = f"RIGHT {v_dir.upper()}"

        # Combine diagnoses intelligently:
        if h_tropia and v_tropia:
            tropia_type = f"{h_tropia} WITH {v_tropia}"
        elif h_tropia:
            tropia_type = h_tropia
        elif v_tropia:
            tropia_type = v_tropia
        else:
            tropia_type = "ORTHOPHORIA (NORMAL)"

        # Determine Latent Phoria (Uncover Phase)
        latent_findings = []
        if tropia_type == "ORTHOPHORIA (NORMAL)":
            if self.od_recovery_h > PHORIA_THRESH_DEG or self.os_recovery_h > PHORIA_THRESH_DEG:
                phoria_pd = max(abs(od_recov_pd_h), abs(os_recov_pd_h))
                latent_findings.append(f"Exophoria detected ({phoria_pd:.1f} Delta PD inward fusional recovery)")
                tropia_type = "EXOPHORIA (LATENT)"
            elif self.od_recovery_h < -PHORIA_THRESH_DEG or self.os_recovery_h < -PHORIA_THRESH_DEG:
                phoria_pd = max(abs(od_recov_pd_h), abs(os_recov_pd_h))
                latent_findings.append(f"Esophoria detected ({phoria_pd:.1f} Delta PD outward fusional recovery)")
                tropia_type = "ESOPHORIA (LATENT)"

        if manifest_findings:
            tropia_detail = " | ".join(manifest_findings)
        elif latent_findings:
            tropia_detail = " | ".join(latent_findings)

        self.diagnosis = {
            "category": tropia_type,
            "detail": tropia_detail,
            "os_refix_deg_h": self.os_refix_h,
            "os_refix_pd_h": os_refix_pd_h,
            "od_refix_deg_h": self.od_refix_h,
            "od_refix_pd_h": od_refix_pd_h,
            "od_recov_pd_h": od_recov_pd_h,
            "os_recov_pd_h": os_recov_pd_h,
        }

        print("\n" + "=" * 65)
        print("CLINICAL COVER-UNCOVER TEST DIAGNOSTIC RESULT:")
        print(f"DIAGNOSIS: {tropia_type}")
        print(f"FINDINGS : {tropia_detail}")
        print(f"OS Refixation Saccade (Left eye):  {self.os_refix_h:+.2f} deg ({os_refix_pd_h:+.1f} Delta PD)")
        print(f"OD Refixation Saccade (Right eye): {self.od_refix_h:+.2f} deg ({od_refix_pd_h:+.1f} Delta PD)")
        print("=" * 65)

        self.save_clinical_report()
        self.save_refixation_plot()
        self.report_generated = True

    def save_clinical_report(self):
        """Saves a formal text report in diagnostics/results/."""
        out_dir = os.path.join(PROJECT_ROOT, "diagnostics", "results")
        os.makedirs(out_dir, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(out_dir, f"cover_test_report_{ts_str}.txt")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("================================================================================\n")
            f.write("CLINICAL COVER-UNCOVER TEST REPORT\n")
            f.write("Automated Strabismus & Latent Phoria Screening System\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("================================================================================\n\n")
            f.write(f"FINAL CLINICAL DIAGNOSIS: {self.diagnosis.get('category', 'UNKNOWN')}\n")
            f.write(f"Diagnostic Breakdown    : {self.diagnosis.get('detail', 'N/A')}\n\n")
            f.write("--- MEASUREMENT PARAMETERS ---\n")
            f.write(f"Baseline OS (Left)  Nasal Ratio: {self.baseline_lh:.4f} | Vert: {self.baseline_lv:.4f}\n")
            f.write(f"Baseline OD (Right) Nasal Ratio: {self.baseline_rh:.4f} | Vert: {self.baseline_rv:.4f}\n\n")
            f.write(f"OS Refixation Saccade (OD Covered) : {self.diagnosis.get('os_refix_deg_h', 0):+.2f} deg | {self.diagnosis.get('os_refix_pd_h', 0):+.1f} Delta PD\n")
            f.write(f"OD Refixation Saccade (OS Covered) : {self.diagnosis.get('od_refix_deg_h', 0):+.2f} deg | {self.diagnosis.get('od_refix_pd_h', 0):+.1f} Delta PD\n")
            f.write(f"OD Fusional Recovery (OD Uncovered): {self.diagnosis.get('od_recov_pd_h', 0):+.1f} Delta PD\n")
            f.write(f"OS Fusional Recovery (OS Uncovered): {self.diagnosis.get('os_recov_pd_h', 0):+.1f} Delta PD\n\n")
            f.write("--- CLINICAL INTERPRETATION GUIDE ---\n")
            f.write("- Refixation saccade in uncovered eye during covering -> Manifest Tropia\n")
            f.write("- Outward temporal refixation indicates initial inward turn (Esotropia)\n")
            f.write("- Inward nasal refixation indicates initial outward turn (Exotropia)\n")
            f.write("- Refixation upon uncover with no manifest movement -> Latent Phoria\n")
            f.write("================================================================================\n")
        print(f"[OK] Saved text report: {filepath}")

    def save_refixation_plot(self):
        """Generates a publication-quality time-series curve showing refixation saccades."""
        if len(self.log_time) < 10:
            return
        out_dir = os.path.join(PROJECT_ROOT, "diagnostics", "results")
        os.makedirs(out_dir, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(out_dir, f"cover_test_curves_{ts_str}.png")

        t0 = self.log_time[0]
        t_rel = [t - t0 for t in self.log_time]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Cover-Uncover Test Refixation Kinematics\nDiagnosis: {self.diagnosis.get('category', 'N/A')}",
                     fontsize=14, fontweight="bold")

        # Plot Left Eye (OS)
        ax1.plot(t_rel, self.log_l_deg_h, color="#1f77b4", lw=2, label="Left Eye (OS) Horizontal (°)")
        ax1.axhline(self.baseline_lh * RATIO_TO_DEGREES, color="#1f77b4", ls="--", alpha=0.6, label="OS Baseline")
        ax1.set_ylabel("OS Gaze (°)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")

        # Plot Right Eye (OD)
        ax2.plot(t_rel, self.log_r_deg_h, color="#ff7f0e", lw=2, label="Right Eye (OD) Horizontal (°)")
        ax2.axhline(self.baseline_rh * RATIO_TO_DEGREES, color="#ff7f0e", ls="--", alpha=0.6, label="OD Baseline")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("OD Gaze (°)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")

        # Highlight protocol phases with background bands
        phase_colors = {
            CoverTestPhase.BASELINE:   ("#e0f3f8", "Baseline"),
            CoverTestPhase.COVER_OD:   ("#fee090", "Cover OD"),
            CoverTestPhase.UNCOVER_OD: ("#e0e0e0", "Uncover OD"),
            CoverTestPhase.COVER_OS:   ("#fdae61", "Cover OS"),
            CoverTestPhase.UNCOVER_OS: ("#e0e0e0", "Uncover OS"),
        }

        cur_p = None
        t_start_p = 0
        for i, p in enumerate(self.log_phase):
            if p != cur_p:
                if cur_p in phase_colors:
                    col, _ = phase_colors[cur_p]
                    for ax in (ax1, ax2):
                        ax.axvspan(t_start_p, t_rel[i], color=col, alpha=0.45)
                cur_p = p
                t_start_p = t_rel[i]
        if cur_p in phase_colors:
            col, _ = phase_colors[cur_p]
            for ax in (ax1, ax2):
                ax.axvspan(t_start_p, t_rel[-1], color=col, alpha=0.45)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[OK] Saved refixation plot: {plot_path}")


# ============================================================
# RENDERING HUD & FIXATION TARGET
# ============================================================
def draw_cover_test_hud(canvas, engine, r_occluded, l_occluded, r_pip, l_pip, frame_count, vid_time_str=None, is_paused=False):
    """Draws clinical fixation target, PiP eye cards, phase status, shortcuts, and instructions."""
    h_can, w_can, _ = canvas.shape
    cx, cy = w_can // 2, h_can // 2

    # 1. Pulsating High-Contrast Fixation Target in Center
    pulse = int(5 * np.sin(frame_count * 0.15))
    cv2.circle(canvas, (cx, cy), 28 + pulse, (0, 0, 255), 2)
    cv2.circle(canvas, (cx, cy), 16, (255, 255, 255), -1)
    cv2.circle(canvas, (cx, cy), 6, (0, 0, 255), -1)
    cv2.line(canvas, (cx - 38, cy), (cx + 38, cy), (0, 255, 255), 1)
    cv2.line(canvas, (cx, cy - 38), (cx, cy + 38), (0, 255, 255), 1)

    # 2. Phase Banner Header
    phase_title, phase_instruction, phase_dur, phase_color = PHASE_CONFIG[engine.phase]
    cv2.rectangle(canvas, (0, 0), (w_can, 72), (24, 24, 24), -1)
    cv2.rectangle(canvas, (0, 68), (w_can, 72), phase_color, -1)

    # Elapsed countdown bar
    if engine.phase not in [CoverTestPhase.IDLE, CoverTestPhase.COMPLETE] and phase_dur > 0:
        elapsed = time.time() - engine.phase_start_time
        pct = min(1.0, elapsed / phase_dur)
        cv2.rectangle(canvas, (0, 68), (int(w_can * pct), 72), (0, 255, 0), -1)

    cv2.putText(canvas, f"PHASE: {phase_title}", (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, phase_color, 2)
    cv2.putText(canvas, phase_instruction, (25, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1)

    # Video timestamp & pause indicator (top-right)
    if vid_time_str is not None:
        status_txt = f"{vid_time_str} {'[PAUSED]' if is_paused else '[PLAY]'}"
        s_col = (0, 165, 255) if is_paused else (0, 255, 0)
        cv2.putText(canvas, status_txt, (w_can - 240, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, s_col, 2)

    # 3. Dual-Eye PiP Cards
    # Right Eye (OD) on top-left
    if r_pip is not None:
        rh_pip, rw_pip, _ = r_pip.shape
        x_od, y_od = 25, 85
        cv2.rectangle(canvas, (x_od - 2, y_od - 2), (x_od + rw_pip + 2, y_od + rh_pip + 2), (60, 60, 60), 1)
        canvas[y_od:y_od + rh_pip, x_od:x_od + rw_pip] = r_pip
        badge = "OD (RIGHT EYE): COVERED" if r_occluded else "OD (RIGHT EYE): OPEN"
        b_col = (0, 0, 255) if r_occluded else (0, 255, 0)
        cv2.putText(canvas, badge, (x_od + 5, y_od - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, b_col, 1)

    # Left Eye (OS) on top-right
    if l_pip is not None:
        lh_pip, lw_pip, _ = l_pip.shape
        x_os, y_os = w_can - lw_pip - 25, 85
        cv2.rectangle(canvas, (x_os - 2, y_os - 2), (x_os + lw_pip + 2, y_os + lh_pip + 2), (60, 60, 60), 1)
        canvas[y_os:y_os + lh_pip, x_os:x_os + lw_pip] = l_pip
        badge = "OS (LEFT EYE): COVERED" if l_occluded else "OS (LEFT EYE): OPEN"
        b_col = (0, 0, 255) if l_occluded else (0, 255, 0)
        cv2.putText(canvas, badge, (x_os + 5, y_os - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, b_col, 1)

    # 4. Live Metric Gauge & Diagnostic Banner (Bottom)
    gauge_y = h_can - 70
    cv2.rectangle(canvas, (0, gauge_y - 10), (w_can, h_can), (18, 18, 18), -1)

    if engine.phase == CoverTestPhase.COMPLETE:
        diag = engine.diagnosis.get("category", "COMPLETE")
        det = engine.diagnosis.get("detail", "")
        cv2.putText(canvas, f"DIAGNOSIS: {diag}", (25, gauge_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 0), 2)
        cv2.putText(canvas, det, (25, gauge_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    else:
        # Show real-time relative displacement from baseline
        os_dev = (engine.l_filt_h.smooth - engine.baseline_lh) * RATIO_TO_DEGREES * PD_PER_DEGREE
        od_dev = (engine.r_filt_h.smooth - engine.baseline_rh) * RATIO_TO_DEGREES * PD_PER_DEGREE
        txt_m = f"OS (Left) Shift: {os_dev:+.1f} Delta PD   |   OD (Right) Shift: {od_dev:+.1f} Delta PD"
        cv2.putText(canvas, txt_m, (25, gauge_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)

    # 5. Bottom Shortcut Navigation Hint Bar
    hint_txt = "[SPACE]=Next Phase | [P]=Pause | [R]=Reset | [1]=Norm(0s) [2]=Eso(31s) [3]=Exo(51s) [4]=Hyp(79s)"
    cv2.putText(canvas, hint_txt, (25, h_can - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 140, 140), 1)


# ============================================================
# MAIN APPLICATION LOOP
# ============================================================
def run_cover_test(video_source=None, auto_start=True, phase_duration=3.0, save_video=False):
    """
    Runs the Cover Test either on live webcam or on a pre-recorded video file.
    Supports pause, fast rewind/forward, and 1-key jumping between clinical segments.
    """
    task_path = find_asset("face_landmarker.task")
    if not os.path.exists(task_path):
        print(f"Error: Could not locate face_landmarker.task at {task_path}")
        return

    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    if video_source is None:
        cap = cv2.VideoCapture(0)
        is_live = True
        total_frames = 0
        total_dur_s = 0.0
        print("[INFO] Opened Live Webcam stream.")
    else:
        cap = cv2.VideoCapture(video_source)
        is_live = False
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_dur_s = total_frames / fps_v if total_frames > 0 else 0.0
        print(f"[INFO] Opened Video File: {video_source} ({total_dur_s:.1f}s, {int(total_frames)} frames)")

    if not cap.isOpened():
        print(f"Error: Unable to open video source: {video_source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    DISP_W, DISP_H = 1024, 640

    out_writer = None
    if save_video:
        out_dir = os.path.join(PROJECT_ROOT, "diagnostics", "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "cover_test_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (DISP_W, DISP_H))
        print(f"[INFO] Recording output to: {out_path}")

    if not is_live and total_dur_s > 8.0:
        phase_duration = max(2.5, total_dur_s / 5.2)
        print(f"[INFO] Auto-adapted phase duration to {phase_duration:.1f}s for video length ({total_dur_s:.1f}s).")

    engine = CoverTestEngine(mode_auto=auto_start, phase_duration=phase_duration)
    if auto_start:
        engine.start()

    start_wall_time = time.time()
    frame_idx = 0
    is_paused = False
    mp_ts_ms = 0
    frame_step_ms = max(1, int(1000.0 / fps))

    print("\n" + "-" * 70)
    print("Cover Test Active! Shortcuts:")
    print("  [SPACE]  : Advance phase manually")
    print("  [P]      : Pause / Resume playback")
    print("  [R]      : Reset & Restart Cover Test from Baseline")
    print("  [1]      : Jump to 00s (Normal Eyes)")
    print("  [2]      : Jump to 31s (Esotropia - Inward)")
    print("  [3]      : Jump to 51s (Exotropia - Outward)")
    print("  [4]      : Jump to 79s (Hypertropia - Vertical)")
    print("  [A] / [D]: Rewind / Forward 5s")
    print("  [Q]      : Quit")
    print("-" * 70 + "\n")

    current_frame = None

    while cap.isOpened():
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video stream completed.")
                break
            frame_idx += 1
            current_frame = frame
            # Monotonically increasing timestamp for MediaPipe (guaranteed strictly increasing)
            mp_ts_ms += frame_step_ms
        else:
            frame = current_frame
            if frame is None:
                break

        h_f, w_f, _ = frame.shape

        if is_live:
            t_now = time.time()
            vid_time_str = datetime.now().strftime("%H:%M:%S")
        else:
            cur_sec = frame_idx / fps
            t_now = start_wall_time + (mp_ts_ms / 1000.0)
            vid_time_str = f"{int(cur_sec//60):02d}:{cur_sec%60:04.1f} / {int(total_dur_s//60):02d}:{total_dur_s%60:04.1f}"

        r_pip, l_pip = None, None
        r_occluded, l_occluded = False, False

        rh_f, rv_f = engine.r_filt_h.smooth, engine.r_filt_v.smooth
        lh_f, lv_f = engine.l_filt_h.smooth, engine.l_filt_v.smooth

        # Only process MediaPipe when not paused to prevent timestamp stalls
        if not is_paused:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            res = landmarker.detect_for_video(mp_img, mp_ts_ms)

            if res.face_landmarks:
                lms = res.face_landmarks[0]

                # Right Eye (OD)
                rh_raw, rv_raw, r_blink, r_box, _ = extract_single_eye_metrics(
                    lms, w_f, h_f, RIGHT_OUTER, RIGHT_INNER, RIGHT_TOP, RIGHT_BOTTOM, RIGHT_IRIS
                )
                r_occluded = detect_eye_occlusion(frame, r_box, r_blink)
                if not r_blink and not r_occluded:
                    rh_f = engine.r_filt_h.update(rh_raw)
                    rv_f = engine.r_filt_v.update(rv_raw)

                # Left Eye (OS)
                lh_raw, lv_raw, l_blink, l_box, _ = extract_single_eye_metrics(
                    lms, w_f, h_f, LEFT_INNER, LEFT_OUTER, LEFT_TOP, LEFT_BOTTOM, LEFT_IRIS
                )
                l_occluded = detect_eye_occlusion(frame, l_box, l_blink)
                if not l_blink and not l_occluded:
                    lh_f = engine.l_filt_h.update(lh_raw)
                    lv_f = engine.l_filt_v.update(lv_raw)

                # Extract PiP Visuals
                x1, y1, x2, y2 = r_box
                if x2 > x1 and y2 > y1:
                    r_crop = frame[y1:y2, x1:x2].copy()
                    r_pip = cv2.resize(r_crop, (150, 95))

                x1, y1, x2, y2 = l_box
                if x2 > x1 and y2 > y1:
                    l_crop = frame[y1:y2, x1:x2].copy()
                    l_pip = cv2.resize(l_crop, (150, 95))

            # Update test engine state
            engine.update_frame(t_now, rh_f, rv_f, lh_f, lv_f, r_occluded, l_occluded)

        # Composite HUD Canvas
        canvas = np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)
        small_bg = cv2.resize(frame, (DISP_W, DISP_H))
        canvas = cv2.addWeighted(small_bg, 0.40, canvas, 0.60, 0)

        draw_cover_test_hud(canvas, engine, r_occluded, l_occluded, r_pip, l_pip, frame_idx, vid_time_str, is_paused)

        if out_writer is not None:
            out_writer.write(canvas)

        cv2.imshow("Clinical Cover-Uncover Test", canvas)

        key = cv2.waitKey(int(1000 / fps) if (not is_live and not is_paused) else 1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # Spacebar advances phase
            engine.next_phase()
        elif key == ord('p') or key == ord('P'):  # Pause / resume
            is_paused = not is_paused
            print(f"[INFO] {'PAUSED' if is_paused else 'RESUMED'}")
        elif key == ord('r') or key == ord('R'):  # Reset test
            engine.reset()
            engine.start()
        # Direct video jumping keys:
        elif not is_live and key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            target_sec = 0.0
            if key == ord('1'): target_sec = 0.0
            elif key == ord('2'): target_sec = 31.0
            elif key == ord('3'): target_sec = 51.0
            elif key == ord('4'): target_sec = 79.0
            frame_idx = int(target_sec * fps)
            cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
            engine.reset()
            engine.start()
            print(f"[JUMP] Jumped to {target_sec:.1f}s and restarted Cover Test!")
        # Seek +/- 5 seconds
        elif not is_live and (key == ord('a') or key == ord('A')):
            cur_s = max(0.0, (frame_idx / fps) - 5.0)
            frame_idx = int(cur_s * fps)
            cap.set(cv2.CAP_PROP_POS_MSEC, cur_s * 1000.0)
            print(f"[SEEK] Rewound to {cur_s:.1f}s")
        elif not is_live and (key == ord('d') or key == ord('D')):
            cur_s = min(total_dur_s, (frame_idx / fps) + 5.0)
            frame_idx = int(cur_s * fps)
            cap.set(cv2.CAP_PROP_POS_MSEC, cur_s * 1000.0)
            print(f"[SEEK] Forwarded to {cur_s:.1f}s")

    if not engine.report_generated:
        engine._on_phase_finish(engine.phase)
        engine.analyze_results()

    cap.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()


# ============================================================
# CLI ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Automated Cover-Uncover Test")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to pre-recorded video file (e.g. strabismus-video/vidssave.com Cover Test 480P.mp4)")
    parser.add_argument("--webcam", type=int, default=None, help="Webcam device index (default 0)")
    parser.add_argument("--manual", action="store_true", help="Manual mode (press Space to advance phases)")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration in seconds per phase (default 3.0s)")
    parser.add_argument("--save_video", action="store_true", help="Save the annotated output video")
    args = parser.parse_args()

    video_src = args.video
    if video_src is None and args.webcam is not None:
        video_src = args.webcam

    run_cover_test(video_source=video_src, auto_start=not args.manual, phase_duration=args.duration, save_video=args.save_video)
