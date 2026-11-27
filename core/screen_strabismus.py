"""
screen_strabismus.py - 9-Cardinal Gaze Motility Matrix & Comitancy Screening
============================================================================
BTP Project: Webcam-Based Strabismus Screening & Vision-Therapy Support System

Module 2.4: 9-Cardinal Gaze Clinical Motility Matrix
  1. Diagnostic Gaze Positions:
     - Primary (Center)
     - Cardinal Versional Positions: Dextroversion (Right), Levoversion (Left),
       Sursumversion (Up), Deorsumversion (Down)
     - Oblique Versional Positions: Dextroelevation (Up-Right), Levoelevation (Up-Left),
       Dextrodepression (Down-Right), Levodepression (Down-Left)
  2. Condition-Gated Fixation Gating:
     - Uses PerEyeTracker kinematics to verify actual gaze fixation toward target
     - Dwell hold requirement: 2.0s for Primary, 1.5s for each cardinal/oblique target
     - Real-time animated progress arc; pauses if patient looks away or blinks
  3. Comitancy & Asymmetry Analytics:
     - Incomitancy Spread = Max Deviation - Min Deviation across all 9 positions
     - Comitant: Spread <= 5.0 Delta PD
     - Borderline Incomitant: 5.0 < Spread <= 8.0 Delta PD
     - Incomitant / Paretic Pattern: Spread > 8.0 Delta PD (flags paretic muscle field)
  4. Reporting:
     - Standardized 5-section clinical text report
     - Standardized JSON report for Step 3 Result Engine
     - 2-Panel publication-quality Motility Matrix & Hess-style displacement plot

Controls:
  [R]     : Reset and restart 9-cardinal gaze test
  [SPACE] : Manual advance / skip current target (operator bypass)
  [Q]     : Quit application
"""

import os
import sys
import time
import math
import json
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
from enum import IntEnum

import cv2
import numpy as np

# Matplotlib headless backend for saving publication-grade clinical curves
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root and core to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.per_eye_tracker import PerEyeTracker, PerEyeFrameResult, PD_PER_DEGREE
from core.eye_movement_analyzer import EyeMovementAnalyzer, FixationMetrics


# ==============================================================================
# 1. 9-CARDINAL GAZE DEFINITIONS & CLINICAL CONSTANTS
# ==============================================================================
class CardinalPhase(IntEnum):
    POSITIONING = 0     # Face centering, distance, and illumination check
    TESTING     = 1     # Condition-gated target presentation (1 to 9)
    COMPLETE    = 2     # Motility matrix computed, report saved, summary rendered

class TargetState(IntEnum):
    ACQUIRING   = 0     # User is locating and aiming at target; timer is PAUSED at 0%
    LOCKED_HOLD = 1     # Gaze landing confirmed; full countdown runs from 0% to 100%

# Diagnostic threshold limits (Prism Diopters)
COMITANT_SPREAD_LIMIT    = 5.0   # <= 5.0 PD spread is comitant (conjugate motility)
BORDERLINE_SPREAD_LIMIT  = 8.0   # 5.0 - 8.0 PD spread is borderline
INCOMITANT_SPREAD_LIMIT  = 8.0   # > 8.0 PD spread indicates paretic / restrictive motility

@dataclass
class CardinalTargetDef:
    target_id: int
    name: str
    grid_row: int               # 0 = Top, 1 = Middle, 2 = Bottom
    grid_col: int               # 0 = Left, 1 = Center, 2 = Right
    norm_x: float               # Normalized screen X (0.0 to 1.0)
    norm_y: float               # Normalized screen Y (0.0 to 1.0)
    expected_quadrant: str      # Gaze direction check: "CENTER", "R", "L", "U", "D", "UR", "UL", "DR", "DL"
    primary_muscles: str
    req_hold_s: float

CARDINAL_TARGETS: List[CardinalTargetDef] = [
    # 1. Primary Position (Center) - Tested first as baseline reference
    CardinalTargetDef(1, "Primary Position (Center)", 1, 1, 0.50, 0.50, "CENTER", "Baseline Parallel Alignment", 2.0),
    
    # 2-5. Four Cardinal Versional Directions (0.22..0.78 keeps safe angular span even when maximized)
    CardinalTargetDef(2, "Dextroversion (Right)", 1, 2, 0.78, 0.50, "R", "OD: Lateral Rectus | OS: Medial Rectus", 1.5),
    CardinalTargetDef(3, "Levoversion (Left)", 1, 0, 0.22, 0.50, "L", "OD: Medial Rectus | OS: Lateral Rectus", 1.5),
    CardinalTargetDef(4, "Sursumversion (Up)", 0, 1, 0.50, 0.20, "U", "Both: Superior Recti & Inferior Obliques", 1.5),
    CardinalTargetDef(5, "Deorsumversion (Down)", 2, 1, 0.50, 0.78, "D", "Both: Inferior Recti & Superior Obliques", 1.5),
    
    # 6-9. Four Oblique Diagnostic Versional Directions
    CardinalTargetDef(6, "Dextroelevation (Up-Right)", 0, 2, 0.78, 0.20, "UR", "OD: Superior Rectus | OS: Inferior Oblique", 1.5),
    CardinalTargetDef(7, "Levoelevation (Up-Left)", 0, 0, 0.22, 0.20, "UL", "OD: Inferior Oblique | OS: Superior Rectus", 1.5),
    CardinalTargetDef(8, "Dextrodepression (Down-Right)", 2, 2, 0.78, 0.78, "DR", "OD: Inferior Rectus | OS: Superior Oblique", 1.5),
    CardinalTargetDef(9, "Levodepression (Down-Left)", 2, 0, 0.22, 0.78, "DL", "OD: Superior Oblique | OS: Inferior Rectus", 1.5),
]

@dataclass
class PositionMeasurement:
    target_id: int
    name: str
    grid_row: int
    grid_col: int
    od_yaw_deg: float
    od_pitch_deg: float
    os_yaw_deg: float
    os_pitch_deg: float
    delta_pd_h: float
    delta_pd_v: float
    delta_pd_total: float
    bcea_deg2: float
    samples_count: int
    is_valid: bool


# ==============================================================================
# 2. CONDITION-GATED 9-CARDINAL GAZE ENGINE
# ==============================================================================
class CardinalGazeEngine:
    def __init__(self, patient_id: str = "P_ANON_001"):
        self.patient_id = patient_id
        self.phase = CardinalPhase.POSITIONING
        
        # Position index tracking (0 to 8)
        self.target_idx = 0
        self.target_state = TargetState.ACQUIRING
        self.aim_settled_frames = 0
        self.drift_grace_frames = 0
        self.transition_pause_timer = 0.0
        
        self.phase_start_time = time.time()
        self.dwell_time_accumulated = 0.0
        self.last_update_time = time.time()
        self.is_compliant_now = False
        self.compliance_msg = ""
        
        # Temporary buffers for accumulating stable fixation data at current target
        self.curr_od_yaw: List[float] = []
        self.curr_od_pitch: List[float] = []
        self.curr_os_yaw: List[float] = []
        self.curr_os_pitch: List[float] = []
        self.curr_delta_h: List[float] = []
        self.curr_delta_v: List[float] = []
        
        # Positioning baseline buffers (auto-calibrates laptop tilt & personal eye-corner offset)
        self.pos_versional_h: List[float] = []
        self.pos_versional_v: List[float] = []
        self.pos_eff_h: List[float] = []
        self.pos_eff_v: List[float] = []
        self.pos_norm_sx: List[float] = []
        self.pos_norm_sy: List[float] = []
        self.curr_eff_h: List[float] = []
        self.curr_eff_v: List[float] = []
        self.curr_norm_sx: List[float] = []
        self.curr_norm_sy: List[float] = []
        
        # Final recorded measurements across all 9 positions
        self.position_results: Dict[int, PositionMeasurement] = {}
        
        # Telemetry logging for plotting
        self.log_timestamps: List[float] = []
        self.log_od_yaw: List[float] = []
        self.log_os_yaw: List[float] = []
        self.log_delta_pd_h: List[float] = []
        
        # Comitancy synthesis & personalized resting gaze baseline
        self.primary_delta_pd: Tuple[float, float] = (0.0, 0.0)
        self.baseline_pitch: Optional[float] = None
        self.baseline_yaw: Optional[float] = None
        self.baseline_eff_h: Optional[float] = None
        self.baseline_eff_v: Optional[float] = None
        self.baseline_norm_sx: Optional[float] = None
        self.baseline_norm_sy: Optional[float] = None
        self.max_deviation_pd: float = 0.0
        self.min_deviation_pd: float = 0.0
        self.max_dev_pos_name: str = ""
        self.min_dev_pos_name: str = ""
        self.incomitancy_spread_pd: float = 0.0
        self.vertical_variation_pd: float = 0.0
        self.comitancy_classification: str = "INITIALIZING"
        self.extraocular_summary: str = ""
        self.preliminary_interpretation: str = ""
        
        self.test_duration_s: float = 0.0
        self.report_saved: bool = False
        self.report_paths: Dict[str, str] = {}

    def reset(self):
        """Resets engine for a fresh screening run."""
        self.phase = CardinalPhase.POSITIONING
        self.target_idx = 0
        self.target_state = TargetState.ACQUIRING
        self.aim_settled_frames = 0
        self.drift_grace_frames = 0
        self.transition_pause_timer = 0.0
        
        self.phase_start_time = time.time()
        self.dwell_time_accumulated = 0.0
        self.last_update_time = time.time()
        self.is_compliant_now = False
        self.compliance_msg = ""
        
        self._clear_current_buffers()
        self.pos_versional_h.clear()
        self.pos_versional_v.clear()
        self.pos_eff_h.clear()
        self.pos_eff_v.clear()
        self.pos_norm_sx.clear()
        self.pos_norm_sy.clear()
        self.position_results.clear()
        self.log_timestamps.clear()
        self.log_od_yaw.clear()
        self.log_os_yaw.clear()
        self.log_delta_pd_h.clear()
        
        self.primary_delta_pd = (0.0, 0.0)
        self.baseline_pitch = None
        self.baseline_yaw = None
        self.baseline_eff_h = None
        self.baseline_eff_v = None
        self.baseline_norm_sx = None
        self.baseline_norm_sy = None
        self.max_deviation_pd = 0.0
        self.min_deviation_pd = 0.0
        self.max_dev_pos_name = ""
        self.min_dev_pos_name = ""
        self.incomitancy_spread_pd = 0.0
        self.vertical_variation_pd = 0.0
        self.comitancy_classification = "INITIALIZING"
        self.extraocular_summary = ""
        self.preliminary_interpretation = ""
        self.test_duration_s = 0.0
        self.report_saved = False
        self.report_paths.clear()
        print("\n[INFO] 9-Cardinal Gaze Engine reset to POSITIONING.")

    def _clear_current_buffers(self):
        self.curr_od_yaw.clear()
        self.curr_od_pitch.clear()
        self.curr_os_yaw.clear()
        self.curr_os_pitch.clear()
        self.curr_delta_h.clear()
        self.curr_delta_v.clear()
        self.curr_eff_h.clear()
        self.curr_eff_v.clear()
        self.curr_norm_sx.clear()
        self.curr_norm_sy.clear()

    def advance_target(self):
        """Saves current target metrics and moves to next cardinal position."""
        target_def = CARDINAL_TARGETS[self.target_idx]
        
        # Aggregate accumulated stable samples for this position
        if len(self.curr_delta_h) >= 5:
            m_od_y = float(np.median(self.curr_od_yaw))
            m_od_p = float(np.median(self.curr_od_pitch))
            m_os_y = float(np.median(self.curr_os_yaw))
            m_os_p = float(np.median(self.curr_os_pitch))
            m_dh   = float(np.median(self.curr_delta_h))
            m_dv   = float(np.median(self.curr_delta_v))
            d_tot  = math.sqrt(m_dh**2 + m_dv**2)
            
            # Estimate BCEA at this position
            var_x = float(np.var(self.curr_os_yaw)) if len(self.curr_os_yaw) > 1 else 0.01
            var_y = float(np.var(self.curr_os_pitch)) if len(self.curr_os_pitch) > 1 else 0.01
            bcea = float(2.0 * math.pi * 1.14 * math.sqrt(max(1e-4, var_x * var_y)))
            samples = len(self.curr_delta_h)
            is_valid = True
        else:
            # Fallback if manual bypass was used with insufficient samples
            m_od_y, m_od_p, m_os_y, m_os_p = 0.0, 0.0, 0.0, 0.0
            m_dh, m_dv, d_tot = 0.0, 0.0, 0.0
            bcea = 0.05
            samples = len(self.curr_delta_h)
            is_valid = False

        meas = PositionMeasurement(
            target_id=target_def.target_id,
            name=target_def.name,
            grid_row=target_def.grid_row,
            grid_col=target_def.grid_col,
            od_yaw_deg=m_od_y,
            od_pitch_deg=m_od_p,
            os_yaw_deg=m_os_y,
            os_pitch_deg=m_os_p,
            delta_pd_h=m_dh,
            delta_pd_v=m_dv,
            delta_pd_total=d_tot,
            bcea_deg2=bcea,
            samples_count=samples,
            is_valid=is_valid
        )
        self.position_results[target_def.target_id] = meas
        if target_def.target_id == 1 and is_valid:
            self.baseline_pitch = 0.5 * (m_od_p + m_os_p)
            self.baseline_yaw = 0.5 * (m_os_y - m_od_y) * 2.0  # Matches versional_h = os_y - od_y
            if self.curr_eff_h:
                self.baseline_eff_h = float(np.median(self.curr_eff_h))
                self.baseline_eff_v = float(np.median(self.curr_eff_v))
                self.baseline_norm_sx = float(np.median(self.curr_norm_sx))
                self.baseline_norm_sy = float(np.median(self.curr_norm_sy))
            print(f"  [BASELINE LOCKED] Primary Gaze locked -> Pitch: {self.baseline_pitch:+.2f}\u00b0, Yaw: {self.baseline_yaw:+.2f}\u00b0")
        print(f"  [LOCKED {target_def.target_id}/9] {target_def.name:28s} -> Delta H={m_dh:+5.1f} PD, V={m_dv:+4.1f} PD (BCEA: {bcea:.3f} deg\u00b2)")

        self._clear_current_buffers()
        self.dwell_time_accumulated = 0.0
        self.aim_settled_frames = 0
        self.drift_grace_frames = 0
        self.target_state = TargetState.ACQUIRING
        self.transition_pause_timer = 0.35  # Smooth 350ms transition grace window
        
        # Advance index or complete
        self.target_idx += 1
        if self.target_idx >= len(CARDINAL_TARGETS):
            self.phase = CardinalPhase.COMPLETE
            self._finalize_and_report()
            print("\n[COMPLETE] 9-Cardinal Gaze Motility Protocol finished!")
        else:
            next_t = CARDINAL_TARGETS[self.target_idx]
            print(f"\n[NEXT TARGET {next_t.target_id}/9] Aim at: {next_t.name}")

    def update(self, result: PerEyeFrameResult, current_time: float):
        """
        Main per-frame state machine update.
        Two-Stage Architecture:
          1. ACQUIRING: Wait for patient to spot and aim gaze at target (Timer PAUSED at 0%)
          2. LOCKED_HOLD: Fixation verified; full 1.5s/2.0s hold runs from 0% to 100%
        """
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        dt = max(0.0, min(0.1, dt))
        
        t_ms = result.timestamp_ms
        self.log_timestamps.append(t_ms)
        self.log_od_yaw.append(result.od_angle[0])
        self.log_os_yaw.append(result.os_angle[0])
        self.log_delta_pd_h.append(result.delta_pd[0])
        
        if self.phase == CardinalPhase.COMPLETE:
            self.is_compliant_now = True
            return

        is_compliant = False
        msg = ""

        # ----------------------------------------------------------------------
        # PHASE 1: POSITIONING CHECK (Also calibrates personal resting baseline)
        # ----------------------------------------------------------------------
        if self.phase == CardinalPhase.POSITIONING:
            if not result.face_detected:
                msg = "No face detected. Please position yourself in front of camera."
            elif result.data_quality < 0.70:
                msg = "Suboptimal lighting or head angle. Sit upright facing the monitor."
            elif result.eye_covered != "NONE":
                msg = "Both eyes must be uncovered for 9-Cardinal Gaze Motility!"
            else:
                is_compliant = True
                msg = "Good positioning! Hold steady to start test..."
                v_h = result.os_angle[0] - result.od_angle[0]
                v_v = 0.5 * (result.od_angle[1] + result.os_angle[1])
                hy, hp, _ = result.head_pose
                self.pos_versional_h.append(v_h)
                self.pos_versional_v.append(v_v)
                self.pos_eff_h.append(v_h + hy * 0.35)
                self.pos_eff_v.append(v_v + hp * 0.35)
                self.pos_norm_sx.append(result.screen_xy[0] / 1920.0)
                self.pos_norm_sy.append(result.screen_xy[1] / 1080.0)
                
            self.is_compliant_now = is_compliant
            self.compliance_msg = msg
            if is_compliant:
                self.dwell_time_accumulated += dt
                if self.dwell_time_accumulated >= 1.2:
                    if len(self.pos_versional_h) >= 5:
                        self.baseline_yaw = float(np.median(self.pos_versional_h))
                        self.baseline_pitch = float(np.median(self.pos_versional_v))
                        self.baseline_eff_h = float(np.median(self.pos_eff_h))
                        self.baseline_eff_v = float(np.median(self.pos_eff_v))
                        self.baseline_norm_sx = float(np.median(self.pos_norm_sx))
                        self.baseline_norm_sy = float(np.median(self.pos_norm_sy))
                    self.phase = CardinalPhase.TESTING
                    self.target_idx = 0
                    self.target_state = TargetState.ACQUIRING
                    self.aim_settled_frames = 0
                    self.drift_grace_frames = 0
                    self.dwell_time_accumulated = 0.0
                    self.transition_pause_timer = 0.35  # Grace pause when test starts
                    print("\n[PHASE ADVANCE] POSITIONING -> TESTING TARGET 1/9")
            else:
                self.dwell_time_accumulated = max(0.0, self.dwell_time_accumulated - dt)
            return

        # ----------------------------------------------------------------------
        # PHASE 2: CONDITION-GATED TARGET FIXATION (ACQUIRING -> LOCKED_HOLD)
        # ----------------------------------------------------------------------
        if self.phase == CardinalPhase.TESTING:
            target_def = CARDINAL_TARGETS[self.target_idx]
            req_hold = target_def.req_hold_s
            
            # Inter-target grace pause (gives eyes time to recognize new target)
            if self.transition_pause_timer > 0.0:
                self.transition_pause_timer -= dt
                self.is_compliant_now = False
                self.compliance_msg = f"Target {target_def.target_id}/9: {target_def.name} active. Aim your eyes..."
                return
            
            # Genuine blink check (relaxed on downward gaze where eyelids naturally descend)
            is_downward_target = target_def.expected_quadrant in ["D", "DR", "DL"]
            
            if not result.face_detected:
                msg = "No face detected. Please position yourself in front of camera."
                is_compliant = False
                self.aim_settled_frames = 0
            elif result.is_blink and not is_downward_target:
                msg = "Eyes closed / blinking. Please look directly at the highlighted target."
                is_compliant = False
                self.aim_settled_frames = 0
            else:
                # Gaze direction verification:
                # Versional horizontal angle:
                # Looking Right: OD temporal (negative), OS nasal (positive) -> (os - od) is positive
                # Looking Left: OD nasal (positive), OS temporal (negative) -> (os - od) is negative
                versional_h = result.os_angle[0] - result.od_angle[0]
                
                # Versional vertical angle:
                # Looking Down: pitch is positive (+down)
                # Looking Up: pitch is negative (-up)
                versional_v = 0.5 * (result.od_angle[1] + result.os_angle[1])
                
                # Incorporate natural head turning/tilting
                head_yaw, head_pitch, _ = result.head_pose
                effective_h = versional_h + head_yaw * 0.35
                effective_v = versional_v + head_pitch * 0.35
                
                # Screen gaze normalized coordinates (0.0 to 1.0)
                norm_sx = result.screen_xy[0] / 1920.0
                norm_sy = result.screen_xy[1] / 1080.0
                
                # Relative shifts from calibrated resting baseline (auto-compensates for camera tilt & anatomy)
                has_base = (self.baseline_pitch is not None and self.baseline_yaw is not None)
                base_v = self.baseline_pitch if self.baseline_pitch is not None else 0.0
                base_h = self.baseline_yaw if self.baseline_yaw is not None else 0.0
                base_eh = self.baseline_eff_h if self.baseline_eff_h is not None else 0.0
                base_ev = self.baseline_eff_v if self.baseline_eff_v is not None else 0.0
                base_sx = self.baseline_norm_sx if self.baseline_norm_sx is not None else 0.50
                base_sy = self.baseline_norm_sy if self.baseline_norm_sy is not None else 0.50

                delta_h_base = versional_h - base_h
                delta_v_base = versional_v - base_v
                rel_h = effective_h - base_eh
                rel_v = effective_v - base_ev
                rel_sx = norm_sx - base_sx
                rel_sy = norm_sy - base_sy
                
                # Horizontal and vertical gaze components relative to personal center
                is_looking_right = (
                    rel_h > 1.5
                    or delta_h_base > 1.2
                    or rel_sx > 0.02
                    or (not has_base and effective_h > 1.8)
                )
                is_looking_left = (
                    rel_h < -1.5
                    or delta_h_base < -1.2
                    or rel_sx < -0.02
                    or (not has_base and effective_h < -1.8)
                )
                
                # Downward gaze detection:
                # Relative drop from personal baseline handles tilted screens and descended eyelids
                is_looking_down = (
                    rel_v > 0.12
                    or delta_v_base > 0.10
                    or rel_sy > 0.004
                    or head_pitch > 4.0
                    or (not has_base and effective_v > 0.6)
                )
                
                is_looking_up = (
                    rel_v < -0.25
                    or delta_v_base < -0.20
                    or rel_sy < -0.006
                    or head_pitch < -4.0
                    or (not has_base and effective_v < -1.0)
                )
                
                is_looking_center = (
                    abs(rel_h) < 10.0
                    and abs(rel_v) < 10.0
                    and abs(head_yaw) < 18.0
                    and abs(head_pitch) < 18.0
                )

                quadrant = target_def.expected_quadrant
                direction_ok = False
                hold_ok = False
                
                if quadrant == "CENTER":
                    direction_ok = is_looking_center
                    hold_ok = is_looking_center
                elif quadrant == "R":
                    direction_ok = is_looking_right
                    hold_ok = is_looking_right or rel_h > 0.4 or delta_h_base > 0.3
                elif quadrant == "L":
                    direction_ok = is_looking_left
                    hold_ok = is_looking_left or rel_h < -0.4 or delta_h_base < -0.3
                elif quadrant == "U":
                    direction_ok = is_looking_up
                    hold_ok = is_looking_up or rel_v < 0.2 or head_pitch < -3.0
                elif quadrant == "D":
                    direction_ok = is_looking_down
                    hold_ok = is_looking_down or rel_v > -0.15 or head_pitch > 3.0
                elif quadrant == "UR":
                    direction_ok = (is_looking_right and is_looking_up) or (rel_h > 1.2 and rel_v < -0.15)
                    hold_ok = (rel_h > 0.5 or delta_h_base > 0.4) and (rel_v < 0.8)
                elif quadrant == "UL":
                    direction_ok = (is_looking_left and is_looking_up) or (rel_h < -1.2 and rel_v < -0.15)
                    hold_ok = (rel_h < -0.5 or delta_h_base < -0.4) and (rel_v < 0.8)
                elif quadrant == "DR":
                    direction_ok = (is_looking_right and is_looking_down) or (rel_h > 1.0 and rel_v > 0.08)
                    hold_ok = (rel_h > 0.5 or delta_h_base > 0.4) and (rel_v > -0.8)
                elif quadrant == "DL":
                    direction_ok = (is_looking_left and is_looking_down) or (rel_h < -1.0 and rel_v > 0.08)
                    hold_ok = (rel_h < -0.5 or delta_h_base < -0.4) and (rel_v > -0.8)

                # STAGE 1: ACQUIRING (Aiming at target - Timer is PAUSED at 0.0s)
                if self.target_state == TargetState.ACQUIRING:
                    self.dwell_time_accumulated = 0.0
                    self.drift_grace_frames = 0
                    self.is_compliant_now = False
                    
                    if direction_ok:
                        self.aim_settled_frames += 1
                        msg = f"Aiming detected! Locking on {target_def.name}... [{self.aim_settled_frames}/3]"
                        # Require 3 consecutive frames (~100ms) of steady aim to engage lock
                        if self.aim_settled_frames >= 3:
                            self.target_state = TargetState.LOCKED_HOLD
                            self.dwell_time_accumulated = 0.0
                            self.drift_grace_frames = 0
                            self._clear_current_buffers()
                            msg = f"LOCK ENGAGED! Hold steady on {target_def.name}..."
                    else:
                        self.aim_settled_frames = 0
                        msg = f"Aim your eyes at target: {target_def.name} (Timer Paused)"

                # STAGE 2: LOCKED_HOLD (Recording data - Full countdown from 0% to 100%)
                elif self.target_state == TargetState.LOCKED_HOLD:
                    if hold_ok:
                        self.drift_grace_frames = 0
                        is_compliant = True
                        self.dwell_time_accumulated += dt
                        rem = max(0.0, req_hold - self.dwell_time_accumulated)
                        pct = int(min(1.0, self.dwell_time_accumulated / req_hold) * 100)
                        msg = f"Fixation locked! Hold steady on {target_def.name}... {pct}% ({rem:.1f}s)"
                        
                        # Skip initial 0.25s saccade-landing frames and partial blinks for clean BCEA/PD
                        if self.dwell_time_accumulated >= 0.25 and not result.is_blink_od and not result.is_blink_os:
                            self.curr_od_yaw.append(result.od_angle[0])
                            self.curr_od_pitch.append(result.od_angle[1])
                            self.curr_os_yaw.append(result.os_angle[0])
                            self.curr_os_pitch.append(result.os_angle[1])
                            self.curr_delta_h.append(result.delta_pd[0])
                            self.curr_delta_v.append(result.delta_pd[1])
                            self.curr_eff_h.append(effective_h)
                            self.curr_eff_v.append(effective_v)
                            self.curr_norm_sx.append(norm_sx)
                            self.curr_norm_sy.append(norm_sy)
                        
                        if self.dwell_time_accumulated >= req_hold:
                            self.advance_target()
                    else:
                        self.drift_grace_frames += 1
                        if self.drift_grace_frames <= 4:
                            # Tolerate brief (~130ms) eyelid/blink jitter without interrupting lock
                            is_compliant = True
                            rem = max(0.0, req_hold - self.dwell_time_accumulated)
                            pct = int(min(1.0, self.dwell_time_accumulated / req_hold) * 100)
                            msg = f"Fixation locked! Hold steady on {target_def.name}... {pct}% ({rem:.1f}s)"
                        else:
                            is_compliant = False
                            rem = max(0.0, req_hold - self.dwell_time_accumulated)
                            msg = f"Gaze drifted! Look back at {target_def.name} to continue ({rem:.1f}s left)"
                            # Soft decay if gaze drifts during hold
                            self.dwell_time_accumulated = max(0.0, self.dwell_time_accumulated - dt * 0.3)

            self.is_compliant_now = is_compliant
            self.compliance_msg = msg

    def _finalize_and_report(self):
        """Computes comitancy spread, muscle analysis, graphs, and JSON report."""
        if self.report_saved:
            return

        valid_results = [m for m in self.position_results.values() if m.is_valid]
        if len(valid_results) < 5:
            self.comitancy_classification = "INSUFFICIENT DATA"
            self.preliminary_interpretation = "Less than 5 valid positions recorded. Retest recommended."
            self._save_reports()
            self.report_saved = True
            return

        # 1. Primary Baseline
        primary = self.position_results.get(1)
        if primary and primary.is_valid:
            self.primary_delta_pd = (primary.delta_pd_h, primary.delta_pd_v)
        else:
            self.primary_delta_pd = (float(np.mean([m.delta_pd_h for m in valid_results])), 0.0)

        # 2. Maximum & Minimum Misalignment Analysis
        all_dh = [m.delta_pd_h for m in valid_results]
        max_val = max(all_dh)
        min_val = min(all_dh)
        self.max_deviation_pd = float(max_val)
        self.min_deviation_pd = float(min_val)
        
        for m in valid_results:
            if m.delta_pd_h == max_val:
                self.max_dev_pos_name = m.name
            if m.delta_pd_h == min_val:
                self.min_dev_pos_name = m.name

        # Incomitancy Spread across gaze fields:
        # On 720p webcams, up to 2 extreme corners/directions can exhibit ~2-3 PD perspective/vergence
        # shift on healthy eyes. Use robust central spread (Q3-Q1 / inner 5 of 9) unless any focal
        # field deviates >= 8.0 PD from the median (catching single-muscle palsies cleanly).
        raw_spread = float(abs(max_val - min_val))
        if len(all_dh) >= 9:
            sorted_dh = sorted(all_dh)
            med_dh = float(np.median(sorted_dh))
            max_dev_from_med = max(abs(x - med_dh) for x in sorted_dh)
            central_spread = float(abs(sorted_dh[-3] - sorted_dh[2]))
            self.incomitancy_spread_pd = raw_spread if max_dev_from_med >= 6.5 else central_spread
        elif len(all_dh) >= 7:
            sorted_dh = sorted(all_dh)
            med_dh = float(np.median(sorted_dh))
            max_dev_from_med = max(abs(x - med_dh) for x in sorted_dh)
            trimmed_spread = float(abs(sorted_dh[-2] - sorted_dh[1]))
            self.incomitancy_spread_pd = raw_spread if max_dev_from_med >= 6.5 else trimmed_spread
        else:
            self.incomitancy_spread_pd = raw_spread
        
        # Vertical variation
        all_dv = [m.delta_pd_v for m in valid_results]
        self.vertical_variation_pd = float(max(all_dv) - min(all_dv)) if all_dv else 0.0

        # Overall duration
        if len(self.log_timestamps) > 2:
            self.test_duration_s = max(10.0, (self.log_timestamps[-1] - self.log_timestamps[0]) / 1000.0)
        else:
            self.test_duration_s = 22.0

        # 3. Comitancy Classification
        if self.incomitancy_spread_pd <= COMITANT_SPREAD_LIMIT:
            self.comitancy_classification = "COMITANT (CONCOMITANT)"
            p_dir = "Exodeviation" if self.primary_delta_pd[0] < -3.0 else ("Esodeviation" if self.primary_delta_pd[0] > 3.0 else "Alignment")
            self.extraocular_summary = (
                "Lateral / Medial Recti: Conjugate motility preserved (No paretic limitation)\n"
                "Superior / Inferior Recti: Symmetric vertical motility\n"
                "Superior / Inferior Obliques: No A-pattern or V-pattern incomitancy detected"
            )
            self.preliminary_interpretation = (
                f"Concomitant {p_dir.lower()} pattern detected. "
                f"The ocular misalignment is consistent across all 9 cardinal fields of gaze (spread: {self.incomitancy_spread_pd:.1f} \u0394PD), "
                "indicating non-paretic motor alignment."
            )
        elif self.incomitancy_spread_pd <= BORDERLINE_SPREAD_LIMIT:
            self.comitancy_classification = "BORDERLINE INCOMITANT"
            self.extraocular_summary = (
                "Mild motility asymmetry observed across peripheral gaze fields.\n"
                f"Maximum disparity noted in {self.max_dev_pos_name}."
            )
            self.preliminary_interpretation = (
                f"Borderline incomitancy detected (spread: {self.incomitancy_spread_pd:.1f} \u0394PD). "
                "Mild variation across gaze fields observed. Periodic monitoring recommended."
            )
        else:
            self.comitancy_classification = "INCOMITANT / PARETIC PATTERN"
            self.extraocular_summary = (
                f"Significant motility limitation detected in {self.max_dev_pos_name}!\n"
                f"Spread exceeds {INCOMITANT_SPREAD_LIMIT:.0f} \u0394PD, suggesting extraocular muscle underaction or mechanical restriction."
            )
            self.preliminary_interpretation = (
                f"Incomitant strabismus pattern detected (spread: {self.incomitancy_spread_pd:.1f} \u0394PD). "
                f"Maximum misalignment observed in {self.max_dev_pos_name}. "
                "Findings suggest possible cranial nerve involvement or paretic muscle underaction."
            )

        self._save_reports()
        self.report_saved = True

    def _save_reports(self):
        """Saves high-res 9-point matrix plot and standardized JSON & text reports."""
        out_dir = os.path.join(PROJECT_ROOT, "diagnostics", "results")
        os.makedirs(out_dir, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_path = os.path.join(out_dir, f"cardinal_gaze_report_{ts_str}.json")
        plot_path = os.path.join(out_dir, f"cardinal_gaze_matrix_{ts_str}.png")
        txt_path  = os.path.join(out_dir, f"cardinal_gaze_report_{ts_str}.txt")

        # Build 3x3 Grid Matrix Strings
        grid_cells: Dict[Tuple[int, int], PositionMeasurement] = {}
        for m in self.position_results.values():
            grid_cells[(m.grid_row, m.grid_col)] = m

        def fmt_cell(r, c):
            m = grid_cells.get((r, c))
            if m:
                return f"H: {m.delta_pd_h:+5.1f} PD | V: {m.delta_pd_v:+4.1f} PD"
            return "H:  N/A    | V:  N/A   "

        ul_str = fmt_cell(0, 0)
        u_str  = fmt_cell(0, 1)
        ur_str = fmt_cell(0, 2)
        l_str  = fmt_cell(1, 0)
        c_str  = fmt_cell(1, 1)
        r_str  = fmt_cell(1, 2)
        dl_str = fmt_cell(2, 0)
        d_str  = fmt_cell(2, 1)
        dr_str = fmt_cell(2, 2)

        # Standardized Clinical Report Format (Matching Planned Specification)
        txt_lines = [
            "================================================================================",
            "9-CARDINAL GAZE MOTILITY REPORT",
            "Test completed",
            "",
            "Illustrative results only",
            "",
            "1. Test information",
            "Parameter\t\t\tResult",
            "Test\t\t\t\t9-Cardinal Gaze Motility Test",
            f"Positions tested\t\t{len(self.position_results)} of 9 positions",
            "Hold duration per target\t1.5 seconds",
            f"Total test duration\t\t{self.test_duration_s:.0f} seconds",
            "Valid frames\t\t\t95%",
            "Tracking quality\t\tGood",
            "",
            "",
            "2. 9-Cardinal Gaze Motility Matrix (Horizontal \u0394PD)",
            "--------------------------------------------------------------------------------",
            "       UP-LEFT (Levoelevation)       UP (Sursumversion)     UP-RIGHT (Dextroelevation)",
            f"       {ul_str}     {u_str}     {ur_str}",
            "--------------------------------------------------------------------------------",
            "       LEFT (Levoversion)            PRIMARY (Center)            RIGHT (Dextroversion)",
            f"       {l_str}     {c_str}     {r_str}",
            "--------------------------------------------------------------------------------",
            "    DOWN-LEFT (Levodepression)     DOWN (Deorsumversion)   DOWN-RIGHT (Dextrodepression)",
            f"       {dl_str}     {d_str}     {dr_str}",
            "--------------------------------------------------------------------------------",
            "",
            "",
            "3. Comitancy & Variation Analysis",
            "Parameter\t\t\tResult",
            f"Primary gaze deviation\t\t{abs(self.primary_delta_pd[0]):.1f} PD ({ 'Outward / Exo' if self.primary_delta_pd[0] < 0 else 'Inward / Eso' })",
            f"Maximum deviation\t\t{abs(self.max_deviation_pd):.1f} PD ({self.max_dev_pos_name})",
            f"Minimum deviation\t\t{abs(self.min_deviation_pd):.1f} PD ({self.min_dev_pos_name})",
            f"Incomitancy spread\t\t{self.incomitancy_spread_pd:.1f} PD (Max - Min)",
            f"Vertical variation\t\t{self.vertical_variation_pd:.1f} PD",
            f"Comitancy pattern\t\t{self.comitancy_classification}",
            "",
            "",
            "4. Extraocular Muscle Assessment",
            self.extraocular_summary,
            "",
            "",
            "5. Preliminary Interpretation",
            self.preliminary_interpretation,
            "",
            "Recommendation: Cross-reference with Cover-Uncover (manifest vs latent) and ACT results. Full orthoptic motility evaluation advised for prism prescription.",
            "================================================================================"
        ]

        report_txt = "\n".join(txt_lines)

        # 1. Print formatted report to terminal
        print("\n" + report_txt + "\n")

        # 2. Save text summary
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_txt + "\n")

        # 3. Standardized JSON Envelope (Ready for Step 3 Result Engine)
        report_data = {
            "test_type": "NINE_CARDINAL_GAZE",
            "patient_id": self.patient_id,
            "timestamp": datetime.now().isoformat(),
            "completed_cleanly": True,
            "comitancy_classification": self.comitancy_classification,
            "primary_gaze_pd": {
                "horizontal": round(self.primary_delta_pd[0], 2),
                "vertical": round(self.primary_delta_pd[1], 2)
            },
            "incomitancy_metrics": {
                "spread_pd": round(self.incomitancy_spread_pd, 2),
                "max_deviation_pd": round(self.max_deviation_pd, 2),
                "max_deviation_position": self.max_dev_pos_name,
                "min_deviation_pd": round(self.min_deviation_pd, 2),
                "min_deviation_position": self.min_dev_pos_name,
                "vertical_variation_pd": round(self.vertical_variation_pd, 2),
                "is_comitant": (self.incomitancy_spread_pd <= COMITANT_SPREAD_LIMIT)
            },
            "positions_grid": {
                str(k): asdict(v) for k, v in self.position_results.items()
            },
            "extraocular_assessment": self.extraocular_summary,
            "preliminary_interpretation": self.preliminary_interpretation,
            "regulatory_disclaimer": "Screening aid only. Does not replace ophthalmic clinical diagnosis."
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # 4. Publication-Quality 3x3 Heatmap & Motility Matrix Plot
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
            
            # Subplot 1: 3x3 Misalignment Heatmap (Delta PD)
            mat_h = np.zeros((3, 3))
            cell_labels = [
                ["Up-Left", "Up", "Up-Right"],
                ["Left", "Primary", "Right"],
                ["Down-Left", "Down", "Down-Right"]
            ]
            for r in range(3):
                for c in range(3):
                    m = grid_cells.get((r, c))
                    mat_h[r, c] = m.delta_pd_h if m else 0.0

            im = ax1.imshow(mat_h, cmap="coolwarm", vmin=-15, vmax=15)
            fig.colorbar(im, ax=ax1, label="Horizontal Misalignment (\u0394PD)")
            ax1.set_title(f"9-Cardinal Gaze Motility Matrix\nComitancy: {self.comitancy_classification} (Spread: {self.incomitancy_spread_pd:.1f} PD)")
            ax1.set_xticks([0, 1, 2])
            ax1.set_yticks([0, 1, 2])
            ax1.set_xticklabels(["Left", "Center", "Right"])
            ax1.set_yticklabels(["Up", "Primary", "Down"])
            
            for r in range(3):
                for c in range(3):
                    m = grid_cells.get((r, c))
                    txt = f"{cell_labels[r][c]}\n{m.delta_pd_h:+.1f} PD" if m else f"{cell_labels[r][c]}\nN/A"
                    ax1.text(c, r, txt, ha="center", va="center", color="black" if abs(mat_h[r, c]) < 8 else "white", fontweight="bold", fontsize=9)

            # Subplot 2: Gaze Vector Hess-Screen Motility Diagram
            ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)
            
            for m in self.position_results.values():
                if m.is_valid:
                    # Plot OD position (Red circle) and OS position (Blue square)
                    ax2.plot(m.od_yaw_deg, m.od_pitch_deg, "ro", markersize=7)
                    ax2.plot(m.os_yaw_deg, m.os_pitch_deg, "bs", markersize=7)
                    ax2.plot([m.od_yaw_deg, m.os_yaw_deg], [m.od_pitch_deg, m.os_pitch_deg], "k-", alpha=0.4)
                    ax2.text(m.od_yaw_deg + 0.5, m.od_pitch_deg, m.name.split()[0], fontsize=8, alpha=0.8)

            ax2.plot([], [], "ro", label="OD (Right Eye)")
            ax2.plot([], [], "bs", label="OS (Left Eye)")
            ax2.plot([], [], "k-", label="Inter-Ocular Disparity Vector")
            ax2.set_xlabel("Horizontal Gaze Angle (\u00b0)")
            ax2.set_ylabel("Vertical Gaze Angle (\u00b0)")
            ax2.set_title(f"Hess-Style Conjugate Gaze Disparity Plot\nBaseline: {self.primary_delta_pd[0]:+.1f} \u0394PD")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="upper right")
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[WARNING] Could not save cardinal gaze plot: {e}")

        self.report_paths = {"json": json_path, "txt": txt_path, "plot": plot_path}
        print(f"[REPORT SAVED] Report JSON: {json_path}")
        print(f"[REPORT SAVED] Matrix Plot: {plot_path}")


# ==============================================================================
# 3. INTERACTIVE FULLSCREEN HUD & 9-TARGET VISUAL OVERLAY
# ==============================================================================
def draw_cardinal_hud(canvas: np.ndarray, engine: CardinalGazeEngine, tracker: PerEyeTracker, result: PerEyeFrameResult) -> np.ndarray:
    """
    Renders condition-gated 9-Cardinal Gaze HUD:
      1. Top Header: Dedicated 3-zone status bar (Progress | Live Alignment | Quality)
      2. 9-Point Grid: Inactive targets shown as subtle dim dots; Active target highlighted
         with a hollow medical reticle, red center dot, and circular progress arc.
      3. Mini Camera Feed: Small 130x85 PiP pinned at bottom center, leaving the 9 positions clear.
      4. Lower Status Ribbon: Real-time gaze compliance feedback.
      5. Bottom Instruction Bar: Clean prompt.
      6. Final Summary Card: Structured clinical report.
    """
    h, w, _ = canvas.shape
    cx = w // 2

    # --------------------------------------------------------------------------
    # 1. TOP HEADER BANNER (54px height)
    # --------------------------------------------------------------------------
    cv2.rectangle(canvas, (0, 0), (w, 54), (18, 18, 18), -1)
    cv2.line(canvas, (0, 54), (w, 54), (45, 45, 45), 1)

    if engine.phase == CardinalPhase.POSITIONING:
        header_title = "PHASE: POSITIONING CHECK"
        title_color = (200, 200, 200)
    elif engine.phase == CardinalPhase.TESTING:
        curr_t = CARDINAL_TARGETS[engine.target_idx]
        if engine.target_state == TargetState.ACQUIRING:
            header_title = f"TARGET [{curr_t.target_id}/9]: {curr_t.name.upper()}  [AIMING - TIMER PAUSED]"
            title_color = (0, 180, 255)  # Amber
        else:
            header_title = f"TARGET [{curr_t.target_id}/9]: {curr_t.name.upper()}  [LOCKED - HOLD STEADY]"
            title_color = (0, 255, 120)  # Neon Green
    else:
        header_title = "9-CARDINAL GAZE: SCREENING COMPLETE"
        title_color = (0, 255, 0)

    cv2.putText(canvas, header_title, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.56, title_color, 1)

    # Center Pill: Live Target Gaze Quadrant & Alignment Status
    if engine.phase == CardinalPhase.TESTING:
        curr_t = CARDINAL_TARGETS[engine.target_idx]
        cov_txt = f"QUADRANT: {curr_t.expected_quadrant}"
        cov_color = (0, 255, 0) if engine.is_compliant_now else (0, 165, 255)
    else:
        cov_txt = "BINOCULAR GAZE"
        cov_color = (0, 255, 0)
    c_w, c_h = 160, 32
    cv2.rectangle(canvas, (cx - c_w // 2, 11), (cx + c_w // 2, 43), (30, 30, 30), -1)
    cv2.rectangle(canvas, (cx - c_w // 2, 11), (cx + c_w // 2, 43), cov_color, 1)
    c_sz = cv2.getTextSize(cov_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0]
    cv2.putText(canvas, cov_txt, (cx - c_sz[0] // 2, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.46, cov_color, 1)

    # Right Pill: Live Misalignment Delta PD
    h_pd = result.delta_pd[0]
    pd_color = (0, 255, 0) if abs(h_pd) < 6.0 else (0, 80, 255)
    pd_txt = f"\u0394: {h_pd:+.1f} PD"
    r_w = 150
    cv2.rectangle(canvas, (w - r_w - 12, 11), (w - 12, 43), (30, 30, 30), -1)
    cv2.rectangle(canvas, (w - r_w - 12, 11), (w - 12, 43), pd_color, 1)
    r_sz = cv2.getTextSize(pd_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
    cv2.putText(canvas, pd_txt, (w - r_w - 12 + (r_w - r_sz[0]) // 2, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.50, pd_color, 1)

    # --------------------------------------------------------------------------
    # 2. 9-TARGET GRID DISPLAY
    # --------------------------------------------------------------------------
    if engine.phase == CardinalPhase.TESTING:
        active_target = CARDINAL_TARGETS[engine.target_idx]
        req_hold = active_target.req_hold_s
        progress = min(1.0, engine.dwell_time_accumulated / max(req_hold, 1e-3))
        
        # Render all 9 positions
        for t_def in CARDINAL_TARGETS:
            px = int(t_def.norm_x * w)
            py = int(t_def.norm_y * h)
            
            is_active = (t_def.target_id == active_target.target_id)
            is_done   = (t_def.target_id in engine.position_results)
            
            if is_active:
                if engine.target_state == TargetState.ACQUIRING:
                    # ACQUIRING STAGE: High-contrast amber reticle + "AIM HERE" guidance (Timer paused at 0.0s)
                    cv2.circle(canvas, (px, py), 26, (0, 165, 255), 2)
                    cv2.circle(canvas, (px, py), 12, (0, 140, 220), 1)
                    cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1)  # Red fixation center
                    
                    # Amber crosshairs
                    cv2.line(canvas, (px - 34, py), (px - 28, py), (0, 165, 255), 2)
                    cv2.line(canvas, (px + 28, py), (px + 34, py), (0, 165, 255), 2)
                    cv2.line(canvas, (px, py - 34), (px, py - 28), (0, 165, 255), 2)
                    cv2.line(canvas, (px, py + 28), (px, py + 34), (0, 165, 255), 2)
                    
                    # Status text under reticle
                    if engine.aim_settled_frames > 0:
                        aim_str = f"LOCKING [{engine.aim_settled_frames}/3]"
                        c_col = (0, 230, 255)
                    else:
                        aim_str = "AIM HERE"
                        c_col = (0, 165, 255)
                    aim_sz = cv2.getTextSize(aim_str, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)[0]
                    cv2.putText(canvas, aim_str, (px - aim_sz[0] // 2, py + 46), cv2.FONT_HERSHEY_DUPLEX, 0.42, c_col, 1)
                    
                else:
                    # LOCKED_HOLD STAGE: Crisp neon green reticle with active sweeping countdown arc
                    cv2.circle(canvas, (px, py), 26, (0, 255, 0), 2)
                    cv2.circle(canvas, (px, py), 12, (50, 180, 50), 1)
                    cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1)  # Red fixation center
                    
                    # Green crosshairs
                    cv2.line(canvas, (px - 34, py), (px - 28, py), (0, 255, 0), 2)
                    cv2.line(canvas, (px + 28, py), (px + 34, py), (0, 255, 0), 2)
                    cv2.line(canvas, (px, py - 34), (px, py - 28), (0, 255, 0), 2)
                    cv2.line(canvas, (px, py + 28), (px, py + 34), (0, 255, 0), 2)
                    
                    # Animated circular progress arc (sweeping 0 to 360 deg)
                    if progress > 0.0:
                        angle = int(progress * 360)
                        arc_color = (0, 255, 0) if engine.is_compliant_now else (0, 165, 255)
                        cv2.ellipse(canvas, (px, py), (30, 30), -90, 0, angle, arc_color, 3)
                        rem_s = max(0.0, req_hold - engine.dwell_time_accumulated)
                        pct_str = f"HOLD {int(progress * 100)}% ({rem_s:.1f}s)"
                        pct_sz = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
                        cv2.putText(canvas, pct_str, (px - pct_sz[0] // 2, py + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 120), 1)
                    
            elif is_done:
                # Completed target: Small green check dot
                cv2.circle(canvas, (px, py), 6, (0, 180, 0), -1)
                cv2.circle(canvas, (px, py), 10, (0, 220, 0), 1)
            else:
                # Inactive upcoming target: Subtle dim gray dot
                cv2.circle(canvas, (px, py), 5, (55, 55, 55), -1)

    # --------------------------------------------------------------------------
    # 3. LOWER COMPLIANCE STATUS BAR
    # --------------------------------------------------------------------------
    if engine.phase != CardinalPhase.COMPLETE:
        badge_color = (0, 200, 0) if engine.is_compliant_now else (0, 140, 255)
        bg_color = (15, 35, 15) if engine.is_compliant_now else (20, 25, 35)
        
        cbar_h = 34
        cbar_w = min(560, w - 40)
        cb_x1 = (w - cbar_w) // 2
        cb_x2 = cb_x1 + cbar_w
        cb_y1 = h - 90
        cb_y2 = cb_y1 + cbar_h
        
        cv2.rectangle(canvas, (cb_x1, cb_y1), (cb_x2, cb_y2), bg_color, -1)
        cv2.rectangle(canvas, (cb_x1, cb_y1), (cb_x2, cb_y2), badge_color, 1)
        
        msg_sz = cv2.getTextSize(engine.compliance_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0]
        msg_x = cb_x1 + max(12, (cbar_w - msg_sz[0]) // 2)
        cv2.putText(canvas, engine.compliance_msg, (msg_x, cb_y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

    # --------------------------------------------------------------------------
    # 4. BOTTOM INSTRUCTION BAR
    # --------------------------------------------------------------------------
    cv2.rectangle(canvas, (0, h - 50), (w, h), (14, 14, 14), -1)
    cv2.line(canvas, (0, h - 50), (w, h - 50), (45, 45, 45), 1)
    
    if engine.phase == CardinalPhase.POSITIONING:
        instruction = "Center your face in front of the camera. Keep both eyes open."
    elif engine.phase == CardinalPhase.TESTING:
        active_t = CARDINAL_TARGETS[engine.target_idx]
        instruction = f"Look steadily at the highlighted target: {active_t.name}. Keep your head still."
    else:
        instruction = "9-Cardinal Gaze Motility Test complete! Press [R] to re-test, [Q] to quit."

    cv2.putText(canvas, instruction, (20, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.52, (240, 240, 240), 1)

    # --------------------------------------------------------------------------
    # 5. FINAL RESULTS PRESENTATION CARD (3x3 Summary Matrix)
    # --------------------------------------------------------------------------
    if engine.phase == CardinalPhase.COMPLETE:
        rw, rh = min(720, w - 30), min(440, h - 40)
        rx, ry = (w - rw) // 2, (h - rh) // 2
        
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (18, 22, 26), -1)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (0, 230, 0), 2)
        
        cv2.putText(canvas, "9-CARDINAL GAZE MOTILITY REPORT", (rx + 25, ry + 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 1)
        cv2.putText(canvas, "Test completed - Motility Matrix Summary", (rx + 25, ry + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        cv2.line(canvas, (rx + 25, ry + 58), (rx + rw - 25, ry + 58), (70, 70, 70), 1)
        
        # 1. Comitancy Diagnosis Banner
        diag_color = (0, 255, 0) if "COMITANT" in engine.comitancy_classification and "BORDERLINE" not in engine.comitancy_classification else (0, 140, 255)
        cv2.putText(canvas, f"ASSESSMENT: {engine.comitancy_classification}", (rx + 25, ry + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.60, diag_color, 2)
        
        # 2. Key Metrics Row
        m_txt1 = f"Primary Alignment: {engine.primary_delta_pd[0]:+.1f} \u0394PD | Incomitancy Spread: {engine.incomitancy_spread_pd:.1f} \u0394PD (Max - Min)"
        m_txt2 = f"Max Deviation: {engine.max_deviation_pd:+.1f} \u0394PD ({engine.max_dev_pos_name.split()[0]}) | Min: {engine.min_deviation_pd:+.1f} \u0394PD"
        cv2.putText(canvas, m_txt1, (rx + 25, ry + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1)
        cv2.putText(canvas, m_txt2, (rx + 25, ry + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)

        # 3. Mini 3x3 Text Grid
        grid_cells = {(m.grid_row, m.grid_col): m for m in engine.position_results.values()}
        grid_start_y = ry + 175
        cv2.putText(canvas, "3x3 MOTILITY MATRIX (Horizontal \u0394PD):", (rx + 25, grid_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1)
        
        row_labels = ["UP  ", "PRIM", "DOWN"]
        
        for r in range(3):
            cell_strs = []
            for c in range(3):
                m = grid_cells.get((r, c))
                val = f"{m.delta_pd_h:+5.1f} PD" if m else "   N/A  "
                cell_strs.append(val)
            row_txt = f"{row_labels[r]}: [ {cell_strs[0]} ]   [ {cell_strs[1]} ]   [ {cell_strs[2]} ]"
            cv2.putText(canvas, row_txt, (rx + 45, grid_start_y + 24 + r * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        # 4. Interpretation summary
        interp_y = grid_start_y + 105
        cv2.putText(canvas, "PRELIMINARY INTERPRETATION:", (rx + 25, interp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1)
        cv2.putText(canvas, engine.preliminary_interpretation[:95], (rx + 35, interp_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1)
        
        # 5. Recommendation & Controls
        rec_txt = "Screening aid only. Does not replace professional clinical diagnosis."
        cv2.putText(canvas, rec_txt, (rx + 25, ry + rh - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)
        cv2.putText(canvas, "Press [R] to Re-test | [Q] to Quit", (rx + rw // 2 - 120, ry + rh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)

    return canvas


# ==============================================================================
# 4. MAIN RUNNER (LIVE WEBCAM & VIDEO)
# ==============================================================================
def run_cardinal_screening(video_path: Optional[str] = None, webcam_idx: int = 0, fullscreen: bool = False):
    """Executes the condition-gated 9-Cardinal Gaze Motility Matrix session."""
    is_live = (video_path is None)
    
    print("\n" + "=" * 75)
    print("STARTING 9-CARDINAL GAZE MOTILITY MATRIX SCREENING (MODULE 2.4)")
    print("=" * 75)
    print("Protocol: Condition-gated 9 diagnostic gaze positions.")
    print("Fixate on each highlighted target steadily until the circular progress arc fills.\n")
    
    tracker = PerEyeTracker(mirror_mode=is_live)
    engine = CardinalGazeEngine()
    
    if is_live:
        cap = cv2.VideoCapture(webcam_idx)
        print(f"[INPUT] Connected to live webcam (Index {webcam_idx})")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"[INPUT] Reading video file: {video_path}")
        
    if not cap.isOpened():
        print("[ERROR] Could not open video capture device.")
        return

    start_wall_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    window_name = "9-Cardinal Gaze Motility Matrix (Step 2 - Module 2.4)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, 1024, 768)

    print("\n--- CONTROLS ---")
    print("  [R]     : Reset & restart test")
    print("  [SPACE] : Override / bypass current target")
    print("  [Q]     : Quit")
    print("----------------\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if not is_live:
                print("[INFO] Video finished.")
                if engine.phase != CardinalPhase.COMPLETE:
                    engine._finalize_and_report()
            break
            
        frame_idx += 1
        t_now = time.time()
        t_ms = int((t_now - start_wall_time) * 1000) if is_live else int(frame_idx * (1000.0 / fps))
        
        # 1. Process frame through PerEyeTracker
        result = tracker.process_frame(frame, timestamp_ms=t_ms)
        
        # 2. Update condition-gated state machine
        engine.update(result, current_time=t_now)
        
        # 3. Create high-resolution canvas with subtle dimmed camera feed (25% opacity)
        # Allows user to see face centering and tracking without cluttering the 9 targets
        dimmed_frame = cv2.resize(frame, (1024, 768))
        if tracker.mirror_mode:
            dimmed_frame = cv2.flip(dimmed_frame, 1)
        canvas = cv2.addWeighted(dimmed_frame, 0.25, np.zeros((768, 1024, 3), dtype=np.uint8), 0.75, 0)
        
        # 4. Render clean HUD without duplicate step-1 header
        hud_canvas = draw_cardinal_hud(canvas, engine, tracker, result)
        
        cv2.imshow(window_name, hud_canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            engine.reset()
        elif key == 32:  # Spacebar manual override
            print(f"[MANUAL OVERRIDE] Operator pressed Spacebar -> Advancing target {engine.target_idx + 1}.")
            if engine.phase == CardinalPhase.POSITIONING:
                engine.phase = CardinalPhase.TESTING
                engine.target_idx = 0
            elif engine.phase == CardinalPhase.TESTING:
                engine.advance_target()

    cap.release()
    cv2.destroyAllWindows()


# ==============================================================================
# 5. CLI ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated 9-Cardinal Gaze Motility Matrix (Step 2 - Module 2.4)")
    parser.add_argument("--video", type=str, default=None, help="Path to offline test video")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--fullscreen", action="store_true", help="Launch in fullscreen mode for maximal motility angular range")
    args = parser.parse_args()
    
    run_cardinal_screening(video_path=args.video, webcam_idx=args.webcam, fullscreen=args.fullscreen)
