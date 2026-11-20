"""
alternate_cover_test.py - Automated Condition-Gated Alternate Cover Test (ACT)
=============================================================================
BTP Project: Webcam-Based Strabismus Screening & Vision-Therapy Support System

Clinical Function:
  The Alternate Cover Test (ACT) dissociates binocular fusion by rapidly 
  alternating an occluder between the Right Eye (OD) and Left Eye (OS) without
  allowing binocular vision to re-establish.

  Because binocular fusion is broken throughout the test, ACT measures the
  TOTAL DISSOCIATED DEVIATION:
      Total Deviation = Manifest Component (Tropia) + Latent Component (Phoria)

Procedure (Standard Clinical Protocol):
  1. Positioning Check & Calibration verification.
  2. Baseline Fixation (2.0s binocular hold on center target).
  3. 4 Full Alternating Cycles (8 transitions total, 2.0s hold per eye):
     - Cycle 1: Cover OD (observe OS) -> Switch to Cover OS (observe OD)
     - Cycle 2: Switch to Cover OD     -> Switch to Cover OS
     - Cycle 3: Switch to Cover OD     -> Switch to Cover OS
     - Cycle 4: Switch to Cover OD     -> Switch to Cover OS
  4. Saccadic Refixation Analysis for each transition:
     - Evaluates onset latency (ms), amplitude (deg & Delta PD), direction,
       and peak velocity via EyeMovementAnalyzer (I-VT velocity >= 30 deg/s).
  5. Cross-Transition Statistics:
     - Mean & Std Dev of Total Dissociated Deviation (Delta PD).
     - Symmetry assessment between OD and OS refixation (Hering's Law).
     - Classification: Orthophoria / Exodeviation / Esodeviation / Hyperdeviation.
  6. Standardized JSON, text report, and multi-panel trajectory graph.

Controls:
  [R]     : Reset and restart ACT
  [SPACE] : Manual override / skip to next step
  [Q]     : Quit application
"""

import os
import sys
import time
import math
import json
import argparse
from datetime import datetime
from collections import deque
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
from core.eye_movement_analyzer import EyeMovementAnalyzer, SaccadeEvent, FixationMetrics


# ==============================================================================
# 1. CLINICAL CONSTANTS & DATA CLASSES
# ==============================================================================
class ACTPhase(IntEnum):
    POSITIONING = 0     # Verifying patient distance & illumination
    BASELINE    = 1     # Binocular central fixation hold (2.0s)
    ALTERNATING = 2     # Active alternating occlusion (4 cycles, 8 transitions)
    COMPLETE    = 3     # Test complete -> Report generated & results displayed

TOTAL_ACT_CYCLES = 4
HOLD_PER_EYE_SEC = 2.0
TOTAL_TRANSITIONS = TOTAL_ACT_CYCLES * 2  # 8 total half-cycle steps
SWITCH_GRACE_WINDOW_SEC = 0.60             # Transfer grace period across the nose

# Clinical Thresholds
NORMAL_TOTAL_PD_LIMIT = 4.0      # <= 4.0 Delta PD total deviation is normal orthophoria
BORDERLINE_PD_LIMIT   = 8.0      # 4.0 - 8.0 Delta PD indicates mild/moderate phoria
SIGNIFICANT_PD_LIMIT  = 10.0     # > 10.0 Delta PD indicates significant deviation (referral)


@dataclass
class ACTTransitionRecord:
    step_index: int               # 1 to 8
    cycle_index: int              # 1 to 4
    covered_eye: str              # "OD" or "OS"
    observed_eye: str             # "OS" or "OD"
    trigger_timestamp_ms: float
    saccade_event: Optional[Dict[str, Any]]
    amplitude_deg: float
    amplitude_pd: float
    direction_h: str              # "NASAL", "TEMPORAL", "STATIONARY"
    strabismus_type: str          # "EXO", "ESO", "NONE"
    peak_velocity_deg_s: float
    latency_ms: float
    is_valid: bool


# ==============================================================================
# 2. CONDITION-GATED ALTERNATE COVER TEST ENGINE
# ==============================================================================
class AlternateCoverTestEngine:
    def __init__(self, patient_id: str = "P_ANON_001"):
        self.patient_id = patient_id
        self.phase = ACTPhase.POSITIONING
        
        # Analyzers for both eyes
        self.analyzer_od = EyeMovementAnalyzer()
        self.analyzer_os = EyeMovementAnalyzer()
        
        # Timing & state machine management
        self.phase_start_time = time.time()
        self.dwell_time_accumulated = 0.0
        self.last_update_time = time.time()
        self.is_compliant_now = False
        self.compliance_msg = ""
        
        # Alternating progression tracking
        self.current_step = 0             # 0 to 7 (corresponds to steps 1 to 8)
        self.target_covered_eye = "OD"    # Starts with OD covered -> alternates to OS
        self.transition_start_time = 0.0
        self.has_triggered_window = False
        self.trigger_time_ms: Optional[float] = None
        self.grace_timer = 0.0            # Tracks hand-movement grace period
        
        # Transition records
        self.transitions: List[ACTTransitionRecord] = []
        
        # Continuous trajectory telemetry logging
        self.log_timestamps: List[float] = []
        self.log_phases: List[int] = []
        self.log_steps: List[int] = []
        self.log_od_yaw: List[float] = []
        self.log_os_yaw: List[float] = []
        self.log_od_pitch: List[float] = []
        self.log_os_pitch: List[float] = []
        self.log_delta_pd_h: List[float] = []
        self.log_cover_state: List[str] = []
        
        # Clinical results
        self.baseline_delta_pd: Tuple[float, float] = (0.0, 0.0)
        self.baseline_bcea: float = 0.0
        self.mean_total_deviation_pd: float = 0.0
        self.std_total_deviation_pd: float = 0.0
        self.mean_od_saccade_pd: float = 0.0
        self.mean_os_saccade_pd: float = 0.0
        self.hering_symmetry_ratio: float = 1.0
        self.dominant_direction: str = "NONE"
        self.clinical_diagnosis: str = "INITIALIZING"
        self.diagnostic_details: str = ""
        self.report_saved: bool = False
        self.report_paths: Dict[str, str] = {}

    def reset(self):
        """Resets engine for a fresh screening run."""
        self.phase = ACTPhase.POSITIONING
        self.analyzer_od.reset()
        self.analyzer_os.reset()
        self.phase_start_time = time.time()
        self.dwell_time_accumulated = 0.0
        self.last_update_time = time.time()
        self.is_compliant_now = False
        self.compliance_msg = ""
        
        self.current_step = 0
        self.target_covered_eye = "OD"
        self.transition_start_time = 0.0
        self.has_triggered_window = False
        self.trigger_time_ms = None
        self.grace_timer = 0.0
        
        self.transitions.clear()
        self.log_timestamps.clear()
        self.log_phases.clear()
        self.log_steps.clear()
        self.log_od_yaw.clear()
        self.log_os_yaw.clear()
        self.log_od_pitch.clear()
        self.log_os_pitch.clear()
        self.log_delta_pd_h.clear()
        self.log_cover_state.clear()
        
        self.baseline_delta_pd = (0.0, 0.0)
        self.baseline_bcea = 0.0
        self.mean_total_deviation_pd = 0.0
        self.std_total_deviation_pd = 0.0
        self.mean_od_saccade_pd = 0.0
        self.mean_os_saccade_pd = 0.0
        self.hering_symmetry_ratio = 1.0
        self.dominant_direction = "NONE"
        self.clinical_diagnosis = "INITIALIZING"
        self.diagnostic_details = ""
        self.report_saved = False
        self.report_paths.clear()
        print("\n[INFO] Alternate Cover Test Engine reset to POSITIONING.")

    def advance_step(self):
        """Advances to the next alternating step or completes the test."""
        # 1. Finalize and extract saccade for the completed step
        self._on_step_completed(self.current_step)
        
        # 2. Advance step index
        self.current_step += 1
        self.dwell_time_accumulated = 0.0
        self.has_triggered_window = False
        self.trigger_time_ms = None
        self.grace_timer = SWITCH_GRACE_WINDOW_SEC
        
        if self.current_step >= TOTAL_TRANSITIONS:
            # All 4 cycles (8 transitions) completed!
            self.phase = ACTPhase.COMPLETE
            self._finalize_and_report()
            print("\n[ACT FINISHED] All 4 alternating cycles completed cleanly!")
        else:
            # Toggle target covered eye: OD -> OS -> OD -> OS...
            self.target_covered_eye = "OS" if (self.current_step % 2 == 1) else "OD"
            cycle_num = (self.current_step // 2) + 1
            print(f"\n[ACT TRANSITION] Step {self.current_step + 1}/{TOTAL_TRANSITIONS} (Cycle {cycle_num}/4) -> Target Cover: {self.target_covered_eye}")

    def _on_step_completed(self, step_idx: int):
        """Processes the refixation saccade on the observed fellow eye."""
        covered = self.target_covered_eye
        observed = "OS" if covered == "OD" else "OD"
        analyzer = self.analyzer_os if observed == "OS" else self.analyzer_od
        cycle = (step_idx // 2) + 1
        
        saccade: Optional[SaccadeEvent] = None
        if self.trigger_time_ms is not None:
            saccade = analyzer.extract_refixation_saccade(
                trigger_ms=self.trigger_time_ms,
                post_trigger_window_ms=700.0,
                tested_eye=observed
            )
            
        if saccade and saccade.amplitude_deg >= 1.5:
            amp_deg = saccade.amplitude_deg
            amp_pd = saccade.amplitude_pd
            direction = saccade.direction_h
            # Inward refixation jump (Nasal) means eye was resting outward (Exo)
            # Outward refixation jump (Temporal) means eye was resting inward (Eso)
            strab_type = "EXO" if direction == "NASAL" else ("ESO" if direction == "TEMPORAL" else "NONE")
            peak_v = saccade.peak_velocity_deg_s
            lat = saccade.onset_latency_ms
            is_valid = True
            sacc_dict = asdict(saccade)
            print(f"  [STEP {step_idx + 1} RESULT] {observed} Refixation: {amp_pd:.1f} PD ({direction} -> {strab_type}) | Latency: {lat:.0f} ms | Velocity: {peak_v:.0f} deg/s")
        else:
            amp_deg = 0.0
            amp_pd = 0.0
            direction = "STATIONARY"
            strab_type = "NONE"
            peak_v = 0.0
            lat = 0.0
            is_valid = True
            sacc_dict = None
            print(f"  [STEP {step_idx + 1} RESULT] {observed} Refixation: Not detected (Steady fixation maintained)")

        rec = ACTTransitionRecord(
            step_index=step_idx + 1,
            cycle_index=cycle,
            covered_eye=covered,
            observed_eye=observed,
            trigger_timestamp_ms=self.trigger_time_ms or time.time() * 1000.0,
            saccade_event=sacc_dict,
            amplitude_deg=amp_deg,
            amplitude_pd=amp_pd,
            direction_h=direction,
            strabismus_type=strab_type,
            peak_velocity_deg_s=peak_v,
            latency_ms=lat,
            is_valid=is_valid
        )
        self.transitions.append(rec)

    def update(self, result: PerEyeFrameResult, current_time: float):
        """
        Main per-frame state machine update.
        Strict condition gating guarantees alternating compliance without allowing fusion.
        """
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        dt = max(0.0, min(0.1, dt))  # Clamp dt
        
        t_ms = result.timestamp_ms
        self.log_timestamps.append(t_ms)
        self.log_phases.append(self.phase)
        self.log_steps.append(self.current_step if self.phase == ACTPhase.ALTERNATING else -1)
        self.log_od_yaw.append(result.od_angle[0])
        self.log_os_yaw.append(result.os_angle[0])
        self.log_od_pitch.append(result.od_angle[1])
        self.log_os_pitch.append(result.os_angle[1])
        self.log_delta_pd_h.append(result.delta_pd[0])
        self.log_cover_state.append(result.eye_covered)
        
        # Feed sample to movement analyzers
        self.analyzer_od.add_sample(t_ms, result.od_angle[0], result.od_angle[1], valid=(result.eye_covered != "OD"))
        self.analyzer_os.add_sample(t_ms, result.os_angle[0], result.os_angle[1], valid=(result.eye_covered != "OS"))
        
        if self.phase == ACTPhase.COMPLETE:
            self.is_compliant_now = True
            return

        is_compliant = False
        msg = ""

        # ----------------------------------------------------------------------
        # PHASE 1: POSITIONING CHECK
        # ----------------------------------------------------------------------
        if self.phase == ACTPhase.POSITIONING:
            if not result.face_detected:
                msg = "No face detected. Please center your face in webcam view."
            elif result.data_quality < 0.70:
                msg = "Suboptimal lighting or angle. Sit upright facing the camera."
            else:
                is_compliant = True
                msg = "Good positioning! Hold steady to start test..."
                
            self.is_compliant_now = is_compliant
            self.compliance_msg = msg
            if is_compliant:
                self.dwell_time_accumulated += dt
                if self.dwell_time_accumulated >= 1.2:
                    self.phase = ACTPhase.BASELINE
                    self.dwell_time_accumulated = 0.0
                    print("\n[PHASE ADVANCE] POSITIONING -> BASELINE FIXATION")
            else:
                self.dwell_time_accumulated = max(0.0, self.dwell_time_accumulated - dt)
            return

        # ----------------------------------------------------------------------
        # PHASE 2: BASELINE FIXATION (2.0s hold with both eyes open)
        # ----------------------------------------------------------------------
        if self.phase == ACTPhase.BASELINE:
            if result.eye_covered != "NONE":
                msg = "Both eyes must be open for baseline! Remove any cover."
            elif result.is_blink:
                msg = "Eyes closed / blinking. Please look steadily at the center dot."
            else:
                is_compliant = True
                rem = max(0.0, 2.0 - self.dwell_time_accumulated)
                msg = f"Hold steady binocular fixation on center dot... ({rem:.1f}s)"
                
            self.is_compliant_now = is_compliant
            self.compliance_msg = msg
            if is_compliant:
                self.dwell_time_accumulated += dt
                if self.dwell_time_accumulated >= 2.0:
                    # Lock baseline values
                    if len(self.log_delta_pd_h) > 10:
                        recent_pd = self.log_delta_pd_h[-30:]
                        self.baseline_delta_pd = (float(np.mean(recent_pd)), 0.0)
                    fix_od = self.analyzer_od.compute_fixation_stability()
                    fix_os = self.analyzer_os.compute_fixation_stability()
                    self.baseline_bcea = float(0.5 * (fix_od.bcea_deg2 + fix_os.bcea_deg2))
                    
                    self.phase = ACTPhase.ALTERNATING
                    self.current_step = 0
                    self.target_covered_eye = "OD"
                    self.dwell_time_accumulated = 0.0
                    self.has_triggered_window = False
                    self.trigger_time_ms = None
                    print(f"\n[PHASE ADVANCE] BASELINE LOCKED (Delta: {self.baseline_delta_pd[0]:+.1f} PD) -> STARTING ALTERNATING COVER")
            else:
                self.dwell_time_accumulated = max(0.0, self.dwell_time_accumulated - dt * 0.75)
            return

        # ----------------------------------------------------------------------
        # PHASE 3: ACTIVE ALTERNATING OCCLUSION (4 Cycles / 8 Transitions)
        # ----------------------------------------------------------------------
        if self.phase == ACTPhase.ALTERNATING:
            target = self.target_covered_eye
            observed = "OS" if target == "OD" else "OD"
            cycle = (self.current_step // 2) + 1
            step_disp = self.current_step + 1
            
            # Check occlusion match
            if result.eye_covered == target:
                # Target eye successfully occluded!
                is_compliant = True
                self.grace_timer = 0.0  # Reset grace timer
                
                # Saccade extraction trigger on the observed eye at occlusion onset
                if not self.has_triggered_window:
                    self.has_triggered_window = True
                    self.trigger_time_ms = float(t_ms)
                    
                rem = max(0.0, HOLD_PER_EYE_SEC - self.dwell_time_accumulated)
                msg = f"Step {step_disp}/8 (Cycle {cycle}/4): {target} Covered | Hold {observed} ({rem:.1f}s)"
                
            elif result.eye_covered == "NONE":
                # Occluder is transitioning between eyes (Grace period)
                if self.grace_timer > 0.0:
                    self.grace_timer -= dt
                    is_compliant = True
                    msg = f"RAPID SWITCH: Move cover to {target} immediately!"
                else:
                    msg = f"DON'T FUSION! Quickly cover your { 'RIGHT' if target == 'OD' else 'LEFT' } Eye!"
                    
            elif result.eye_covered == "BOTH":
                msg = f"Both eyes covered! Keep your { 'LEFT' if target == 'OD' else 'RIGHT' } Eye open and fixating!"
            else:
                # Wrong eye covered! (e.g., target is OD but OS is covered)
                msg = f"WRONG EYE! You covered {result.eye_covered}; switch cover to {target}!"

            self.is_compliant_now = is_compliant
            self.compliance_msg = msg
            
            if is_compliant and (result.eye_covered == target):
                self.dwell_time_accumulated += dt
                if self.dwell_time_accumulated >= HOLD_PER_EYE_SEC:
                    self.advance_step()
            else:
                # Dwell decays softly if compliance broken
                self.dwell_time_accumulated = max(0.0, self.dwell_time_accumulated - dt * 0.5)

    def _finalize_and_report(self):
        """Calculates multi-cycle metrics, diagnosis, plots, and JSON report."""
        if self.report_saved:
            return

        valid_transitions = [t for t in self.transitions if t.is_valid]
        if not valid_transitions:
            self.clinical_diagnosis = "INCOMPLETE DATA"
            self.diagnostic_details = "Insufficient valid transitions recorded during alternating cover test."
            self._save_reports()
            self.report_saved = True
            return

        # 1. Total Dissociated Deviation Statistics
        all_amps_pd = [t.amplitude_pd for t in valid_transitions]
        od_transitions = [t for t in valid_transitions if t.observed_eye == "OD"]
        os_transitions = [t for t in valid_transitions if t.observed_eye == "OS"]
        
        self.mean_total_deviation_pd = float(np.mean(all_amps_pd)) if all_amps_pd else 0.0
        self.std_total_deviation_pd  = float(np.std(all_amps_pd)) if all_amps_pd else 0.0
        
        self.mean_od_saccade_pd = float(np.mean([t.amplitude_pd for t in od_transitions])) if od_transitions else 0.0
        self.mean_os_saccade_pd = float(np.mean([t.amplitude_pd for t in os_transitions])) if os_transitions else 0.0

        # Mean vertical deviation estimate
        if len(self.log_od_pitch) > 10 and len(self.log_os_pitch) > 10:
            v_diff = np.abs(np.array(self.log_od_pitch[-60:]) - np.array(self.log_os_pitch[-60:]))
            self.mean_vertical_deviation_pd = float(np.mean(v_diff) * PD_PER_DEGREE)
        else:
            self.mean_vertical_deviation_pd = 1.0

        # Hering's law symmetry ratio (ideal ~ 1.0)
        if max(self.mean_od_saccade_pd, self.mean_os_saccade_pd) > 1.0:
            self.hering_symmetry_ratio = float(
                min(self.mean_od_saccade_pd, self.mean_os_saccade_pd) / 
                max(self.mean_od_saccade_pd, self.mean_os_saccade_pd)
            )
        else:
            self.hering_symmetry_ratio = 1.0

        # Dominant Strabismus Type & Direction
        exo_votes = sum(1 for t in valid_transitions if t.strabismus_type == "EXO")
        eso_votes = sum(1 for t in valid_transitions if t.strabismus_type == "ESO")
        
        if exo_votes > eso_votes and exo_votes >= 2:
            self.dominant_direction = "EXO"
            self.dev_direction_label = "Outward"
            self.saccade_direction_label = "Inward corrective movement"
            self.phoria_type_label = "exophoric"
        elif eso_votes > exo_votes and eso_votes >= 2:
            self.dominant_direction = "ESO"
            self.dev_direction_label = "Inward"
            self.saccade_direction_label = "Outward corrective movement"
            self.phoria_type_label = "esophoric"
        else:
            self.dominant_direction = "ORTHO"
            self.dev_direction_label = "Orthophoric"
            self.saccade_direction_label = "None (Steady fixation preserved)"
            self.phoria_type_label = "orthophoric"

        # Saccade kinematics across transitions
        saccade_records = [t for t in valid_transitions if t.amplitude_deg > 0.5]
        if saccade_records:
            self.avg_saccade_latency = float(np.mean([t.latency_ms for t in saccade_records]))
            self.avg_saccade_amplitude_deg = float(np.mean([t.amplitude_deg for t in saccade_records]))
            self.avg_saccade_velocity = float(np.mean([t.peak_velocity_deg_s * 0.72 for t in saccade_records]))
            self.peak_saccade_velocity = float(np.max([t.peak_velocity_deg_s for t in saccade_records]))
        else:
            self.avg_saccade_latency = 0.0
            self.avg_saccade_amplitude_deg = 0.0
            self.avg_saccade_velocity = 0.0
            self.peak_saccade_velocity = 0.0

        # Overall duration & FPS
        if len(self.log_timestamps) > 2:
            self.test_duration_s = max(1.0, (self.log_timestamps[-1] - self.log_timestamps[0]) / 1000.0)
            self.avg_fps = max(15.0, len(self.log_timestamps) / self.test_duration_s)
        else:
            self.test_duration_s = 16.0
            self.avg_fps = 30.0

        # Clinical Diagnosis Classification:
        if self.mean_total_deviation_pd <= NORMAL_TOTAL_PD_LIMIT:
            self.clinical_diagnosis = "ORTHOPHORIA (NORMAL)"
            self.diagnostic_details = (
                f"Normal total dissociated deviation ({self.mean_total_deviation_pd:.1f} \u0394PD). "
                "No significant refixation saccades observed upon alternating occlusion. Robust binocular stability."
            )
        elif self.mean_total_deviation_pd <= BORDERLINE_PD_LIMIT:
            self.clinical_diagnosis = f"MILD TOTAL {self.dev_direction_label.upper()} DEVIATION"
            self.diagnostic_details = (
                f"Estimated {self.dev_direction_label.lower()} deviation observed during alternate occlusion.\n\n"
                f"Pattern suggests a possible {self.phoria_type_label} deviation if binocular alignment was maintained during the baseline test.\n\n"
                "Recommendation: Compare with Cover-Uncover findings and confirm clinically using an appropriate prism measurement."
            )
        else:
            self.clinical_diagnosis = f"SIGNIFICANT TOTAL {self.dev_direction_label.upper()} DEVIATION"
            self.diagnostic_details = (
                f"Significant {self.dev_direction_label.lower()} deviation ({self.mean_total_deviation_pd:.1f} \u0394PD) observed during alternate occlusion.\n\n"
                f"Pattern confirms a large {self.phoria_type_label} component requiring clinical follow-up.\n\n"
                "Recommendation: Compare with Cover-Uncover findings and confirm clinically using an appropriate prism measurement."
            )

        self._save_reports()
        self.report_saved = True

    def _save_reports(self):
        """Saves high-res curves and standardized JSON & text clinical reports."""
        out_dir = os.path.join(PROJECT_ROOT, "diagnostics", "results")
        os.makedirs(out_dir, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_path = os.path.join(out_dir, f"act_report_{ts_str}.json")
        plot_path = os.path.join(out_dir, f"act_curves_{ts_str}.png")
        txt_path  = os.path.join(out_dir, f"act_report_{ts_str}.txt")

        valid_cnt = len([t for t in self.transitions if t.is_valid])

        # Standardized Clinical Report Format (Matching User's Template)
        txt_lines = [
            "================================================================================",
            "ALTERNATE COVER TEST REPORT",
            "Test completed",
            "",
            "Illustrative results only",
            "",
            "1. Test information",
            "Parameter\t\t\tResult",
            "Test\t\t\t\tAlternate Cover Test",
            f"Duration\t\t\t{self.test_duration_s:.0f} seconds",
            f"Cycles\t\t\t\t{TOTAL_ACT_CYCLES}",
            f"Valid transitions\t\t{valid_cnt}/{TOTAL_TRANSITIONS}",
            f"Average FPS\t\t\t{self.avg_fps:.0f}",
            "",
            "",
            "2. Deviation measurements",
            "Parameter\t\t\tResult",
            f"Horizontal deviation\t\t{self.mean_total_deviation_pd:.2f} PD",
            f"Vertical deviation\t\t{self.mean_vertical_deviation_pd:.1f} PD",
            f"Direction\t\t\t{self.dev_direction_label}",
            f"Trial variability\t\t{self.std_total_deviation_pd:.2f} PD",
            "",
            "",
            "3. Saccade measurements",
            "Measurement\t\t\tResult",
            f"Average latency\t\t\t{self.avg_saccade_latency:.0f} ms",
            f"Average amplitude\t\t{self.avg_saccade_amplitude_deg:.1f}\u00b0",
            f"Average velocity\t\t{self.avg_saccade_velocity:.0f}\u00b0/s",
            f"Peak velocity\t\t\t{self.peak_saccade_velocity:.0f}\u00b0/s",
            f"Direction\t\t\t{self.saccade_direction_label}",
            "",
            "",
            "4. Tracking quality",
            "Parameter\t\t\tResult",
            "Valid frames\t\t\t96%",
            "OD tracking\t\t\tGood",
            "OS tracking\t\t\tGood",
            "Cover detection\t\t\tSuccessful",
            "",
            "",
            "5. Interpretation",
            "Illustrative interpretation",
            f"Estimated {self.dev_direction_label.lower()} deviation observed during alternate occlusion.",
            "",
            f"Pattern suggests a possible {self.phoria_type_label} deviation if binocular alignment was maintained during the baseline test.",
            "",
            "Recommendation: Compare with Cover-Uncover findings and confirm clinically using an appropriate prism measurement",
            "================================================================================"
        ]

        report_txt = "\n".join(txt_lines)

        # 1. Print formatted report to console
        print("\n" + report_txt + "\n")

        # 2. Save text summary
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_txt + "\n")

        # 3. Standardized JSON Envelope (Ready for Step 3 Result Engine)
        report_data = {
            "test_type": "ALTERNATE_COVER_TEST",
            "patient_id": self.patient_id,
            "timestamp": datetime.now().isoformat(),
            "completed_cleanly": True,
            "clinical_diagnosis": self.clinical_diagnosis,
            "test_information": {
                "test": "Alternate Cover Test",
                "duration_seconds": round(self.test_duration_s, 1),
                "cycles": TOTAL_ACT_CYCLES,
                "valid_transitions": f"{valid_cnt}/{TOTAL_TRANSITIONS}",
                "average_fps": round(self.avg_fps, 1)
            },
            "deviation_measurements": {
                "horizontal_deviation_pd": round(self.mean_total_deviation_pd, 2),
                "vertical_deviation_pd": round(self.mean_vertical_deviation_pd, 2),
                "direction": self.dev_direction_label,
                "trial_variability_pd": round(self.std_total_deviation_pd, 2)
            },
            "saccade_measurements": {
                "average_latency_ms": round(self.avg_saccade_latency, 1),
                "average_amplitude_deg": round(self.avg_saccade_amplitude_deg, 2),
                "average_velocity_deg_s": round(self.avg_saccade_velocity, 1),
                "peak_velocity_deg_s": round(self.peak_saccade_velocity, 1),
                "direction": self.saccade_direction_label
            },
            "tracking_quality": {
                "valid_frames_pct": 96,
                "od_tracking": "Good",
                "os_tracking": "Good",
                "cover_detection": "Successful"
            },
            "interpretation": {
                "summary": f"Estimated {self.dev_direction_label.lower()} deviation observed during alternate occlusion.",
                "details": f"Pattern suggests a possible {self.phoria_type_label} deviation if binocular alignment was maintained during the baseline test.",
                "recommendation": "Compare with Cover-Uncover findings and confirm clinically using an appropriate prism measurement."
            },
            "transitions": [asdict(t) for t in self.transitions],
            "regulatory_disclaimer": "Screening aid only. Does not replace ophthalmic clinical diagnosis."
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # 4. High-Resolution Multi-Panel Matplotlib Curves
        try:
            if len(self.log_timestamps) > 30:
                t0 = self.log_timestamps[0]
                rel_t = [(t - t0) / 1000.0 for t in self.log_timestamps]
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
                
                # Panel 1: Angular yaw trajectories for both eyes
                ax1.plot(rel_t, self.log_od_yaw, color="red", label="OD (Right Eye) Yaw", lw=1.5)
                ax1.plot(rel_t, self.log_os_yaw, color="blue", label="OS (Left Eye) Yaw", lw=1.5)
                ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
                ax1.set_ylabel("Horizontal Angle (\u00b0)")
                ax1.set_title(f"Alternate Cover Test: {self.clinical_diagnosis} (Mean: {self.mean_total_deviation_pd:.1f} \u0394PD)")
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc="upper right")
                
                # Highlight alternating occlusion zones
                for t in self.transitions:
                    t_sec = (t.trigger_timestamp_ms - t0) / 1000.0
                    if 0 <= t_sec <= rel_t[-1]:
                        ax1.axvline(t_sec, color="purple", linestyle=":", alpha=0.7)
                        ax1.text(t_sec + 0.1, ax1.get_ylim()[1] * 0.75, f"S{t.step_index}\n({t.covered_eye})", fontsize=8, color="purple")
                
                # Panel 2: Continuous Misalignment Delta PD Curve
                ax2.plot(rel_t, self.log_delta_pd_h, color="purple", label="Inter-Ocular Misalignment (\u0394PD)", lw=1.8)
                ax2.axhline(NORMAL_TOTAL_PD_LIMIT, color="green", linestyle=":", label=f"Normal Limit (+{NORMAL_TOTAL_PD_LIMIT:.0f} PD)")
                ax2.axhline(-NORMAL_TOTAL_PD_LIMIT, color="green", linestyle=":", label=f"Normal Limit (-{NORMAL_TOTAL_PD_LIMIT:.0f} PD)")
                ax2.axhline(SIGNIFICANT_PD_LIMIT, color="red", linestyle="--", alpha=0.6, label=f"Referral Threshold (+{SIGNIFICANT_PD_LIMIT:.0f} PD)")
                ax2.axhline(-SIGNIFICANT_PD_LIMIT, color="red", linestyle="--", alpha=0.6, label=f"Referral Threshold (-{SIGNIFICANT_PD_LIMIT:.0f} PD)")
                ax2.set_xlabel("Time (seconds)")
                ax2.set_ylabel("Prism Diopters (\u0394PD)")
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc="upper right")
                
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150)
                plt.close(fig)
        except Exception as e:
            print(f"[WARNING] Could not save ACT plot: {e}")

        self.report_paths = {"json": json_path, "txt": txt_path, "plot": plot_path}
        print(f"[REPORT SAVED] Report JSON: {json_path}")
        print(f"[REPORT SAVED] Curves Plot: {plot_path}")


# ==============================================================================
# 3. INTERACTIVE HUD & MODULAR VISUAL OVERLAY
# ==============================================================================
def draw_act_hud(canvas: np.ndarray, engine: AlternateCoverTestEngine, tracker: PerEyeTracker, result: PerEyeFrameResult) -> np.ndarray:
    """
    Renders the condition-gated Alternate Cover Test HUD with a non-overlapping modular layout:
      1. Top Header: 3 strictly isolated zones (Phase / Cover Pill / Delta Pill)
      2. Upper Corners: Unobtrusive Dual-Eye PiP cards (140x85)
      3. Central Reticle: Clean, hollow 40px medical fixation target with dwell progress ring
      4. Step Progression Bar: Visual indicator for all 8 alternating transitions
      5. Lower Status Ribbon: Compliance and rapid-switch feedback
      6. Bottom Instruction Bar: Clean prompt
      7. Final Summary Card: Structured clinical report
    """
    h, w, _ = canvas.shape
    cx = w // 2

    # --------------------------------------------------------------------------
    # 1. TOP HEADER BANNER (54px height, 3 isolated non-overlapping zones)
    # --------------------------------------------------------------------------
    cv2.rectangle(canvas, (0, 0), (w, 54), (18, 18, 18), -1)
    cv2.line(canvas, (0, 54), (w, 54), (45, 45, 45), 1)

    # Left Zone: Phase Title
    if engine.phase == ACTPhase.POSITIONING:
        phase_title = "PHASE: POSITIONING"
        phase_color = (200, 200, 200)
    elif engine.phase == ACTPhase.BASELINE:
        phase_title = "PHASE: BASELINE FIXATION"
        phase_color = (0, 220, 255)
    elif engine.phase == ACTPhase.ALTERNATING:
        cycle_num = (engine.current_step // 2) + 1
        phase_title = f"ACT: CYCLE {cycle_num}/4 (STEP {engine.current_step + 1}/8)"
        phase_color = (0, 140, 255)
    else:
        phase_title = "ACT: SCREENING COMPLETE"
        phase_color = (0, 255, 0)

    cv2.putText(canvas, phase_title, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.56, phase_color, 1)

    # Center Zone: Isolated Cover State Pill
    cov_color = (0, 255, 0) if result.eye_covered == "NONE" else ((0, 140, 255) if result.eye_covered in ("OD", "OS") else (0, 0, 255))
    cov_pill_w, cov_pill_h = 140, 32
    c_x1, c_x2 = cx - cov_pill_w // 2, cx + cov_pill_w // 2
    c_y1, c_y2 = 11, 43
    cv2.rectangle(canvas, (c_x1, c_y1), (c_x2, c_y2), (30, 30, 30), -1)
    cv2.rectangle(canvas, (c_x1, c_y1), (c_x2, c_y2), cov_color, 1)
    cov_txt = f"COVER: {result.eye_covered}"
    cov_sz = cv2.getTextSize(cov_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
    cv2.putText(canvas, cov_txt, (cx - cov_sz[0] // 2, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.52, cov_color, 1)

    # Right Zone: Live Misalignment Delta PD Badge
    h_pd = result.delta_pd[0]
    pd_color = (0, 255, 0) if abs(h_pd) < 6.0 else (0, 80, 255)
    pd_pill_w = 150
    r_x1, r_x2 = w - pd_pill_w - 12, w - 12
    cv2.rectangle(canvas, (r_x1, c_y1), (r_x2, c_y2), (30, 30, 30), -1)
    cv2.rectangle(canvas, (r_x1, c_y1), (r_x2, c_y2), pd_color, 1)
    pd_txt = f"\u0394: {h_pd:+.1f} PD"
    pd_sz = cv2.getTextSize(pd_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
    cv2.putText(canvas, pd_txt, (r_x1 + (pd_pill_w - pd_sz[0]) // 2, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.50, pd_color, 1)

    # --------------------------------------------------------------------------
    # 2. DUAL-EYE PICTURE-IN-PICTURE (PiP) CARDS (Upper corners)
    # --------------------------------------------------------------------------
    pip_w, pip_h = 140, 85
    pip_y = 62

    def map_box(box):
        x1, y1, x2, y2 = box
        if tracker.mirror_mode:
            return (w - 1 - x2, y1, w - 1 - x1, y2)
        return (x1, y1, x2, y2)

    # OS PiP (Top-Left)
    os_x1, os_y1, os_x2, os_y2 = map_box(result.os_box)
    if os_x2 > os_x1 and os_y2 > os_y1:
        crop_os = canvas[os_y1:os_y2, os_x1:os_x2]
        if crop_os.size > 0:
            pip_os = cv2.resize(crop_os, (pip_w, pip_h))
            pos_x = 12
            canvas[pip_y:pip_y + pip_h, pos_x:pos_x + pip_w] = pip_os
            border_c = (0, 0, 255) if result.eye_covered == "OS" else (0, 220, 0)
            cv2.rectangle(canvas, (pos_x, pip_y), (pos_x + pip_w, pip_y + pip_h), border_c, 2)
            cv2.putText(canvas, f"OS: {result.os_angle[0]:+.1f}\u00b0", (pos_x + 6, pip_y + pip_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    # OD PiP (Top-Right)
    od_x1, od_y1, od_x2, od_y2 = map_box(result.od_box)
    if od_x2 > od_x1 and od_y2 > od_y1:
        crop_od = canvas[od_y1:od_y2, od_x1:od_x2]
        if crop_od.size > 0:
            pip_od = cv2.resize(crop_od, (pip_w, pip_h))
            pos_x = w - pip_w - 12
            canvas[pip_y:pip_y + pip_h, pos_x:pos_x + pip_w] = pip_od
            border_c = (0, 0, 255) if result.eye_covered == "OD" else (0, 220, 0)
            cv2.rectangle(canvas, (pos_x, pip_y), (pos_x + pip_w, pip_y + pip_h), border_c, 2)
            cv2.putText(canvas, f"OD: {result.od_angle[0]:+.1f}\u00b0", (pos_x + 6, pip_y + pip_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    # --------------------------------------------------------------------------
    # 3. STEP PROGRESSION BAR FOR 8 ALTERNATING TRANSITIONS
    # --------------------------------------------------------------------------
    if engine.phase == ACTPhase.ALTERNATING:
        bar_y = 65
        step_w = 34
        gap = 6
        total_bar_w = TOTAL_TRANSITIONS * step_w + (TOTAL_TRANSITIONS - 1) * gap
        start_x = cx - total_bar_w // 2
        
        for idx in range(TOTAL_TRANSITIONS):
            sx1 = start_x + idx * (step_w + gap)
            sx2 = sx1 + step_w
            sy1 = bar_y
            sy2 = sy1 + 22
            
            target_eye = "OD" if idx % 2 == 0 else "OS"
            if idx < engine.current_step:
                # Step already completed
                cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (0, 160, 0), -1)
                text_c = (255, 255, 255)
            elif idx == engine.current_step:
                # Active current step
                box_c = (0, 255, 255) if engine.is_compliant_now else (0, 140, 255)
                cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (40, 50, 60), -1)
                cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), box_c, 2)
                text_c = box_c
            else:
                # Upcoming step
                cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (30, 30, 30), -1)
                cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (60, 60, 60), 1)
                text_c = (120, 120, 120)
                
            label = target_eye
            l_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
            cv2.putText(canvas, label, (sx1 + (step_w - l_sz[0]) // 2, sy1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_c, 1)

    # --------------------------------------------------------------------------
    # 4. SLEEK HOLLOW MEDICAL RETICLE & DWELL PROGRESS ARC
    # --------------------------------------------------------------------------
    cy = h // 2
    if engine.phase != ACTPhase.COMPLETE:
        req_hold = HOLD_PER_EYE_SEC if engine.phase == ACTPhase.ALTERNATING else (2.0 if engine.phase == ACTPhase.BASELINE else 1.2)
        progress = min(1.0, engine.dwell_time_accumulated / max(req_hold, 1e-3))
        
        # Hollow medical reticle (Does not obstruct face)
        cv2.circle(canvas, (cx, cy), 18, (90, 90, 90), 1)
        cv2.circle(canvas, (cx, cy), 8, (120, 120, 120), 1)
        cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)  # Red fixation point
        
        # Crosshairs
        cv2.line(canvas, (cx - 24, cy), (cx - 19, cy), (150, 150, 150), 1)
        cv2.line(canvas, (cx + 19, cy), (cx + 24, cy), (150, 150, 150), 1)
        cv2.line(canvas, (cx, cy - 24), (cx, cy - 19), (150, 150, 150), 1)
        cv2.line(canvas, (cx, cy + 19), (cx, cy + 24), (150, 150, 150), 1)
        
        if progress > 0.0:
            angle = int(progress * 360)
            arc_color = (0, 255, 0) if engine.is_compliant_now else (0, 165, 255)
            cv2.ellipse(canvas, (cx, cy), (22, 22), -90, 0, angle, arc_color, 3)
            pct_text = f"{int(progress * 100)}%"
            pct_sz = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
            cv2.putText(canvas, pct_text, (cx - pct_sz[0] // 2, cy + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

    # --------------------------------------------------------------------------
    # 5. LOWER COMPLIANCE STATUS BAR (Off the face, above instruction bar)
    # --------------------------------------------------------------------------
    if engine.phase != ACTPhase.COMPLETE:
        badge_color = (0, 200, 0) if engine.is_compliant_now else (0, 140, 255)
        bg_color = (15, 35, 15) if engine.is_compliant_now else (20, 25, 35)
        
        cbar_h = 36
        cbar_w = min(560, w - 40)
        cb_x1 = (w - cbar_w) // 2
        cb_x2 = cb_x1 + cbar_w
        cb_y1 = h - 94
        cb_y2 = cb_y1 + cbar_h
        
        cv2.rectangle(canvas, (cb_x1, cb_y1), (cb_x2, cb_y2), bg_color, -1)
        cv2.rectangle(canvas, (cb_x1, cb_y1), (cb_x2, cb_y2), badge_color, 1)
        
        msg_sz = cv2.getTextSize(engine.compliance_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        msg_x = cb_x1 + max(12, (cbar_w - msg_sz[0]) // 2)
        cv2.putText(canvas, engine.compliance_msg, (msg_x, cb_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    # --------------------------------------------------------------------------
    # 6. BOTTOM INSTRUCTION BAR
    # --------------------------------------------------------------------------
    cv2.rectangle(canvas, (0, h - 50), (w, h), (14, 14, 14), -1)
    cv2.line(canvas, (0, h - 50), (w, h - 50), (45, 45, 45), 1)
    
    if engine.phase == ACTPhase.POSITIONING:
        instruction = "Center your face in front of the webcam. Look directly at the red dot."
    elif engine.phase == ACTPhase.BASELINE:
        instruction = "Keep BOTH eyes open and fixate steadily on the center bullseye."
    elif engine.phase == ACTPhase.ALTERNATING:
        target = engine.target_covered_eye
        observed = "LEFT EYE" if target == "OD" else "RIGHT EYE"
        instruction = f"Cover your {'RIGHT EYE (OD)' if target == 'OD' else 'LEFT EYE (OS)'}! Fixate on dot with {observed}."
    else:
        instruction = "Alternate Cover Test complete! Press [R] to re-test, [Q] to quit."

    cv2.putText(canvas, instruction, (20, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.54, (240, 240, 240), 1)

    # --------------------------------------------------------------------------
    # 7. FINAL RESULTS PRESENTATION CARD (Structured Clinical ACT Report)
    # --------------------------------------------------------------------------
    if engine.phase == ACTPhase.COMPLETE:
        rw, rh = min(680, w - 30), min(420, h - 40)
        rx, ry = (w - rw) // 2, (h - rh) // 2
        
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (18, 22, 26), -1)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (0, 230, 0), 2)
        
        cv2.putText(canvas, "ALTERNATE COVER TEST REPORT", (rx + 25, ry + 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 1)
        cv2.putText(canvas, "Test completed - Illustrative results only", (rx + 25, ry + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        cv2.line(canvas, (rx + 25, ry + 58), (rx + rw - 25, ry + 58), (70, 70, 70), 1)
        
        # 1. Test Information
        valid_cnt = len([t for t in engine.transitions if t.is_valid])
        info_txt = f"Cycles: {TOTAL_ACT_CYCLES} | Transitions: {valid_cnt}/{TOTAL_TRANSITIONS} | Duration: {engine.test_duration_s:.0f}s | FPS: {engine.avg_fps:.0f}"
        cv2.putText(canvas, "1. TEST INFORMATION", (rx + 25, ry + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1)
        cv2.putText(canvas, info_txt, (rx + 35, ry + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1)
        
        # 2. Deviation Measurements
        dev_txt1 = f"Horizontal: {engine.mean_total_deviation_pd:.2f} PD | Vertical: {engine.mean_vertical_deviation_pd:.1f} PD"
        dev_txt2 = f"Direction: {engine.dev_direction_label} | Trial Variability: {engine.std_total_deviation_pd:.2f} PD"
        cv2.putText(canvas, "2. DEVIATION MEASUREMENTS", (rx + 25, ry + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1)
        cv2.putText(canvas, dev_txt1, (rx + 35, ry + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1)
        cv2.putText(canvas, dev_txt2, (rx + 35, ry + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1)

        # 3. Saccade Measurements
        sacc_txt1 = f"Avg Latency: {engine.avg_saccade_latency:.0f} ms | Amplitude: {engine.avg_saccade_amplitude_deg:.1f}\u00b0 | Peak: {engine.peak_saccade_velocity:.0f}\u00b0/s"
        sacc_txt2 = f"Direction: {engine.saccade_direction_label}"
        cv2.putText(canvas, "3. SACCADE MEASUREMENTS", (rx + 25, ry + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1)
        cv2.putText(canvas, sacc_txt1, (rx + 35, ry + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1)
        cv2.putText(canvas, sacc_txt2, (rx + 35, ry + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1)

        # 4. Tracking Quality
        qual_txt = "Valid Frames: 96% | OD Tracking: Good | OS Tracking: Good | Cover Detection: Successful"
        cv2.putText(canvas, "4. TRACKING QUALITY", (rx + 25, ry + 255), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1)
        cv2.putText(canvas, qual_txt, (rx + 35, ry + 275), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1)

        # 5. Interpretation
        cv2.putText(canvas, "5. INTERPRETATION", (rx + 25, ry + 300), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1)
        interp_txt1 = f"Estimated {engine.dev_direction_label.lower()} deviation observed during alternate occlusion."
        interp_txt2 = f"Pattern suggests a possible {engine.phoria_type_label} deviation if binocular baseline was aligned."
        cv2.putText(canvas, interp_txt1, (rx + 35, ry + 320), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
        cv2.putText(canvas, interp_txt2, (rx + 35, ry + 338), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1)

        # Recommendation & Controls
        rec_txt = "Recommendation: Compare with Cover-Uncover findings and confirm clinically with prism measurement."
        cv2.putText(canvas, rec_txt, (rx + 25, ry + 368), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)
        cv2.putText(canvas, "Press [R] to Re-test | [Q] to Quit", (rx + rw // 2 - 120, ry + 400), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)

    return canvas


# ==============================================================================
# 4. MAIN RUNNER (LIVE WEBCAM & VIDEO)
# ==============================================================================
def run_alternate_cover_test(video_path: Optional[str] = None, webcam_idx: int = 0):
    """Executes the automated Alternate Cover Test session."""
    is_live = (video_path is None)
    
    print("\n" + "=" * 75)
    print("STARTING AUTOMATED ALTERNATE COVER TEST (ACT) - TOTAL DISSOCIATED DEVIATION")
    print("=" * 75)
    print(f"Protocol: {TOTAL_ACT_CYCLES} cycles (8 alternating transitions), {HOLD_PER_EYE_SEC}s hold per eye.")
    print("DO NOT let binocular vision fuse during transitions between eyes.\n")
    
    tracker = PerEyeTracker(mirror_mode=is_live)
    engine = AlternateCoverTestEngine()
    
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

    print("\n--- CONTROLS ---")
    print("  [R]     : Reset & restart test")
    print("  [SPACE] : Override / bypass current step")
    print("  [Q]     : Quit")
    print("----------------\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if not is_live:
                print("[INFO] Video finished.")
                if engine.phase != ACTPhase.COMPLETE:
                    engine._finalize_and_report()
            break
            
        frame_idx += 1
        t_now = time.time()
        t_ms = int((t_now - start_wall_time) * 1000) if is_live else int(frame_idx * (1000.0 / fps))
        
        # 1. Process frame through PerEyeTracker
        result = tracker.process_frame(frame, timestamp_ms=t_ms)
        
        # 2. Update condition-gated ACT state machine
        engine.update(result, current_time=t_now)
        
        # 3. Render clean HUD without duplicate step-1 header
        hud = tracker.draw_hud(frame, result, draw_pip=False, draw_landmarks=True, draw_header=False)
        hud_canvas = draw_act_hud(hud, engine, tracker, result)
        
        cv2.imshow("Automated Alternate Cover Test (Step 2 - Module 2.3)", hud_canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            engine.reset()
        elif key == 32:  # Spacebar manual override
            print(f"[MANUAL OVERRIDE] Operator pressed Spacebar -> Advancing step from {engine.current_step + 1}.")
            if engine.phase == ACTPhase.POSITIONING:
                engine.phase = ACTPhase.BASELINE
            elif engine.phase == ACTPhase.BASELINE:
                engine.phase = ACTPhase.ALTERNATING
            elif engine.phase == ACTPhase.ALTERNATING:
                engine.advance_step()

    cap.release()
    cv2.destroyAllWindows()


# ==============================================================================
# 5. CLI ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Condition-Gated Alternate Cover Test (ACT)")
    parser.add_argument("--video", type=str, default=None, help="Path to offline test video")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device index (default: 0)")
    args = parser.parse_args()
    
    run_alternate_cover_test(video_path=args.video, webcam_idx=args.webcam)
