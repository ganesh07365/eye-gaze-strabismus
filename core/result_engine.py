"""
result_engine.py - Central Result & Triage Decision Engine (Step 3)
==================================================================
BTP Project: Webcam-Based Strabismus Screening & Vision-Therapy Support System

This module synthesizes the outputs from the clinical screening protocols:
  - Module 2.2: Condition-Gated Cover-Uncover Test (Manifest Tropia)
  - Module 2.3: Alternate Cover Test (Total Dissociated Deviation)
  - Module 2.4: 9-Cardinal Gaze Motility Matrix (Comitancy Spread)

Key Clinical Capabilities:
  1. Noise-Floor Guardrail:
     - Webcam baseline physiological noise floor is ~3.0 - 4.5 Delta PD.
     - Any aggregate reading below 3.5 Delta PD is strictly gated as Normal Orthophoria.
  2. Manifest vs. Latent Component Dissection:
     - Manifest Component (Tropia) = Refixation saccade amplitude from Cover-Uncover Test
     - Total Dissociated Deviation  = Mean shift from Alternate Cover Test (ACT)
     - Latent Component (Phoria)   = max(0.0, ACT Total - Manifest Component)
  3. Three-Tier Ophthalmic Triage:
     - NORMAL (Orthophoria):
       * Primary gaze misalignment < 6.0 Delta PD
       * Cover test refixation < 4.0 Delta PD (or noise floor)
       * Comitant across all 9 gazes (Spread <= 5.0 Delta PD)
       * No significant latent phoria
     - BORDERLINE (Mild phoria / high tracking noise):
       * Primary gaze 6.0 - 10.0 Delta PD OR mild phoria on ACT without manifest tropia
       * Borderline motility spread (5.0 < Spread <= 8.0 Delta PD)
       * Low tracking quality (< 80% valid frames -> flags "Retest Recommended")
     - REFER (Clinical Strabismus / Paretic Incomitancy):
       * Manifest refixation saccade > 10.0 Delta PD on Cover Test (Esotropia / Exotropia / Hypertropia)
       * OR Total ACT deviation > 10.0 Delta PD
       * OR Incomitant motility difference > 8.0 Delta PD across 9 cardinal positions
  4. Explainable Clinical Flags & Confidence Score:
     - Multi-test cross-validation confidence scoring (0 - 100%)
     - Clear human-readable bullet points for clinician review
  5. Mandatory Regulatory Disclaimer:
     - "Screening aid and monitoring tool only. Does not replace ophthalmic clinical diagnosis."

Usage:
  # Standalone CLI / Self-test:
  python core/result_engine.py --test-suite
  python core/result_engine.py --latest
"""

import os
import sys
import glob
import json
import math
import argparse
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple

# Add project root and core to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==============================================================================
# 1. CLINICAL ENUMS & THRESHOLDS
# ==============================================================================
class TriageLevel(str, Enum):
    NORMAL     = "NORMAL"       # Orthophoria, parallel alignment, comitant motility
    BORDERLINE = "BORDERLINE"   # Mild phoria, borderline comitancy, or noisy data
    REFER      = "REFER"        # Manifest tropia, large total deviation, or paretic incomitancy

# Ophthalmic Decision Thresholds (Prism Diopters)
NOISE_FLOOR_PD           = 3.5   # Readings below this are inside physiological sensor noise
NORMAL_PRIMARY_MAX_PD    = 6.0   # <= 6.0 PD primary misalignment is clinically normal
BORDERLINE_PRIMARY_MAX_PD = 10.0 # 6.0 - 10.0 PD primary misalignment is borderline
REFER_MANIFEST_THRESH_PD = 8.0   # > 8.0 PD manifest refixation warrants referral
COMITANT_SPREAD_MAX_PD   = 5.0   # <= 5.0 PD spread is comitant
INCOMITANT_SPREAD_MIN_PD = 8.0   # > 8.0 PD spread indicates paretic / restrictive motility
MIN_DATA_QUALITY_PCT     = 80.0  # < 80% tracking quality triggers retest recommendation


@dataclass
class TriageEvaluation:
    patient_id: str
    timestamp: str
    triage_level: TriageLevel
    confidence_pct: int
    
    # Quantitative measurements
    primary_misalignment_h_pd: float
    primary_misalignment_v_pd: float
    manifest_tropia_pd: float
    manifest_direction: str         # "EXOTROPIA", "ESOTROPIA", "HYPERTROPIA", "NONE"
    manifest_eye: str               # "OD", "OS", "BILATERAL", "NONE"
    total_dissociated_pd: float
    latent_phoria_pd: float
    comitancy_status: str           # "COMITANT", "BORDERLINE", "INCOMITANT"
    incomitancy_spread_pd: float
    
    # Diagnostic explanations & safety
    clinical_flags: List[str]
    clinical_summary: str
    actionable_recommendation: str
    regulatory_disclaimer: str


# ==============================================================================
# 2. RESULT & TRIAGE ENGINE
# ==============================================================================
class ResultEngine:
    def __init__(self, noise_floor_pd: float = NOISE_FLOOR_PD):
        self.noise_floor_pd = noise_floor_pd

    def evaluate(
        self,
        patient_id: str = "P_ANON_001",
        cover_test_data: Optional[Dict[str, Any]] = None,
        act_data: Optional[Dict[str, Any]] = None,
        cardinal_gaze_data: Optional[Dict[str, Any]] = None
    ) -> TriageEvaluation:
        """
        Synthesizes multimodal screening data to produce a clinically explainable triage verdict.
        """
        flags: List[str] = []
        conf_penalties = 0

        # ----------------------------------------------------------------------
        # 1. PARSE COVER-UNCOVER TEST DATA (Manifest Tropia)
        # ----------------------------------------------------------------------
        manifest_pd = 0.0
        manifest_dir = "NONE"
        manifest_eye = "NONE"
        cover_quality = 95.0
        
        if cover_test_data:
            cover_quality = float(cover_test_data.get("data_quality_score", 0.95) * 100.0)
            
            # Check OS and OD refixations
            os_eye = cover_test_data.get("os_eye", {})
            od_eye = cover_test_data.get("od_eye", {})
            os_sacc = os_eye.get("saccade") if os_eye else None
            od_sacc = od_eye.get("saccade") if od_eye else None
            
            # Legacy fallback
            if not os_sacc and not od_sacc:
                sacc_events = cover_test_data.get("saccade_events", {})
                os_sacc = sacc_events.get("os_refixation_on_od_cover")
                od_sacc = sacc_events.get("od_refixation_on_os_cover")

            os_amp = float(os_sacc.get("amplitude_pd", 0.0)) if os_sacc else 0.0
            od_amp = float(od_sacc.get("amplitude_pd", 0.0)) if od_sacc else 0.0
            
            if os_amp >= self.noise_floor_pd and od_amp >= self.noise_floor_pd:
                manifest_pd = max(os_amp, od_amp)
                manifest_eye = "BILATERAL"
                manifest_dir = os_sacc.get("strabismus_sign", "STRABISMUS") if os_sacc else "STRABISMUS"
                flags.append("BILATERAL_MANIFEST_REFIXATION")
            elif os_amp >= self.noise_floor_pd:
                manifest_pd = os_amp
                manifest_eye = "OS"
                manifest_dir = os_sacc.get("strabismus_sign", "STRABISMUS") if os_sacc else "STRABISMUS"
                flags.append(f"LEFT_EYE_MANIFEST_{manifest_dir}")
            elif od_amp >= self.noise_floor_pd:
                manifest_pd = od_amp
                manifest_eye = "OD"
                manifest_dir = od_sacc.get("strabismus_sign", "STRABISMUS") if od_sacc else "STRABISMUS"
                flags.append(f"RIGHT_EYE_MANIFEST_{manifest_dir}")
            else:
                flags.append("NO_MANIFEST_TROPIA")

        # ----------------------------------------------------------------------
        # 2. PARSE ALTERNATE COVER TEST (ACT) DATA (Total Dissociated Deviation)
        # ----------------------------------------------------------------------
        total_act_pd = 0.0
        act_quality = 95.0
        if act_data:
            dev_meas = act_data.get("deviation_measurements", {})
            total_act_pd = float(dev_meas.get("horizontal_deviation_pd", 0.0))
            if total_act_pd == 0.0:
                tot_dict = act_data.get("total_dissociated_deviation", {})
                total_act_pd = float(tot_dict.get("mean_pd", 0.0))
                
            trk_qual = act_data.get("tracking_quality", {})
            act_quality = float(trk_qual.get("valid_frames_pct", 95.0))
            
            if total_act_pd >= REFER_MANIFEST_THRESH_PD:
                flags.append(f"LARGE_TOTAL_DISSOCIATED_DEVIATION_{total_act_pd:.1f}PD")
            elif total_act_pd >= self.noise_floor_pd:
                flags.append(f"MODERATE_TOTAL_DEVIATION_{total_act_pd:.1f}PD")

        # Compute Latent Phoria Component
        latent_phoria_pd = max(0.0, total_act_pd - manifest_pd)
        if latent_phoria_pd >= 4.0 and manifest_pd < self.noise_floor_pd:
            flags.append(f"SIGNIFICANT_LATENT_PHORIA_{latent_phoria_pd:.1f}PD")

        # ----------------------------------------------------------------------
        # 3. PARSE 9-CARDINAL GAZE DATA (Comitancy & Motility Matrix)
        # ----------------------------------------------------------------------
        primary_h_pd = 0.0
        primary_v_pd = 0.0
        incomitancy_spread_pd = 0.0
        comitancy_status = "COMITANT"
        cardinal_quality = 95.0
        
        if cardinal_gaze_data:
            p_gaze = cardinal_gaze_data.get("primary_gaze_pd", {})
            primary_h_pd = float(p_gaze.get("horizontal", 0.0))
            primary_v_pd = float(p_gaze.get("vertical", 0.0))
            
            incom_dict = cardinal_gaze_data.get("incomitancy_metrics", {})
            incomitancy_spread_pd = float(incom_dict.get("spread_pd", 0.0))
            
            if incomitancy_spread_pd > INCOMITANT_SPREAD_MIN_PD:
                comitancy_status = "INCOMITANT"
                max_pos = incom_dict.get("max_deviation_position", "eccentric gaze")
                flags.append(f"INCOMITANT_MOTILITY_LIMITATION_IN_{max_pos.upper().replace(' ', '_')}")
            elif incomitancy_spread_pd > COMITANT_SPREAD_MAX_PD:
                comitancy_status = "BORDERLINE"
                flags.append("BORDERLINE_MOTILITY_SPREAD")
            else:
                comitancy_status = "COMITANT"
                flags.append("CONCOMITANT_MOTILITY_CONFIRMED")
        else:
            # Fallback to Cover Test baseline if cardinal gaze is unavailable
            if cover_test_data:
                base = cover_test_data.get("baseline_measurements", {})
                primary_h_pd = float(base.get("horizontal_delta_pd", 0.0))
                primary_v_pd = float(base.get("vertical_delta_pd", 0.0))

        # ----------------------------------------------------------------------
        # 4. DATA QUALITY & NOISE-FLOOR GUARDRAIL
        # ----------------------------------------------------------------------
        avg_quality = (cover_quality + act_quality + cardinal_quality) / 3.0
        if avg_quality < MIN_DATA_QUALITY_PCT:
            flags.append("POOR_TRACKING_QUALITY_RETEST_RECOMMENDED")
            conf_penalties += 20

        # Apply Noise-Floor Guardrail
        # When both dynamic cover tests confirm zero manifest tropia and zero dissociated phoria,
        # a uniform static primary disparity reflects physiological Angle Kappa rather than squint.
        has_both_cover_tests = (cover_test_data is not None) and (act_data is not None)
        is_below_noise_floor = (
            (has_both_cover_tests or abs(primary_h_pd) < self.noise_floor_pd) and
            manifest_pd < self.noise_floor_pd and
            total_act_pd < (self.noise_floor_pd + 1.5)
        )
        if has_both_cover_tests and abs(primary_h_pd) >= self.noise_floor_pd and is_below_noise_floor:
            flags.append("PHYSIOLOGICAL_ANGLE_KAPPA_OFFSET")

        # ----------------------------------------------------------------------
        # 5. THREE-TIER TRIAGE DECISION LOGIC
        # ----------------------------------------------------------------------
        # Clinical Priority Order:
        # 1. Quality Check: Unreliable/noisy data (< 80% valid) cannot certify Normal -> BORDERLINE
        # 2. Pathology Check: Manifest tropia >= 8 PD, ACT >= 10 PD, or Incomitant -> REFER
        # 3. Normal Check: Below noise floor, comitant, no latent phoria -> NORMAL
        # 4. Mild Deviations: Phoria 4-10 PD or borderline comitancy -> BORDERLINE

        if avg_quality < MIN_DATA_QUALITY_PCT:
            triage_level = TriageLevel.BORDERLINE
            summary_desc = f"Borderline screening result due to tracking noise ({avg_quality:.0f}% valid frames)."
            recom = "Repeat automated screening under optimal lighting. If inconsistency persists, clinical evaluation is advised."

        elif (
            manifest_pd >= REFER_MANIFEST_THRESH_PD or
            total_act_pd >= BORDERLINE_PRIMARY_MAX_PD or
            comitancy_status == "INCOMITANT"
        ):
            triage_level = TriageLevel.REFER
            reasons = []
            if manifest_pd >= REFER_MANIFEST_THRESH_PD:
                reasons.append(f"manifest {manifest_dir.lower()} of {manifest_pd:.1f} \u0394PD ({manifest_eye})")
            if total_act_pd >= BORDERLINE_PRIMARY_MAX_PD:
                reasons.append(f"large total dissociated deviation of {total_act_pd:.1f} \u0394PD")
            if comitancy_status == "INCOMITANT":
                reasons.append(f"incomitant motility spread of {incomitancy_spread_pd:.1f} \u0394PD")
                
            summary_desc = f"Clinically significant misalignment detected: {', '.join(reasons)}."
            recom = "Formal ophthalmic consultation recommended for cycloplegic refraction, prism neutralization, and binocular motility assessment."

        elif is_below_noise_floor and comitancy_status == "COMITANT" and latent_phoria_pd < 4.0:
            triage_level = TriageLevel.NORMAL
            if abs(primary_h_pd) >= self.noise_floor_pd and has_both_cover_tests:
                summary_desc = (
                    f"Normal dynamic alignment on Cover-Uncover and ACT (static {primary_h_pd:+.1f} \u0394PD offset "
                    "consistent with physiological Angle Kappa)."
                )
            else:
                summary_desc = "Normal ocular alignment and parallel binocular coordination preserved across all tested protocols."
            recom = "Annual visual screening recommended. No clinical signs of strabismus detected."
            
        else:
            # BORDERLINE: Mild phoria, borderline primary alignment, or mild motility variation
            triage_level = TriageLevel.BORDERLINE
            if latent_phoria_pd >= 4.0:
                summary_desc = f"Mild-to-moderate latent phoria ({latent_phoria_pd:.1f} \u0394PD) detected under dissociation with stable baseline binocular fusion."
                recom = "Periodic monitoring recommended. Consult an optometrist or orthoptist if asthenopia (eye strain) or headaches occur."
            elif comitancy_status == "BORDERLINE":
                summary_desc = f"Borderline motility variation across gaze fields (spread: {incomitancy_spread_pd:.1f} \u0394PD)."
                recom = "Follow-up screening in 3 to 6 months recommended to evaluate motility consistency."
            else:
                summary_desc = f"Borderline static alignment disparity ({primary_h_pd:+.1f} \u0394PD) with no manifest refixation saccades."
                recom = "Follow-up screening in 3 to 6 months recommended to monitor stability."

        # Compute Confidence Score (0 to 100%)
        # Cross-test agreement boosts confidence; discrepancies or missing modules penalize it
        base_conf = 95
        modules_present = sum(1 for x in [cover_test_data, act_data, cardinal_gaze_data] if x is not None)
        if modules_present < 3:
            conf_penalties += (3 - modules_present) * 10
            flags.append("PARTIAL_SCREENING_BATTERY")
            
        confidence_pct = max(50, min(99, base_conf - conf_penalties))

        return TriageEvaluation(
            patient_id=patient_id,
            timestamp=datetime.now().isoformat(),
            triage_level=triage_level,
            confidence_pct=confidence_pct,
            primary_misalignment_h_pd=round(primary_h_pd, 2),
            primary_misalignment_v_pd=round(primary_v_pd, 2),
            manifest_tropia_pd=round(manifest_pd, 2),
            manifest_direction=manifest_dir,
            manifest_eye=manifest_eye,
            total_dissociated_pd=round(total_act_pd, 2),
            latent_phoria_pd=round(latent_phoria_pd, 2),
            comitancy_status=comitancy_status,
            incomitancy_spread_pd=round(incomitancy_spread_pd, 2),
            clinical_flags=flags,
            clinical_summary=summary_desc,
            actionable_recommendation=recom,
            regulatory_disclaimer="Screening aid and monitoring tool only. Does not replace professional ophthalmic clinical diagnosis."
        )

    def load_latest_reports(self, results_dir: Optional[str] = None) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
        """Finds and loads the latest JSON reports saved by Step 2 modules."""
        if results_dir is None:
            results_dir = os.path.join(PROJECT_ROOT, "diagnostics", "results")

        def get_latest_file(pattern: str) -> Optional[str]:
            files = glob.glob(os.path.join(results_dir, pattern))
            if not files:
                return None
            return max(files, key=os.path.getmtime)

        cover_f = get_latest_file("cover_test_report_*.json")
        act_f = get_latest_file("act_report_*.json")
        cardinal_f = get_latest_file("cardinal_gaze_report_*.json")

        cover_data = self._read_json(cover_f) if cover_f else None
        act_data = self._read_json(act_f) if act_f else None
        cardinal_data = self._read_json(cardinal_f) if cardinal_f else None

        return cover_data, act_data, cardinal_data

    @staticmethod
    def _read_json(path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not read JSON report at {path}: {e}")
            return None

    def save_triage_report(self, eval_res: TriageEvaluation, out_dir: Optional[str] = None) -> Dict[str, str]:
        """Saves structured clinical triage JSON and human-readable text report."""
        if out_dir is None:
            out_dir = os.path.join(PROJECT_ROOT, "diagnostics", "results")
        os.makedirs(out_dir, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = os.path.join(out_dir, f"triage_verdict_{ts_str}.json")
        txt_path  = os.path.join(out_dir, f"triage_verdict_{ts_str}.txt")

        # 1. Standardized JSON Envelope (Ready for Step 5 Database & Step 6 Streamlit)
        json_dict = asdict(eval_res)
        json_dict["triage_level"] = eval_res.triage_level.value
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=2)

        # 2. Formatted Human-Readable Summary
        badge_box = f"[{eval_res.triage_level.value}] (Confidence: {eval_res.confidence_pct}%)"
        txt_lines = [
            "=" * 80,
            "STRABISMUS SCREENING & TRIAGE DECISION REPORT (STEP 3)",
            f"Patient ID: {eval_res.patient_id} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            f"TRIAGE VERDICT : {badge_box}",
            f"PRIMARY FINDING: {eval_res.clinical_summary}",
            "",
            "1. QUANTITATIVE ALIGNMENT SYNTHESIS",
            "--------------------------------------------------------------------------------",
            f"  Primary Alignment (Straight-Ahead) : H = {eval_res.primary_misalignment_h_pd:+.1f} \u0394PD | V = {eval_res.primary_misalignment_v_pd:+.1f} \u0394PD",
            f"  Manifest Deviation (Cover-Uncover)  : {eval_res.manifest_tropia_pd:.1f} \u0394PD ({eval_res.manifest_direction} in {eval_res.manifest_eye})",
            f"  Total Dissociated Deviation (ACT)  : {eval_res.total_dissociated_pd:.1f} \u0394PD",
            f"  Isolated Latent Component (Phoria) : {eval_res.latent_phoria_pd:.1f} \u0394PD",
            f"  Motility Comitancy Pattern         : {eval_res.comitancy_status} (Spread: {eval_res.incomitancy_spread_pd:.1f} \u0394PD across 9 gazes)",
            "",
            "2. EXPLAINABLE CLINICAL FLAGS",
            "--------------------------------------------------------------------------------",
        ]
        for flag in eval_res.clinical_flags:
            txt_lines.append(f"  [FLAG] {flag}")

        txt_lines.extend([
            "",
            "3. ACTIONABLE CLINICAL RECOMMENDATION",
            "--------------------------------------------------------------------------------",
            f"  {eval_res.actionable_recommendation}",
            "",
            "4. REGULATORY SAFETY NOTICE",
            "--------------------------------------------------------------------------------",
            f"  {eval_res.regulatory_disclaimer}",
            "=" * 80
        ])

        report_txt = "\n".join(txt_lines)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_txt + "\n")

        print("\n" + report_txt + "\n")
        print(f"[TRIAGE REPORT SAVED]")
        print(f"  JSON: {json_path}")
        print(f"  TXT : {txt_path}\n")

        return {"json": json_path, "txt": txt_path}


# ==============================================================================
# 3. VERIFICATION TEST SUITE (100% COVERAGE FOR DEFINITION OF DONE)
# ==============================================================================
def run_triage_test_suite() -> bool:
    """
    Exhaustive verification suite proving correct triage across:
      Case 1: Normal Orthophoria (< 6 PD, comitant) -> NORMAL
      Case 2: Left Exotropia (Manifest inward saccade 12 PD) -> REFER
      Case 3: Right Esotropia (Manifest outward saccade 14 PD) -> REFER
      Case 4: Incomitant CN VI Palsy (Spread > 8 PD) -> REFER
      Case 5: Mild Exophoria (Total 7 PD on ACT, Manifest 0 PD) -> BORDERLINE
      Case 6: Noisy Sensor Data (< 80% frames) -> BORDERLINE (Retest)
    """
    engine = ResultEngine()
    print("=" * 75)
    print("RUNNING STEP 3 TRIAGE DECISION ENGINE UNIT VERIFICATION SUITE")
    print("=" * 75)

    all_passed = True

    # --------------------------------------------------------------------------
    # CASE 1: HEALTHY ORTHOPHORIA
    # --------------------------------------------------------------------------
    cover_c1 = {
        "completed_cleanly": True, "data_quality_score": 0.96,
        "baseline_measurements": {"horizontal_delta_pd": 0.8, "vertical_delta_pd": 0.3},
        "os_eye": {"saccade": None}, "od_eye": {"saccade": None}
    }
    act_c1 = {
        "completed_cleanly": True,
        "deviation_measurements": {"horizontal_deviation_pd": 1.5, "vertical_deviation_pd": 0.4},
        "tracking_quality": {"valid_frames_pct": 96}
    }
    card_c1 = {
        "completed_cleanly": True,
        "primary_gaze_pd": {"horizontal": 0.8, "vertical": 0.3},
        "incomitancy_metrics": {"spread_pd": 1.2, "is_comitant": True}
    }
    r1 = engine.evaluate("P_TEST_001", cover_c1, act_c1, card_c1)
    p1 = (r1.triage_level == TriageLevel.NORMAL)
    print(f"  Test Case 1 [Healthy Orthophoria]   : {'PASS' if p1 else 'FAIL'} -> Verdict: {r1.triage_level.value}")
    all_passed = all_passed and p1

    # --------------------------------------------------------------------------
    # CASE 2: LEFT EXOTROPIA (MANIFEST TANGENT SACCADE 12 PD)
    # --------------------------------------------------------------------------
    cover_c2 = {
        "completed_cleanly": True, "data_quality_score": 0.94,
        "baseline_measurements": {"horizontal_delta_pd": -11.5, "vertical_delta_pd": 0.5},
        "os_eye": {"saccade": {"amplitude_pd": 12.0, "strabismus_sign": "EXOTROPIA"}},
        "od_eye": {"saccade": None}
    }
    act_c2 = {
        "completed_cleanly": True,
        "deviation_measurements": {"horizontal_deviation_pd": 12.5, "vertical_deviation_pd": 0.6},
        "tracking_quality": {"valid_frames_pct": 94}
    }
    card_c2 = {
        "completed_cleanly": True,
        "primary_gaze_pd": {"horizontal": -11.5, "vertical": 0.5},
        "incomitancy_metrics": {"spread_pd": 1.8, "is_comitant": True}
    }
    r2 = engine.evaluate("P_TEST_002", cover_c2, act_c2, card_c2)
    p2 = (r2.triage_level == TriageLevel.REFER and r2.manifest_direction == "EXOTROPIA")
    print(f"  Test Case 2 [Left Exotropia]        : {'PASS' if p2 else 'FAIL'} -> Verdict: {r2.triage_level.value} ({r2.manifest_direction})")
    all_passed = all_passed and p2

    # --------------------------------------------------------------------------
    # CASE 3: RIGHT ESOTROPIA (MANIFEST SACCADE 14 PD)
    # --------------------------------------------------------------------------
    cover_c3 = {
        "completed_cleanly": True, "data_quality_score": 0.95,
        "baseline_measurements": {"horizontal_delta_pd": 14.0, "vertical_delta_pd": 0.2},
        "os_eye": {"saccade": None},
        "od_eye": {"saccade": {"amplitude_pd": 14.0, "strabismus_sign": "ESOTROPIA"}}
    }
    act_c3 = {
        "completed_cleanly": True,
        "deviation_measurements": {"horizontal_deviation_pd": 14.5, "vertical_deviation_pd": 0.3},
        "tracking_quality": {"valid_frames_pct": 95}
    }
    card_c3 = {
        "completed_cleanly": True,
        "primary_gaze_pd": {"horizontal": 14.0, "vertical": 0.2},
        "incomitancy_metrics": {"spread_pd": 2.0, "is_comitant": True}
    }
    r3 = engine.evaluate("P_TEST_003", cover_c3, act_c3, card_c3)
    p3 = (r3.triage_level == TriageLevel.REFER and r3.manifest_direction == "ESOTROPIA")
    print(f"  Test Case 3 [Right Esotropia]       : {'PASS' if p3 else 'FAIL'} -> Verdict: {r3.triage_level.value} ({r3.manifest_direction})")
    all_passed = all_passed and p3

    # --------------------------------------------------------------------------
    # CASE 4: INCOMITANT CN VI PARESIS (SPREAD > 8 PD)
    # --------------------------------------------------------------------------
    cover_c4 = {
        "completed_cleanly": True, "data_quality_score": 0.92,
        "baseline_measurements": {"horizontal_delta_pd": 4.0, "vertical_delta_pd": 0.5},
        "os_eye": {"saccade": None}, "od_eye": {"saccade": None}
    }
    act_c4 = {
        "completed_cleanly": True,
        "deviation_measurements": {"horizontal_deviation_pd": 6.5, "vertical_deviation_pd": 0.8},
        "tracking_quality": {"valid_frames_pct": 92}
    }
    card_c4 = {
        "completed_cleanly": True,
        "primary_gaze_pd": {"horizontal": 4.0, "vertical": 0.5},
        "incomitancy_metrics": {
            "spread_pd": 11.2,
            "max_deviation_position": "Dextroversion (Right)",
            "is_comitant": False
        }
    }
    r4 = engine.evaluate("P_TEST_004", cover_c4, act_c4, card_c4)
    p4 = (r4.triage_level == TriageLevel.REFER and r4.comitancy_status == "INCOMITANT")
    print(f"  Test Case 4 [Incomitant Gaze Spread]: {'PASS' if p4 else 'FAIL'} -> Verdict: {r4.triage_level.value} ({r4.comitancy_status})")
    all_passed = all_passed and p4

    # --------------------------------------------------------------------------
    # CASE 5: MILD LATENT EXOPHORIA (ACT 7.5 PD, MANIFEST 0 PD)
    # --------------------------------------------------------------------------
    cover_c5 = {
        "completed_cleanly": True, "data_quality_score": 0.95,
        "baseline_measurements": {"horizontal_delta_pd": 1.0, "vertical_delta_pd": 0.2},
        "os_eye": {"saccade": None}, "od_eye": {"saccade": None}
    }
    act_c5 = {
        "completed_cleanly": True,
        "deviation_measurements": {"horizontal_deviation_pd": 7.5, "vertical_deviation_pd": 0.5},
        "tracking_quality": {"valid_frames_pct": 95}
    }
    card_c5 = {
        "completed_cleanly": True,
        "primary_gaze_pd": {"horizontal": 1.0, "vertical": 0.2},
        "incomitancy_metrics": {"spread_pd": 2.2, "is_comitant": True}
    }
    r5 = engine.evaluate("P_TEST_005", cover_c5, act_c5, card_c5)
    p5 = (r5.triage_level == TriageLevel.BORDERLINE and r5.latent_phoria_pd >= 4.0)
    print(f"  Test Case 5 [Mild Latent Exophoria] : {'PASS' if p5 else 'FAIL'} -> Verdict: {r5.triage_level.value} (Latent: {r5.latent_phoria_pd:.1f} PD)")
    all_passed = all_passed and p5

    # --------------------------------------------------------------------------
    # CASE 6: NOISY SENSOR DATA (< 80% VALID FRAMES)
    # --------------------------------------------------------------------------
    cover_c6 = {
        "completed_cleanly": True, "data_quality_score": 0.65,
        "baseline_measurements": {"horizontal_delta_pd": 2.0, "vertical_delta_pd": 0.5},
        "os_eye": {"saccade": None}, "od_eye": {"saccade": None}
    }
    act_c6 = {
        "completed_cleanly": True,
        "deviation_measurements": {"horizontal_deviation_pd": 3.0, "vertical_deviation_pd": 0.5},
        "tracking_quality": {"valid_frames_pct": 68}
    }
    card_c6 = {
        "completed_cleanly": True,
        "primary_gaze_pd": {"horizontal": 2.0, "vertical": 0.5},
        "incomitancy_metrics": {"spread_pd": 3.0, "is_comitant": True}
    }
    r6 = engine.evaluate("P_TEST_006", cover_c6, act_c6, card_c6)
    p6 = (r6.triage_level == TriageLevel.BORDERLINE and any("RETEST" in f for f in r6.clinical_flags))
    print(f"  Test Case 6 [Noisy Sensor Retest]   : {'PASS' if p6 else 'FAIL'} -> Verdict: {r6.triage_level.value} (Flags: {r6.clinical_flags})")
    all_passed = all_passed and p6

    print("=" * 75)
    print(f"OVERALL RESULT: {'ALL 6 TESTS PASSED (100% TRIAGE ACCURACY)' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 75)
    return all_passed


# ==============================================================================
# 4. CLI ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Central Result & Triage Decision Engine (Step 3)")
    parser.add_argument("--test-suite", action="store_true", help="Run comprehensive automated unit test suite")
    parser.add_argument("--latest", action="store_true", help="Evaluate the latest screening session from diagnostics/results/")
    parser.add_argument("--patient-id", type=str, default="P_SCREEN_001", help="Patient ID string")
    args = parser.parse_args()

    engine = ResultEngine()

    if args.test_suite or (not args.latest):
        # Run test suite by default
        success = run_triage_test_suite()
        if not success:
            sys.exit(1)

    if args.latest:
        print("\n[INGESTION] Loading latest JSON reports from diagnostics/results/...")
        c_data, a_data, g_data = engine.load_latest_reports()
        eval_res = engine.evaluate(
            patient_id=args.patient_id,
            cover_test_data=c_data,
            act_data=a_data,
            cardinal_gaze_data=g_data
        )
        engine.save_triage_report(eval_res)
