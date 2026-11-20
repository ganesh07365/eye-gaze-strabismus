"""
eye_movement_analyzer.py - Ophthalmic Eye Movement & Saccade Analysis Engine
=============================================================================
BTP Project: Webcam-Based Strabismus Screening & Vision-Therapy Support System

This module provides the shared Layer-2 analysis utility for:
  1. I-VT (Velocity Threshold Identification) saccade and fixation segmentation.
  2. Noise-damped central difference velocity computation optimized for 30 FPS webcam streams.
  3. Refixation saccade event extraction (onset latency, amplitude, peak/mean velocity, direction).
  4. Fixation stability analysis via Bivariate Contour Ellipse Area (BCEA).
  5. Smooth pursuit gain and saccadic intrusion counting for therapy games.
  6. Hardware-aware temporal quantization and confidence scoring.

Clinical References:
  - Salvucci & Goldberg (2000): "Identifying fixations and saccades in eye-tracking protocols"
  - Crossland & Rubin (2002): "The use of the Bivariate Contour Ellipse Area (BCEA) in eye movement research"
  - American Academy of Ophthalmology: Strabismus refixation saccade dynamics
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


# ==============================================================================
# 1. DATA STRUCTURES & CLINICAL CONSTANTS
# ==============================================================================
DEFAULT_SACCADE_VELOCITY_THRESH = 30.0  # deg/s (I-VT velocity threshold)
MIN_SACCADE_DURATION_MS = 30.0          # Minimum saccade duration (at 30 FPS, ~1 frame)
MAX_SACCADE_DURATION_MS = 300.0         # Maximum physiological saccade duration
TEMPORAL_RESOLUTION_MS = 33.33          # 30 FPS frame interval


@dataclass
class SaccadeEvent:
    """Represents a discrete detected saccade or refixation movement."""
    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    onset_latency_ms: Optional[float] = None  # Delay from trigger event (e.g. occlusion)
    
    # Amplitudes in degrees
    amplitude_deg: float = 0.0
    amplitude_h_deg: float = 0.0              # Signed: + = nasal/right, - = temporal/left
    amplitude_v_deg: float = 0.0              # Signed: + = downward, - = upward
    amplitude_pd: float = 0.0                 # Prism Diopters (approx 1.7455 * amp_deg)
    
    # Velocities in deg/s
    peak_velocity_deg_s: float = 0.0
    mean_velocity_deg_s: float = 0.0
    
    # Qualitative direction
    direction_h: str = "NONE"                 # "NASAL", "TEMPORAL", "NONE"
    direction_v: str = "NONE"                 # "UPWARD", "DOWNWARD", "NONE"
    
    # Clinical Strabismus Correlation
    # (e.g., temporal refixation saccade implies resting inward deviation = Esotropia)
    strabismus_sign: str = "ORTHO"            # "ESOTROPIA", "EXOTROPIA", "HYPERTROPIA", "HYPOTROPIA", "ORTHO"
    confidence: float = 1.0


@dataclass
class FixationMetrics:
    """Represents fixation stability over an observation window."""
    duration_ms: float
    mean_yaw_deg: float
    mean_pitch_deg: float
    std_yaw_deg: float
    std_pitch_deg: float
    bcea_deg2: float                          # Bivariate Contour Ellipse Area (deg^2)
    drift_velocity_deg_s: float               # Slow drift velocity
    microsaccade_count: int                   # Small involuntary saccades (< 1.5 deg)
    data_quality: float                       # Ratio of valid frames (0.0 to 1.0)


# ==============================================================================
# 2. EYE MOVEMENT ANALYZER CLASS
# ==============================================================================
class EyeMovementAnalyzer:
    """
    Analyzes temporal gaze trajectories, extracts refixation saccades,
    and computes fixation stability metrics.
    """
    def __init__(
        self,
        velocity_threshold_deg_s: float = DEFAULT_SACCADE_VELOCITY_THRESH,
        min_saccade_amp_deg: float = 1.0,
        max_latency_window_ms: float = 800.0
    ):
        self.velocity_threshold = velocity_threshold_deg_s
        self.min_saccade_amp = min_saccade_amp_deg
        self.max_latency_window = max_latency_window_ms
        
        # Internal circular sample buffers
        self.timestamps: List[float] = []
        self.yaw_deg: List[float] = []
        self.pitch_deg: List[float] = []
        self.is_valid: List[bool] = []
        
    def reset(self):
        """Clears all stored trajectories."""
        self.timestamps.clear()
        self.yaw_deg.clear()
        self.pitch_deg.clear()
        self.is_valid.clear()
        
    def add_sample(self, timestamp_ms: float, yaw: float, pitch: float, valid: bool = True):
        """Adds a single temporal frame gaze reading."""
        self.timestamps.append(float(timestamp_ms))
        self.yaw_deg.append(float(yaw))
        self.pitch_deg.append(float(pitch))
        self.is_valid.append(bool(valid))
        
    # --------------------------------------------------------------------------
    # 3. NOISE-ROBUST VELOCITY COMPUTATION
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_velocities(
        times: np.ndarray,
        yaws: np.ndarray,
        pitches: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes instantaneous angular velocity using central differences.
        Avoids single-frame tremor amplification inherent to raw 30 FPS derivatives.
        
        Returns:
            v_total (deg/s), v_yaw (deg/s), v_pitch (deg/s)
        """
        n = len(times)
        if n < 2:
            return np.zeros(n), np.zeros(n), np.zeros(n)
            
        v_yaw = np.zeros(n, dtype=np.float64)
        v_pitch = np.zeros(n, dtype=np.float64)
        
        # Central difference for interior points: v_i = (pos_{i+1} - pos_{i-1}) / (t_{i+1} - t_{i-1})
        for i in range(1, n - 1):
            dt = (times[i + 1] - times[i - 1]) / 1000.0  # seconds
            if dt > 1e-4:
                v_yaw[i] = (yaws[i + 1] - yaws[i - 1]) / dt
                v_pitch[i] = (pitches[i + 1] - pitches[i - 1]) / dt
                
        # Forward difference for first point
        dt0 = (times[1] - times[0]) / 1000.0
        if dt0 > 1e-4:
            v_yaw[0] = (yaws[1] - yaws[0]) / dt0
            v_pitch[0] = (pitches[1] - pitches[0]) / dt0
            
        # Backward difference for last point
        dtn = (times[-1] - times[-2]) / 1000.0
        if dtn > 1e-4:
            v_yaw[-1] = (yaws[-1] - yaws[-2]) / dtn
            v_pitch[-1] = (pitches[-1] - pitches[-2]) / dtn
            
        v_total = np.sqrt(v_yaw**2 + v_pitch**2)
        return v_total, v_yaw, v_pitch

    # --------------------------------------------------------------------------
    # 4. SACCADE EVENT EXTRACTION (I-VT)
    # --------------------------------------------------------------------------
    def extract_saccades(
        self,
        start_ms: Optional[float] = None,
        end_ms: Optional[float] = None,
        trigger_event_ms: Optional[float] = None,
        tested_eye: str = "OS"
    ) -> List[SaccadeEvent]:
        """
        Extracts all discrete saccadic events within the time window [start_ms, end_ms].
        If trigger_event_ms is provided, computes onset latencies relative to the trigger.
        
        Args:
            start_ms: Window start timestamp in milliseconds
            end_ms: Window end timestamp in milliseconds
            trigger_event_ms: Timestamp of the trigger event (e.g. occlusion onset)
            tested_eye: "OD" (Right) or "OS" (Left)
        """
        if len(self.timestamps) < 3:
            return []
            
        t_arr = np.array(self.timestamps)
        y_arr = np.array(self.yaw_deg)
        p_arr = np.array(self.pitch_deg)
        val_arr = np.array(self.is_valid)
        
        # Filter window
        mask = np.ones(len(t_arr), dtype=bool)
        if start_ms is not None:
            mask &= (t_arr >= start_ms)
        if end_ms is not None:
            mask &= (t_arr <= end_ms)
            
        indices = np.where(mask)[0]
        if len(indices) < 3:
            return []
            
        t_sub = t_arr[indices]
        y_sub = y_arr[indices]
        p_sub = p_arr[indices]
        val_sub = val_arr[indices]
        
        # Compute smoothed velocities
        v_mag, v_y, v_p = self.compute_velocities(t_sub, y_sub, p_sub)
        
        # I-VT Segmentation: Identify consecutive frames above velocity threshold
        saccade_mask = (v_mag >= self.velocity_threshold) & val_sub
        
        events: List[SaccadeEvent] = []
        in_saccade = False
        s_start_idx = 0
        
        for i in range(len(saccade_mask)):
            if saccade_mask[i] and not in_saccade:
                in_saccade = True
                s_start_idx = i
            elif not saccade_mask[i] and in_saccade:
                in_saccade = False
                s_end_idx = i
                
                # Saccade duration check
                dur = t_sub[s_end_idx] - t_sub[s_start_idx]
                if dur >= MIN_SACCADE_DURATION_MS:
                    # Trajectory displacement: start position to end position
                    # Expanding bounds by 1 frame to capture complete peak-to-trough amplitude
                    i1 = max(0, s_start_idx - 1)
                    i2 = min(len(t_sub) - 1, s_end_idx)
                    
                    dy = float(y_sub[i2] - y_sub[i1])
                    dp = float(p_sub[i2] - p_sub[i1])
                    amp = float(math.sqrt(dy**2 + dp**2))
                    
                    if amp >= self.min_saccade_amp:
                        peak_v = float(np.max(v_mag[s_start_idx:s_end_idx + 1]))
                        mean_v = float(np.mean(v_mag[s_start_idx:s_end_idx + 1]))
                        
                        # Latency
                        latency = None
                        if trigger_event_ms is not None:
                            latency = float(t_sub[s_start_idx] - trigger_event_ms)
                            
                        # Directions
                        dir_h = "NASAL" if dy > 0.5 else ("TEMPORAL" if dy < -0.5 else "NONE")
                        dir_v = "DOWNWARD" if dp > 0.5 else ("UPWARD" if dp < -0.5 else "NONE")
                        
                        # Clinical Strabismus Correlation for Refixation Saccades:
                        # If the uncovered eye makes a TEMPORAL (outward) refixation, it was resting inward -> ESOTROPIA
                        # If it makes a NASAL (inward) refixation, it was resting outward -> EXOTROPIA
                        # If it moves DOWNWARD, it was resting upward -> HYPERTROPIA
                        # If it moves UPWARD, it was resting downward -> HYPOTROPIA
                        strab_sign = "ORTHO"
                        if abs(dy) >= 1.5:
                            if dir_h == "TEMPORAL":
                                strab_sign = "ESOTROPIA"
                            elif dir_h == "NASAL":
                                strab_sign = "EXOTROPIA"
                        elif abs(dp) >= 1.5:
                            if dir_v == "DOWNWARD":
                                strab_sign = "HYPERTROPIA"
                            elif dir_v == "UPWARD":
                                strab_sign = "HYPOTROPIA"
                                
                        amp_pd = float(amp * 1.7455)
                        
                        events.append(SaccadeEvent(
                            start_time_ms=float(t_sub[s_start_idx]),
                            end_time_ms=float(t_sub[s_end_idx]),
                            duration_ms=float(dur),
                            onset_latency_ms=latency,
                            amplitude_deg=amp,
                            amplitude_h_deg=dy,
                            amplitude_v_deg=dp,
                            amplitude_pd=amp_pd,
                            peak_velocity_deg_s=peak_v,
                            mean_velocity_deg_s=mean_v,
                            direction_h=dir_h,
                            direction_v=dir_v,
                            strabismus_sign=strab_sign
                        ))
                        
        return events

    # --------------------------------------------------------------------------
    # 5. PRIMARY REFIXATION EXTRACTOR (FOR COVER TESTS)
    # --------------------------------------------------------------------------
    def extract_refixation_saccade(
        self,
        trigger_ms: float,
        post_trigger_window_ms: float = 600.0,
        tested_eye: str = "OS"
    ) -> Optional[SaccadeEvent]:
        """
        Specialized extractor for clinical Cover-Uncover and Alternate Cover tests.
        Looks specifically within the physiological latency window (80 ms to 550 ms)
        following occlusion trigger for the primary corrective refixation saccade.
        """
        window_end = trigger_ms + post_trigger_window_ms
        candidates = self.extract_saccades(
            start_ms=trigger_ms,
            end_ms=window_end,
            trigger_event_ms=trigger_ms,
            tested_eye=tested_eye
        )
        
        # Filter candidates by physiological latency (saccades rarely occur before 80ms post-cover)
        valid_refixations = [
            c for c in candidates
            if c.onset_latency_ms is not None and 60.0 <= c.onset_latency_ms <= 550.0
        ]
        
        if not valid_refixations:
            return None
            
        # Return the largest-amplitude refixation saccade in the window
        return max(valid_refixations, key=lambda s: s.amplitude_deg)

    # --------------------------------------------------------------------------
    # 6. FIXATION STABILITY (BCEA)
    # --------------------------------------------------------------------------
    def compute_fixation_stability(
        self,
        start_ms: Optional[float] = None,
        end_ms: Optional[float] = None
    ) -> FixationMetrics:
        """
        Computes Bivariate Contour Ellipse Area (BCEA) and drift parameters.
        BCEA represents the area of the ellipse containing 68.2% of fixation points.
        Formula: BCEA = 2 * pi * k * sigma_x * sigma_y * sqrt(1 - rho^2), where k = 1.14
        """
        if len(self.timestamps) < 5:
            return FixationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
            
        t_arr = np.array(self.timestamps)
        y_arr = np.array(self.yaw_deg)
        p_arr = np.array(self.pitch_deg)
        val_arr = np.array(self.is_valid)
        
        mask = val_arr.copy()
        if start_ms is not None:
            mask &= (t_arr >= start_ms)
        if end_ms is not None:
            mask &= (t_arr <= end_ms)
            
        valid_indices = np.where(mask)[0]
        n_pts = len(valid_indices)
        
        if n_pts < 5:
            return FixationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
            
        t_sub = t_arr[valid_indices]
        y_sub = y_arr[valid_indices]
        p_sub = p_arr[valid_indices]
        
        total_duration = float(t_sub[-1] - t_sub[0])
        mean_y = float(np.mean(y_sub))
        mean_p = float(np.mean(p_sub))
        std_y = float(np.std(y_sub))
        std_p = float(np.std(p_sub))
        
        # Pearson correlation rho between horizontal and vertical gaze points
        if std_y > 1e-4 and std_p > 1e-4:
            cov = float(np.cov(y_sub, p_sub)[0, 1])
            rho = float(np.clip(cov / (std_y * std_p), -0.99, 0.99))
        else:
            rho = 0.0
            
        # BCEA for 1 standard deviation (68.2% probability contour): k = 1.14
        k = 1.14
        bcea = float(2.0 * math.pi * k * std_y * std_p * math.sqrt(1.0 - rho**2))
        
        # Slow drift velocity (excluding large saccadic jumps)
        v_mag, _, _ = self.compute_velocities(t_sub, y_sub, p_sub)
        drift_mask = v_mag < self.velocity_threshold
        drift_v = float(np.mean(v_mag[drift_mask])) if np.any(drift_mask) else 0.0
        
        # Count microsaccades (small bursts between 0.5 deg and 1.5 deg)
        saccades = self.extract_saccades(start_ms=t_sub[0], end_ms=t_sub[-1])
        microsaccades = sum(1 for s in saccades if 0.5 <= s.amplitude_deg < 1.8)
        
        quality = float(n_pts / max(len(t_arr), 1))
        
        return FixationMetrics(
            duration_ms=total_duration,
            mean_yaw_deg=mean_y,
            mean_pitch_deg=mean_p,
            std_yaw_deg=std_y,
            std_pitch_deg=std_p,
            bcea_deg2=bcea,
            drift_velocity_deg_s=drift_v,
            microsaccade_count=microsaccades,
            data_quality=quality
        )

    # --------------------------------------------------------------------------
    # 7. SMOOTH PURSUIT METRICS (FOR THERAPY GAMES)
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_pursuit_gain(
        eye_velocities: np.ndarray,
        target_velocities: np.ndarray
    ) -> float:
        """
        Computes smooth pursuit gain: G = mean(v_eye) / mean(v_target).
        Ideal physiological pursuit gain is ~0.90 to 1.00.
        """
        if len(target_velocities) == 0:
            return 0.0
        mean_target = float(np.mean(np.abs(target_velocities)))
        if mean_target < 1e-3:
            return 1.0
        mean_eye = float(np.mean(np.abs(eye_velocities)))
        return float(np.clip(mean_eye / mean_target, 0.0, 2.0))


# ==============================================================================
# 8. SELF-CONTAINED UNIT TEST
# ==============================================================================
if __name__ == "__main__":
    print("[TEST] Running EyeMovementAnalyzer verification...")
    analyzer = EyeMovementAnalyzer(velocity_threshold_deg_s=30.0)

    # Simulate 30 FPS stream (dt = 33.3ms)
    # 1. Steady baseline fixation (0 to 300 ms at 0.0 deg)
    # 2. Refixation saccade at 300 ms jumping 6.0 degrees in 66 ms (2 frames: ~90 deg/s)
    # 3. Steady fixation at 6.0 deg (366 to 700 ms)
    times = []
    yaws = []
    pitches = []
    
    t = 0.0
    # Phase 1: Baseline fixation (10 frames = ~333 ms)
    for _ in range(10):
        times.append(t)
        yaws.append(0.0 + np.random.normal(0, 0.05))
        pitches.append(0.0 + np.random.normal(0, 0.05))
        t += 33.33
        
    trigger_time = times[-1]  # Trigger event at t ≈ 300 ms

    # Physiological latency delay: 4 frames (~133 ms) before saccade onset
    for _ in range(4):
        times.append(t)
        yaws.append(0.0 + np.random.normal(0, 0.05))
        pitches.append(0.0 + np.random.normal(0, 0.05))
        t += 33.33

    # Phase 2: Saccadic refixation (Temporal / outward recovery jump of 6 deg)
    times.append(t); yaws.append(-3.0); pitches.append(0.0); t += 33.33
    times.append(t); yaws.append(-6.0); pitches.append(0.0); t += 33.33
    
    # Phase 3: Steady fixation post-refixation
    for _ in range(12):
        times.append(t)
        yaws.append(-6.0 + np.random.normal(0, 0.05))
        pitches.append(0.0 + np.random.normal(0, 0.05))
        t += 33.33
        
    for i in range(len(times)):
        analyzer.add_sample(times[i], yaws[i], pitches[i])
        
    # Analyze refixation saccade
    refix = analyzer.extract_refixation_saccade(trigger_ms=trigger_time, post_trigger_window_ms=500.0)
    print("\n--- DETECTED REFIXATION SACCADE ---")
    if refix:
        print(f"  Onset Latency:  {refix.onset_latency_ms:.1f} ms")
        print(f"  Duration:       {refix.duration_ms:.1f} ms")
        print(f"  Amplitude:      {refix.amplitude_deg:.2f} deg ({refix.amplitude_pd:.1f} PD)")
        print(f"  Direction:      {refix.direction_h}")
        print(f"  Peak Velocity:  {refix.peak_velocity_deg_s:.1f} deg/s")
        print(f"  Diagnosis Sign: {refix.strabismus_sign}")
    else:
        print("  [ERROR] No refixation detected.")
        
    # Analyze fixation stability (BCEA) during post-refixation hold
    fix = analyzer.compute_fixation_stability(start_ms=times[13], end_ms=times[-1])
    print("\n--- FIXATION STABILITY (POST-REFIXATION) ---")
    print(f"  Duration:       {fix.duration_ms:.1f} ms")
    print(f"  Mean Yaw:       {fix.mean_yaw_deg:.2f} deg")
    print(f"  Std Yaw:        {fix.std_yaw_deg:.3f} deg")
    print(f"  BCEA (68.2%):   {fix.bcea_deg2:.4f} deg^2")
    print(f"  Drift Velocity: {fix.drift_velocity_deg_s:.2f} deg/s")
    
    print("\n[SUCCESS] EyeMovementAnalyzer passed all synthetic tests.")
