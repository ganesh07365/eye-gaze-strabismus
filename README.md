# Eye Gaze & Strabismus Screening

Webcam-based dual-eye tracking, automated clinical strabismus screening (Cover-Uncover, Alternate Cover, and 9-Cardinal Gaze tests), and clinical triage synthesis in standard Prism Diopters ($\Delta\text{PD}$).

## Directory Structure

```text
eye-gaze-strabismus/
├── core/
│   ├── per_eye_tracker.py          # Bilateral dual-eye 3D tracking, metric scaling & occlusion sensing
│   ├── eye_movement_analyzer.py    # I-VT saccade segmentation & BCEA fixation stability metrics
│   ├── cover_test.py               # Automated condition-gated Cover-Uncover test (Tropia vs Phoria)
│   ├── alternate_cover_test.py     # Automated Alternate Cover Test (Total dissociated deviation)
│   ├── screen_strabismus.py        # 9-Cardinal Gaze motility exam (Comitant vs Incomitant spread)
│   ├── result_engine.py            # Step 3 clinical triage engine (NORMAL / BORDERLINE / REFER)
│   ├── train_strabismus_net.py     # Siamese strabismus classifier training pipeline
│   ├── gaze_to_screen.py           # 9-point calibration + gaze-to-screen tracking
│   ├── live_gaze.py                # Standalone webcam gaze tracking
│   ├── live_gaze_dual.py           # Dual-camera setup gaze tracking
│   └── normalize.py                # Eye normalization utility (MPIIGaze standard)
├── diagnostics/
│   ├── test_strabismus_images.py   # Diagnostic evaluator on Kaggle dataset images
│   ├── diagnose_iris_landmarks.py  # Iris and canthal landmark validation
│   ├── diagnose_vertical.py        # Vertical alignment diagnostic
│   ├── diagnose_vertical_gaze.py   # Vertical gaze ratio diagnostics
│   ├── test_crop_alignment.py      # Eye crop alignment test
│   ├── test_facemesh.py            # MediaPipe face mesh validator
│   ├── test_headpose.py            # Head pose estimation test
│   ├── samples/                    # Benchmark clinical clips (01_normal..04_hypertropia.mp4)
│   └── results/                    # Auto-saved diagnostic JSON/TXT reports & plots
├── models/
│   ├── face_landmarker.task        # MediaPipe face landmark model
│   ├── gaze_model_p00.pth          # Pretrained GazeCNN backbone (MPIIGaze)
│   └── strabismus_classifier.pth   # Trained Siamese strabismus classifier
├── report/
│   ├── BTP_evaluation2.tex         # Evaluation 2 technical report source
│   ├── BTP_evaluation2.pdf         # Compiled Evaluation 2 report
│   └── figures/                    # Architecture diagrams and diagnostic plots
├── strabismus-video/               # Local clinical video samples for offline testing
├── BTP_Evaluation2_Strabismus_Screening.pptx
├── PROJECT_BLUEPRINT.md
├── requirements.txt
└── STRABISMUS_SCREENING_ARCHITECTURE_AND_EVALUATION_REPORT.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

### 1. Cover-Uncover Test (Step 2 - Module 2.2)
```bash
# Live webcam:
python core/cover_test.py

# Offline clinical sample videos:
python core/cover_test.py --video diagnostics/samples/01_normal.mp4
python core/cover_test.py --video diagnostics/samples/02_esotropia.mp4
python core/cover_test.py --video diagnostics/samples/03_exotropia.mp4
python core/cover_test.py --video diagnostics/samples/04_hypertropia.mp4
```

### 2. Alternate Cover Test (Step 2 - Module 2.3)
```bash
python core/alternate_cover_test.py
```

### 3. 9-Cardinal Gaze Motility Exam (Step 2 - Module 2.4)
```bash
python core/screen_strabismus.py
```

### 4. Clinical Triage Synthesis (Step 3)
```bash
# Run automated 6-case verification suite:
python core/result_engine.py --test-suite

# Synthesize triage verdict from the latest session reports in diagnostics/results/:
python core/result_engine.py --latest
```

### 5. Deep Learning Classifier & Legacy Gaze Tracking
```bash
# Evaluate trained classifier on Kaggle dataset images:
python diagnostics/test_strabismus_images.py

# Train Siamese strabismus classifier:
python core/train_strabismus_net.py

# Live gaze & screen calibration:
python core/gaze_to_screen.py
```

## Strabismus Diagnostic Classes

`NORMAL | ESOTROPIA | EXOTROPIA | HYPERTROPIA | HYPOTROPIA`
