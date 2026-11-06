# Eye Gaze & Strabismus Screening

Webcam-based eye tracking (gaze estimation), real-time ocular misalignment screening, and automated clinical Cover-Uncover testing.

## Directory Structure

```text
eye-gaze-strabismus/
├── core/
│   ├── cover_test.py               # Clinical automated Cover-Uncover test (Tropia vs Phoria)
│   ├── screen_strabismus.py        # Real-time bilateral ocular alignment (9-cardinal gaze protocol)
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
│   ├── samples/                    # Sample test video clips
│   └── results/                    # Auto-saved diagnostic reports & curves
├── models/
│   ├── face_landmarker.task        # MediaPipe face landmark model
│   ├── gaze_model_p00.pth          # Pretrained GazeCNN backbone (MPIIGaze)
│   └── strabismus_classifier.pth   # Trained Siamese strabismus classifier
├── strabismus-video/               # Clinical video samples for offline testing
├── requirements.txt
└── STRABISMUS_SCREENING_ARCHITECTURE_AND_EVALUATION_REPORT.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

### 1. Clinical Automated Cover-Uncover Test
```bash
# On live webcam:
python core/cover_test.py

# On sample video:
python core/cover_test.py --video "strabismus-video/vidssave.com Cover Test 480P.mp4"

# Another example on sample video
python core/cover_test.py --video "diagnostics/samples/02_esotropia.mp4"
```

### 2. Real-Time Bilateral Strabismus Screening (9-Cardinal Protocol)
```bash
python core/screen_strabismus.py
```

### 3. Deep Learning Classifier (Dataset Explorer & Training)
```bash
# Evaluate trained classifier on Kaggle dataset images:
python diagnostics/test_strabismus_images.py

# Train Siamese strabismus classifier:
python core/train_strabismus_net.py
```

### 4. Live Gaze & Screen Calibration
```bash
python core/gaze_to_screen.py
```

## Strabismus Diagnostic Classes

`NORMAL | ESOTROPIA | EXOTROPIA | HYPERTROPIA | HYPOTROPIA`
