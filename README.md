# Eye Gaze & Strabismus Screening

Webcam-based eye tracking (gaze estimation) and real-time strabismus screening system.

## Files

| File | What it does | How to run |
|---|---|---|
| `gaze_to_screen.py` | 9-point calibration + live gaze tracking mapped to screen coordinates | `python gaze_to_screen.py` |
| `screen_strabismus.py` | Real-time bilateral eye alignment & strabismus screening (5-class) | `python screen_strabismus.py` |
| `train_strabismus_net.py` | Train the Siamese strabismus classifier on your dataset | `python train_strabismus_net.py` |
| `normalize.py` | Eye normalization utility — imported internally, do not run directly | — |

## Models (place in `models/` folder)

| File | Description |
|---|---|
| `models/face_landmarker.task` | MediaPipe face landmark model |
| `models/gaze_model_p00.pth` | Pretrained GazeCNN gaze model |
| `models/strabismus_classifier.pth` | Trained Siamese strabismus classifier |

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
# Live gaze tracking with screen calibration
python gaze_to_screen.py

# Real-time strabismus screening
python screen_strabismus.py

# Train strabismus model (set dataset path inside the script)
python train_strabismus_net.py
```

## Strabismus Classes

`NORMAL | ESOTROPIA | EXOTROPIA | HYPERTROPIA | HYPOTROPIA`
