"""
test_strabismus_images.py - Strabismus Dataset Diagnostic & Image Testing Tool
===============================================================================
BTP Project: Webcam-Based Eye Tracking & Visual Training for Strabismus Screening

This script connects the project's core GazeCNN neural network and Siamese
Disparity Network (trained on the Kaggle Strabismus dataset) directly to static images:

Dual-Diagnostic Architecture:
  1. AI Model Diagnosis: Uses DualGazeStrabismusNet (models/strabismus_classifier.pth)
     which was pretrained with your GazeCNN backbone (gaze_model_p00.pth).
     Outputs: Predicted Class [NORMAL, ESOTROPIA, EXOTROPIA, HYPERTROPIA, HYPOTROPIA]
     and Confidence Percentage (%).
  2. Clinical Geometric Physics: Calculates physical pupil ratios and inter-ocular
     disparity in clinical Prism Diopters (Delta PD).

Interactive Controls:
  [n] Next image in folder
  [p] Previous image
  [r] Random image from current category
  [1] Switch to NORMAL
  [2] Switch to ESOTROPIA
  [3] Switch to EXOTROPIA
  [4] Switch to HYPERTROPIA
  [5] Switch to HYPOTROPIA
  [q] Quit
"""

import os
import sys
import argparse
import random
import cv2
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from core.train_strabismus_net import DualGazeStrabismusNet, CLASSES

DEFAULT_DATASET_PATHS = [
    r"C:\Users\HP\OneDrive\Documents\BTP\datasets\STRABISMUS",
    r"C:\Users\HP\Documents\BTP\datasets\STRABISMUS",
    os.path.join(PROJECT_ROOT, "datasets", "STRABISMUS"),
]

def find_dataset_root():
    for p in DEFAULT_DATASET_PATHS:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

def find_classifier_path():
    candidates = [
        os.path.join(PROJECT_ROOT, "models", "strabismus_classifier.pth"),
        os.path.join(PROJECT_ROOT, "strabismus_classifier.pth"),
        "models/strabismus_classifier.pth",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def detect_pupil_center(eye_bgr):
    """Locates pupil center in an eye crop."""
    gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    h_e, w_e = gray.shape

    margin_x = int(w_e * 0.12)
    margin_y = int(h_e * 0.12)
    roi = blurred[margin_y:h_e - margin_y, margin_x:w_e - margin_x]

    p_thresh = np.percentile(roi, 8)
    _, binary = cv2.threshold(roi, p_thresh, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    M = cv2.moments(binary)
    if M["m00"] > 10:
        cx = int(M["m10"] / M["m00"]) + margin_x
        cy = int(M["m01"] / M["m00"]) + margin_y
    else:
        min_val, _, min_loc, _ = cv2.minMaxLoc(roi)
        cx = min_loc[0] + margin_x
        cy = min_loc[1] + margin_y

    ratio_x = cx / float(w_e)
    ratio_y = cy / float(h_e)
    return (cx, cy), ratio_x, ratio_y


def analyze_strabismus_image(img, model, device):
    """
    Analyzes an eye-strip image with both AI Deep Learning model and Geometric Disparity.
    """
    h_orig, w_orig, _ = img.shape
    mid = w_orig // 2

    r_eye_bgr = img[:, :mid].copy()
    l_eye_bgr = img[:, mid:].copy()

    # 1. Prepare inputs for DualGazeStrabismusNet
    r_gray = cv2.resize(cv2.cvtColor(r_eye_bgr, cv2.COLOR_BGR2GRAY), (60, 36))
    l_gray = cv2.resize(cv2.cvtColor(l_eye_bgr, cv2.COLOR_BGR2GRAY), (60, 36))
    # Mirror left eye horizontally for canonical eye symmetry (MPIIGaze standard)
    l_flipped = cv2.flip(l_gray, 1)

    r_t = torch.from_numpy(r_gray).float().unsqueeze(0).unsqueeze(0) / 255.0 * 2.0 - 1.0
    l_t = torch.from_numpy(l_flipped).float().unsqueeze(0).unsqueeze(0) / 255.0 * 2.0 - 1.0

    # 2. AI Model Inference
    with torch.no_grad():
        logits = model(r_t.to(device), l_t.to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    pred_conf = float(probs[pred_idx] * 100.0)

    # 3. Geometric Disparity calculation (Physical confirmation)
    (r_px, r_py), r_rx, r_ry = detect_pupil_center(r_eye_bgr)
    (l_px, l_py), l_rx, l_ry = detect_pupil_center(l_eye_bgr)

    r_nasal = r_rx
    l_nasal = 1.0 - l_rx
    disparity_h = (r_nasal - l_nasal)
    disparity_v = (r_ry - l_ry)

    deg_h = disparity_h * 90.0
    deg_v = disparity_v * 60.0
    pd_h = deg_h * 1.75
    pd_v = deg_v * 1.75

    # Color selection
    colors = {
        "NORMAL": (0, 230, 80),        # Green
        "ESOTROPIA": (0, 140, 255),    # Orange/Red
        "EXOTROPIA": (255, 120, 0),    # Blue
        "HYPERTROPIA": (255, 0, 255),  # Magenta
        "HYPOTROPIA": (0, 255, 255),   # Yellow
    }
    badge_color = colors.get(pred_class, (0, 230, 80))

    # 4. Compose UI Visualization
    canvas = np.zeros((650, 920, 3), dtype=np.uint8)
    canvas[:] = (20, 22, 26)

    # Header
    cv2.rectangle(canvas, (0, 0), (920, 50), (30, 34, 40), -1)
    cv2.putText(canvas, "BTP CLINICAL STRABISMUS IMAGE DIAGNOSTICIAN", (20, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2)

    # Render eye crops with annotations
    card_w, card_h = 420, 180
    r_disp = cv2.resize(r_eye_bgr, (card_w, card_h))
    l_disp = cv2.resize(l_eye_bgr, (card_w, card_h))

    # Scale pupil coords
    r_cx_s = int(r_px * (card_w / float(mid)))
    r_cy_s = int(r_py * (card_h / float(h_orig)))
    l_cx_s = int(l_px * (card_w / float(w_orig - mid)))
    l_cy_s = int(l_py * (card_h / float(h_orig)))

    # Draw pupil crosshairs
    cv2.drawMarker(r_disp, (r_cx_s, r_cy_s), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.circle(r_disp, (r_cx_s, r_cy_s), 14, (0, 255, 255), 2)
    cv2.drawMarker(l_disp, (l_cx_s, l_cy_s), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.circle(l_disp, (l_cx_s, l_cy_s), 14, (0, 255, 255), 2)

    # Place cards on canvas
    y_cards = 100
    canvas[y_cards:y_cards + card_h, 25:25 + card_w] = r_disp
    cv2.rectangle(canvas, (25, y_cards), (25 + card_w, y_cards + card_h), (0, 200, 255), 2)
    cv2.putText(canvas, "RIGHT EYE (OD)", (25, y_cards - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    cv2.putText(canvas, f"Pupil Ratio: {r_nasal:.2f}", (25, y_cards + card_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    canvas[y_cards:y_cards + card_h, 475:475 + card_w] = l_disp
    cv2.rectangle(canvas, (475, y_cards), (475 + card_w, y_cards + card_h), (0, 200, 255), 2)
    cv2.putText(canvas, "LEFT EYE (OS)", (475, y_cards - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    cv2.putText(canvas, f"Pupil Ratio: {l_nasal:.2f}", (475, y_cards + card_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Diagnosis Display Card
    y_diag = 350
    cv2.rectangle(canvas, (25, y_diag), (895, y_diag + 185), (28, 30, 36), -1)
    cv2.rectangle(canvas, (25, y_diag), (895, y_diag + 185), badge_color, 2)

    diag_header = f"AI PREDICTION: {pred_class} ({pred_conf:.1f}% Confidence)"
    cv2.putText(canvas, diag_header, (45, y_diag + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, badge_color, 2)

    # All class probabilities bar
    prob_str = " | ".join([f"{c[:4]}: {probs[i]*100:.0f}%" for i, c in enumerate(CLASSES)])
    cv2.putText(canvas, f"Class Distribution: {prob_str}", (45, y_diag + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    metrics_str = f"Clinical Geometry: Delta H = {pd_h:+5.1f} Delta PD ({deg_h:+4.1f} deg) | Delta V = {pd_v:+5.1f} Delta PD ({deg_v:+4.1f} deg)"
    cv2.putText(canvas, metrics_str, (45, y_diag + 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2)

    backbone_str = "Backbone: Dual-Gaze Siamese Net (Transfer Learned from gaze_model_p00.pth)"
    cv2.putText(canvas, backbone_str, (45, y_diag + 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Footer HUD Instructions
    y_foot = 600
    cv2.rectangle(canvas, (0, y_foot - 15), (920, 650), (16, 18, 22), -1)
    hud_txt = "[n] Next  |  [p] Prev  |  [r] Random  |  [1-5] Categories  |  [q] Quit"
    cv2.putText(canvas, hud_txt, (35, y_foot + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)

    diag_dict = {
        "pred_class": pred_class,
        "pred_conf": pred_conf,
        "probs": probs,
        "pd_h": pd_h,
        "pd_v": pd_v,
        "deg_h": deg_h,
        "deg_v": deg_v,
    }
    return canvas, diag_dict


def main():
    parser = argparse.ArgumentParser(description="Test strabismus diagnostic on Kaggle images")
    parser.add_argument("--dataset", type=str, default=None, help="Path to STRABISMUS dataset")
    parser.add_argument("--image", type=str, default=None, help="Path to a single test image")
    args = parser.parse_args()

    dataset_root = args.dataset or find_dataset_root()
    if not dataset_root and not args.image:
        print("ERROR: Dataset directory not found. Please provide --dataset <path>")
        return

    # Load DualGazeStrabismusNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualGazeStrabismusNet().to(device)
    model_path = find_classifier_path()
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[MODEL] Successfully loaded trained Strabismus classifier: {model_path}")
    else:
        print("[WARNING] strabismus_classifier.pth not found. Please run train_strabismus_net.py first.")
    model.eval()

    # Mode 1: Single image provided
    if args.image:
        if not os.path.exists(args.image):
            print(f"ERROR: Image not found: {args.image}")
            return
        img = cv2.imread(args.image)
        canvas, diag = analyze_strabismus_image(img, model, device)
        print(f"\n[DIAGNOSIS for {os.path.basename(args.image)}]:")
        print(f"  AI Prediction: {diag['pred_class']} ({diag['pred_conf']:.1f}% Confidence)")
        print(f"  Disparity H  : {diag['pd_h']:+.1f} Delta PD ({diag['deg_h']:+.1f} deg)")
        print(f"  Disparity V  : {diag['pd_v']:+.1f} Delta PD ({diag['deg_v']:+.1f} deg)")
        cv2.imshow("Strabismus Image Diagnosis", canvas)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Mode 2: Interactive Dataset Explorer
    print("\n" + "=" * 70)
    print("STRABISMUS DATASET INTERACTIVE DIAGNOSTIC EXPLORER")
    print(f"Dataset: {dataset_root}")
    print("Controls:")
    print("  [n] Next image       [p] Previous image    [r] Random image")
    print("  [1] NORMAL           [2] ESOTROPIA         [3] EXOTROPIA")
    print("  [4] HYPERTROPIA      [5] HYPOTROPIA        [q] Quit")
    print("=" * 70)

    cat_idx = 1  # Start on ESOTROPIA
    current_cat = CLASSES[cat_idx]
    cat_dir = os.path.join(dataset_root, current_cat)
    files = sorted([f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    file_idx = 0

    cv2.namedWindow("Strabismus Dataset Diagnosis", cv2.WINDOW_AUTOSIZE)

    while True:
        cat_dir = os.path.join(dataset_root, current_cat)
        files = sorted([f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        if not files:
            print(f"No images found in {cat_dir}")
            break

        file_idx = max(0, min(file_idx, len(files) - 1))
        img_name = files[file_idx]
        img_path = os.path.join(cat_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            file_idx = (file_idx + 1) % len(files)
            continue

        canvas, diag = analyze_strabismus_image(img, model, device)

        # Ground Truth comparison banner at top
        is_correct = (diag["pred_class"] == current_cat)
        gt_color = (0, 230, 80) if is_correct else (0, 140, 255)
        status_txt = "MATCH" if is_correct else "MISMATCH"
        banner = f"Ground Truth: {current_cat} | File: {img_name} ({file_idx+1}/{len(files)}) | AI Result: [{status_txt}]"
        cv2.putText(canvas, banner, (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gt_color, 2)

        cv2.imshow("Strabismus Dataset Diagnosis", canvas)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('n'):
            file_idx = (file_idx + 1) % len(files)
        elif key == ord('p'):
            file_idx = (file_idx - 1 + len(files)) % len(files)
        elif key == ord('r'):
            file_idx = random.randint(0, len(files) - 1)
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            cat_idx = (key - ord('1'))
            current_cat = CLASSES[cat_idx]
            file_idx = 0
            print(f"Switched to category: {current_cat}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
