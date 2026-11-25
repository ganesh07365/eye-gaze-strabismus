import os
import copy
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

TEMPLATE_PATH = r"C:\Users\Gnana Teja Varma\Downloads\Project_Presentation_Template.pptx"
OUTPUT_PATH = r"C:\Users\Gnana Teja Varma\Downloads\BTP_Evaluation2_Strabismus_Screening.pptx"
WORKSPACE_OUTPUT = r"C:\Users\Gnana Teja Varma\eye-gaze-strabismus\BTP_Evaluation2_Strabismus_Screening.pptx"

IMG1_PATH = r"C:\Users\Gnana Teja Varma\.gemini\antigravity\brain\01b3c5c6-758c-492c-bbf2-9923b2719d17\.user_uploaded\media_1795491546092_e5122995.jpg"
IMG2_PATH = r"C:\Users\Gnana Teja Varma\.gemini\antigravity\brain\01b3c5c6-758c-492c-bbf2-9923b2719d17\.user_uploaded\media_1795491546097_8880a000.jpg"

def remove_shape(shape):
    sp = shape._element
    sp.getparent().remove(sp)

def set_text_items(tf, items, default_size=Pt(15), default_space=Pt(4)):
    """
    items is a list of dicts:
    [{'text': '...', 'size': Pt(16), 'bold': True, 'space_after': Pt(4), 'color': RGBColor(...)}]
    """
    tf.clear()
    for idx, item in enumerate(items):
        if idx == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        
        p.text = item.get('text', '')
        p.font.size = item.get('size', default_size)
        p.font.bold = item.get('bold', False)
        p.space_after = item.get('space_after', default_space)
        if 'color' in item:
            p.font.color.rgb = item['color']

def main():
    prs = Presentation(TEMPLATE_PATH)
    print(f"Loaded template with {len(prs.slides)} slides.")

    # ----------------------------------------------------
    # Step 1: Duplicate Slide 6 (index 5) to create Slide 7 (index 6)
    # ----------------------------------------------------
    slide_to_copy = prs.slides[5]
    blank_layout = prs.slide_layouts[0]
    new_slide = prs.slides.add_slide(blank_layout)

    # Clean layout placeholders from new slide
    for ph in list(new_slide.placeholders):
        remove_shape(ph)

    # Copy shapes
    for shape in slide_to_copy.shapes:
        el = copy.deepcopy(shape._element)
        new_slide.shapes._spTree.append(el)

    # Reorder new slide to be at index 6 (Slide 7)
    sldIdLst = prs.slides._sldIdLst
    new_sldId = sldIdLst[-1]
    sldIdLst.insert(6, new_sldId)
    print(f"Total slides after duplication: {len(prs.slides)}")

    # ----------------------------------------------------
    # Slide 1: Title Slide
    # ----------------------------------------------------
    slide1 = prs.slides[0]
    to_remove_s1 = []
    for s in slide1.shapes:
        if s.name in ['Agenda card', 'Text 3', 'Text 4']:
            to_remove_s1.append(s)
    for s in to_remove_s1:
        remove_shape(s)

    # Widen title & subtitle placeholders
    for s in slide1.shapes:
        if s.name == 'Text 0':
            s.width = Inches(11.5)
            if s.top < Inches(2.0):  # Title box
                tf = s.text_frame
                tf.word_wrap = True
                set_text_items(tf, [
                    {'text': 'Webcam-Based Strabismus (Squint) Screening and Vision Therapy System Using Dual-Eye Tracking', 
                     'size': Pt(34), 'bold': True, 'space_after': Pt(12)}
                ])
            else:  # Subtitle box
                tf = s.text_frame
                tf.word_wrap = True
                set_text_items(tf, [
                    {'text': 'Students:', 'size': Pt(18), 'bold': True, 'space_after': Pt(4)},
                    {'text': '  J Gnana Teja Varma  |  Roll No: S20240010088', 'size': Pt(17), 'bold': False, 'space_after': Pt(2)},
                    {'text': '  V Sundhara Sai Ganesh  |  Roll No: S20240010250', 'size': Pt(17), 'bold': False, 'space_after': Pt(10)},
                    {'text': 'Project Type: Bachelor of Technology Project (BTP)', 'size': Pt(17), 'bold': False, 'space_after': Pt(4)},
                    {'text': 'Evaluation: Second Evaluation (Completed up to Step 3)', 'size': Pt(17), 'bold': True, 'space_after': Pt(4)},
                    {'text': 'Project Mentor: Dr. Pavan Kumar B N', 'size': Pt(17), 'bold': False, 'space_after': Pt(4)},
                ])

    # Remove all "Time chip" shapes across Slides 2 to 10
    for i in range(1, len(prs.slides)):
        sl = prs.slides[i]
        to_del = [sh for sh in sl.shapes if sh.name == 'Time chip']
        for sh in to_del:
            remove_shape(sh)

    # Helper function to find shapes by name on a slide
    def get_shapes_dict(slide):
        d = {}
        text_0_list = []
        for sh in slide.shapes:
            if sh.name == 'Text 0':
                text_0_list.append(sh)
            else:
                d[sh.name] = sh
        # text_0_list[0] is typically header title, text_0_list[1] is left body placeholder
        d['HeaderTitle'] = text_0_list[0] if len(text_0_list) > 0 else None
        d['LeftBody'] = text_0_list[1] if len(text_0_list) > 1 else None
        return d

    # ----------------------------------------------------
    # Slide 2: 1 Introduction
    # ----------------------------------------------------
    s2 = prs.slides[1]
    s2_dict = get_shapes_dict(s2)
    s2_dict['HeaderTitle'].text_frame.text = "Introduction"
    s2_dict['Text 3'].text_frame.text = "Problem Statement and Motivation"
    
    set_text_items(s2_dict['LeftBody'].text_frame, [
        {'text': '• The Problem in Simple Words:', 'size': Pt(16), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Strabismus (squint) occurs when both eyes do not look in the same direction. If untreated in childhood, the brain ignores the misaligned eye, causing permanent lazy eye (amblyopia) and permanent loss of 3D depth perception.', 'size': Pt(14), 'space_after': Pt(10)},
        {'text': '• Who is Affected & Why it Matters:', 'size': Pt(16), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Affects 2% to 4% of children globally. Regular mass screening is absent in rural and semi-urban communities because current tests require specialist pediatric ophthalmologists, optical prism bars, and manual occluders.', 'size': Pt(14), 'space_after': Pt(10)},
        {'text': '• Project Objective:', 'size': Pt(16), 'bold': True, 'space_after': Pt(2)},
        {'text': '   To build a low-cost, software-only screening system using a standard laptop webcam (720p, 30 FPS) that tracks both eyes independently and automates clinical eye tests in standard medical units (Prism Diopters).', 'size': Pt(14), 'space_after': Pt(4)},
    ], default_size=Pt(14))

    s2_dict['Text 6'].text_frame.text = "Worked Example: Real-World Screening"
    set_text_items(s2_dict['Text 7'].text_frame, [
        {'text': 'Clinical Case Scenario:', 'size': Pt(15), 'bold': True, 'space_after': Pt(4)},
        {'text': 'A 6-year-old school child has a subtle inward eye turn that parents cannot easily spot.', 'size': Pt(14), 'space_after': Pt(8)},
        {'text': '• Standard Scenario:', 'size': Pt(14), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Goes undetected until age 9, causing irreversible lazy eye and permanent loss of 3D vision.', 'size': Pt(13), 'space_after': Pt(8)},
        {'text': '• Our Webcam System:', 'size': Pt(14), 'bold': True, 'space_after': Pt(2)},
        {'text': '   The child looks at the laptop for 20 seconds. The software measures a 26.5 Prism Diopter deviation and immediately alerts: "REFER to eye specialist".', 'size': Pt(13), 'space_after': Pt(4)},
    ], default_size=Pt(13))

    # ----------------------------------------------------
    # Slide 3: 2 Literature Review
    # ----------------------------------------------------
    s3 = prs.slides[2]
    s3_dict = get_shapes_dict(s3)
    s3_dict['HeaderTitle'].text_frame.text = "Literature Review"
    s3_dict['Text 3'].text_frame.text = "Related Work & The Research Gap"

    set_text_items(s3_dict['LeftBody'].text_frame, [
        {'text': '• Clinical Prism Cover Tests (Gold Standard):', 'size': Pt(15), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Highly accurate, but requires trained pediatric specialists and manual prism bars. Impractical for mass school screening.', 'size': Pt(13), 'space_after': Pt(8)},
        {'text': '• Commercial Eye-Trackers (Tobii, EyeLink):', 'size': Pt(15), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Use dedicated infrared corneal reflection. High accuracy, but cost $1,000 to $5,000+ with proprietary hardware.', 'size': Pt(13), 'space_after': Pt(8)},
        {'text': '• Standard AI Gaze CNNs (MPIIGaze, GazeCapture):', 'size': Pt(15), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Merge both eyes into a single screen coordinate. Because they assume normal binocular vision, they are completely blind to squint.', 'size': Pt(13), 'space_after': Pt(8)},
        {'text': '• The Research Gap We Address:', 'size': Pt(15), 'bold': True, 'space_after': Pt(2)},
        {'text': '   No existing open-source webcam system tracks both eyes independently in real time to automate clinical Cover Tests in standard ophthalmic units (Prism Diopters) on normal laptop hardware.', 'size': Pt(13), 'space_after': Pt(4)},
    ], default_size=Pt(13))

    s3_dict['Text 6'].text_frame.text = "Worked Example: Why Existing AI Fails"
    set_text_items(s3_dict['Text 7'].text_frame, [
        {'text': 'Example: Left eye straight (0°), Right eye turned inward (15°):', 'size': Pt(14), 'bold': True, 'space_after': Pt(8)},
        {'text': '• Standard Deep Gaze Model:', 'size': Pt(14), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Averages both eyes together and predicts a normal gaze point in the center of the screen (Misses the squint completely!).', 'size': Pt(13), 'space_after': Pt(10)},
        {'text': '• Our Dual-Eye Model:', 'size': Pt(14), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Measures Left = 0° and Right = -15° independently. Reports a 26.2 Prism Diopter difference and correctly diagnoses inward strabismus.', 'size': Pt(13), 'space_after': Pt(4)},
    ], default_size=Pt(13))

    # ----------------------------------------------------
    # Slide 4: 3 Quick Recap
    # ----------------------------------------------------
    s4 = prs.slides[3]
    s4_dict = get_shapes_dict(s4)
    s4_dict['HeaderTitle'].text_frame.text = "Quick Recap"
    s4_dict['Text 3'].text_frame.text = "Progress from Previous Evaluation & Mentor Feedback"

    set_text_items(s4_dict['LeftBody'].text_frame, [
        {'text': '• What We Showed in Evaluation 1:', 'size': Pt(15), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Built an initial webcam gaze-tracking pipeline using MediaPipe and a custom ResNet-18 model trained on MPIIFaceGaze, predicting single-point screen cursor gaze with a dwell-time test.', 'size': Pt(13), 'space_after': Pt(8)},
        {'text': '• Mentor Feedback (Dr. Pavan Kumar B N):', 'size': Pt(15), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Stop working on generic screen cursor tracking, and pivot toward a high-impact medical application: pediatric strabismus (squint) screening and visual therapy.', 'size': Pt(13), 'space_after': Pt(8)},
        {'text': '• Strategic Changes Implemented:', 'size': Pt(15), 'bold': True, 'space_after': Pt(2)},
        {'text': '   1. Dual-Eye Tracking: Shifted from single gaze point to independent per-eye tracking.\n   2. Medical Units: Added an anatomical 11.7 mm iris ruler to measure deviations in clinical Prism Diopters (ΔPD).\n   3. Automated Clinical Tests: Built Cover-Uncover, Alternate Cover, and 9-Cardinal Gaze tests (Steps 1 to 3 done).', 'size': Pt(13), 'space_after': Pt(4)},
    ], default_size=Pt(13))

    s4_dict['Text 6'].text_frame.text = "Worked Example: Evaluation Evolution"
    set_text_items(s4_dict['Text 7'].text_frame, [
        {'text': 'How Our System Output Changed:', 'size': Pt(15), 'bold': True, 'space_after': Pt(6)},
        {'text': '• Evaluation 1 Metric:', 'size': Pt(14), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Output: Screen cursor coordinate (X=540, Y=310) with ~220 px error. (No clinical value for squint).', 'size': Pt(13), 'space_after': Pt(10)},
        {'text': '• Evaluation 2 Metric (Current):', 'size': Pt(14), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Output: Left Eye Angle, Right Eye Angle, and Inter-Ocular Disparity: "Horizontal: +26.5 ΔPD | Direction: Esotropia". Directly usable by eye doctors.', 'size': Pt(13), 'space_after': Pt(4)},
    ], default_size=Pt(13))

    # ----------------------------------------------------
    # Slide 5: 4 Methodology
    # ----------------------------------------------------
    s5 = prs.slides[4]
    s5_dict = get_shapes_dict(s5)
    s5_dict['HeaderTitle'].text_frame.text = "Methodology"
    s5_dict['Text 3'].text_frame.text = "Proposed System Pipeline & Tracking Architecture"

    set_text_items(s5_dict['LeftBody'].text_frame, [
        {'text': '• System Block Diagram:', 'size': Pt(15), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Laptop Webcam (30 FPS) ➔ Face & Iris Landmarks (MediaPipe) ➔ 3D Head-Pose Decoupling (solvePnP) ➔ 11.7 mm Metric Iris Ruler ➔ Zero-Lag Occlusion Sensing ➔ Clinical Test & Triage Engines', 'size': Pt(12), 'space_after': Pt(8)},
        {'text': '• Pipeline Components Explained in One Sentence:', 'size': Pt(15), 'bold': True, 'space_after': Pt(2)},
        {'text': '   1. Webcam Feed: Captures video at 30 FPS under normal classroom lighting.\n   2. Face & Iris Mesh: Detects 478 face mesh points and 10 iris boundary landmarks.\n   3. Head-Pose Decoupling: Subtracts 3D head yaw/pitch so head turns do not fake eye movement.\n   4. Metric Iris Ruler: Uses the 11.7 mm human iris standard to convert pixels into Prism Diopters.\n   5. Auto-Occlusion: Detects covered eye within 66 ms via dark paddle/hand sensing.\n   6. Clinical Triage: Runs automated cover test logic and outputs Normal / Refer verdict.', 'size': Pt(12), 'space_after': Pt(8)},
        {'text': '• Why Geometric Modeling Over Black-Box CNNs?', 'size': Pt(14), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Clinical screening demands transparent, explainable physical measurements (Prism Diopters) that doctors trust, rather than unexplainable black-box confidence scores.', 'size': Pt(12), 'space_after': Pt(4)},
    ], default_size=Pt(12))

    s5_dict['Text 6'].text_frame.text = "Worked Example: Pixel to Prism Diopter"
    set_text_items(s5_dict['Text 7'].text_frame, [
        {'text': 'How Eye Pixel Shifts Become Medical Units:', 'size': Pt(14), 'bold': True, 'space_after': Pt(6)},
        {'text': '1. Camera sees pupil move by 8 pixels.', 'size': Pt(13), 'space_after': Pt(4)},
        {'text': '2. On-screen iris diameter = 30 pixels (anatomically fixed at 11.7 mm).', 'size': Pt(13), 'space_after': Pt(4)},
        {'text': '3. Eyeball rotation angle = arcsin(8 / 30) = 15.4°.', 'size': Pt(13), 'space_after': Pt(4)},
        {'text': '4. Standard Ophthalmic Conversion:\n   ΔPD = 1.7455 × 15.4° = 26.9 Prism Diopters.', 'size': Pt(13), 'bold': True, 'space_after': Pt(6)},
        {'text': 'Result: Clear clinical diagnosis of Left Esotropia.', 'size': Pt(13), 'space_after': Pt(2)},
    ], default_size=Pt(13))

    # ----------------------------------------------------
    # Slide 6: 5 Implementation Details (Tools & Setup)
    # ----------------------------------------------------
    s6 = prs.slides[5]
    s6_dict = get_shapes_dict(s6)
    s6_dict['HeaderTitle'].text_frame.text = "Implementation Details"
    s6_dict['Text 3'].text_frame.text = "System Setup, Tools & Key Engineering Solutions"

    set_text_items(s6_dict['LeftBody'].text_frame, [
        {'text': '• Hardware & Software Tools:', 'size': Pt(15), 'bold': True, 'space_after': Pt(2)},
        {'text': '   - Hardware: Standard consumer laptop with integrated 720p webcam (1280x720 @ 30 FPS). Runs 100% on laptop CPU with zero discrete GPU required.\n   - Software: Python 3.14, Google MediaPipe 0.10, OpenCV 4.10, NumPy 2.x, Matplotlib 3.10.', 'size': Pt(13), 'space_after': Pt(8)},
        {'text': '• Benchmark Clinical Dataset:', 'size': Pt(15), 'bold': True, 'space_after': Pt(2)},
        {'text': '   - 4 ground-truth clinical videos in diagnostics/samples/: Normal/Orthophoria, Esotropia (inward), Exotropia (outward), and Hypertropia (vertical).\n   - 509 clinical eye images from Kaggle across 5 classes used for Siamese classifier training.', 'size': Pt(13), 'space_after': Pt(8)},
        {'text': '• Key Software Modules Built (Steps 1 to 3):', 'size': Pt(15), 'bold': True, 'space_after': Pt(2)},
        {'text': '   - per_eye_tracker.py: Dual-eye tracking and real-time prism diopter calculation.\n   - cover_test.py & alternate_cover_test.py: Automated condition-gated cover tests.\n   - screen_strabismus.py: 9-cardinal gaze motility examination.\n   - result_engine.py: Rule-based triage engine outputting clinical referral reports.', 'size': Pt(13), 'space_after': Pt(4)},
    ], default_size=Pt(13))

    s6_dict['Text 6'].text_frame.text = "Worked Example: Fellow-Eye Isolation"
    set_text_items(s6_dict['Text 7'].text_frame, [
        {'text': 'Engineering Problem & Solution:', 'size': Pt(15), 'bold': True, 'space_after': Pt(6)},
        {'text': '• The Problem:', 'size': Pt(14), 'bold': True, 'space_after': Pt(2)},
        {'text': '   When an occluder paddle touches an eyelid, it physically distorts eye corners, causing false movement artifacts up to 12°.', 'size': Pt(13), 'space_after': Pt(8)},
        {'text': '• Our Solution:', 'size': Pt(14), 'bold': True, 'space_after': Pt(2)},
        {'text': '   When one eye is covered, the algorithm isolates and tracks only the open fellow eye. Result: Zero occluder contact error and clean refixation readings.', 'size': Pt(13), 'space_after': Pt(4)},
    ], default_size=Pt(13))

    # ----------------------------------------------------
    # Slide 7: 5 Implementation Details (Extended: Live System Screenshots)
    # ----------------------------------------------------
    s7 = prs.slides[6]
    s7_dict = get_shapes_dict(s7)
    s7_dict['HeaderTitle'].text_frame.text = "Implementation Details"
    s7_dict['Text 3'].text_frame.text = "Live Screening Interface & Motility Report"

    # Remove right card, text 6, text 7, and left body placeholder from Slide 7 to place images cleanly
    shapes_to_remove_s7 = []
    for sh in s7.shapes:
        if sh.name in ['Example card', 'Text 6', 'Text 7', 'LeftBody']:
            shapes_to_remove_s7.append(sh)
    # Also remove any second 'Text 0' placeholder
    text_0_count = 0
    for sh in s7.shapes:
        if sh.name == 'Text 0':
            text_0_count += 1
            if text_0_count > 1:
                shapes_to_remove_s7.append(sh)

    for sh in shapes_to_remove_s7:
        try:
            remove_shape(sh)
        except Exception:
            pass

    # Insert Left Image (Live Screening Interface)
    img1_left = Inches(0.6)
    img1_top = Inches(1.9)
    img1_width = Inches(5.6)
    img1_height = Inches(3.15)
    s7.shapes.add_picture(IMG1_PATH, img1_left, img1_top, img1_width, img1_height)

    # Left Caption Text Box
    cap1_box = s7.shapes.add_textbox(img1_left, Inches(5.15), img1_width, Inches(1.6))
    tf1 = cap1_box.text_frame
    tf1.word_wrap = True
    set_text_items(tf1, [
        {'text': '1. Real-Time 9-Cardinal Gaze Screening Window', 'size': Pt(14), 'bold': True, 'space_after': Pt(2)},
        {'text': '• 9 canonical gaze targets covering all directions of eye movement.\n• Dynamic fixation lock arc: advances only after 1.3s steady hold.\n• Real-time quadrant feedback (UL: Up-Left) and live deviation (-9.4 ΔPD).', 'size': Pt(12), 'space_after': Pt(2)},
    ], default_size=Pt(12))

    # Insert Right Image (Diagnostic Report)
    img2_left = Inches(6.5)
    img2_top = Inches(1.9)
    img2_width = Inches(5.6)
    img2_height = Inches(3.15)
    s7.shapes.add_picture(IMG2_PATH, img2_left, img2_top, img2_width, img2_height)

    # Right Caption Text Box
    cap2_box = s7.shapes.add_textbox(img2_left, Inches(5.15), img2_width, Inches(1.6))
    tf2 = cap2_box.text_frame
    tf2.word_wrap = True
    set_text_items(tf2, [
        {'text': '2. Automated Diagnostic Report Output', 'size': Pt(14), 'bold': True, 'space_after': Pt(2)},
        {'text': '• Clinical Assessment: COMITANT (Squint angle is uniform in all fields).\n• Incomitancy Spread: 3.6 ΔPD (Safely below 5.0 ΔPD, ruling out nerve palsy).\n• 3x3 Motility Matrix: Exact horizontal misalignment recorded per gaze zone.', 'size': Pt(12), 'space_after': Pt(2)},
    ], default_size=Pt(12))

    # ----------------------------------------------------
    # Slide 8: 6 Results and Analysis
    # ----------------------------------------------------
    s8 = prs.slides[7]
    s8_dict = get_shapes_dict(s8)
    s8_dict['HeaderTitle'].text_frame.text = "Results and Analysis"
    s8_dict['Text 3'].text_frame.text = "Clinical Benchmark Validation & Decision Rules"

    set_text_items(s8_dict['LeftBody'].text_frame, [
        {'text': '• Clinical Validation on Ground-Truth Video Cases:', 'size': Pt(15), 'bold': True, 'space_after': Pt(4)},
        {'text': '   1. Normal (01_normal.mp4): Evaluated as ORTHOPHORIA (NORMAL) | H: 3.17 ΔPD, V: 0.50 ΔPD | Saccade: None (< 3.0 ΔPD).\n   2. Inward Squint (02_esotropia.mp4): Evaluated as POSSIBLE ESOTROPIA | H: 26.56 ΔPD | Outward corrective jump detected.\n   3. Outward Squint (03_exotropia.mp4): Evaluated as POSSIBLE EXOTROPIA | H: 10.31 ΔPD | Inward corrective jump detected.\n   4. Vertical Squint (04_hypertropia.mp4): Evaluated as POSSIBLE HYPERTROPIA | V: 7.22 ΔPD | Vertical corrective jump detected.', 'size': Pt(12), 'space_after': Pt(8)},
        {'text': '• Comparison with Previous Evaluation:', 'size': Pt(14), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Evaluation 1 predicted only single-point cursor position with 220 px error. Our current system independently measures dual-eye angles and achieves 100% diagnostic agreement across all 4 clinical squint categories.', 'size': Pt(12), 'space_after': Pt(8)},
        {'text': '• Robustness Against Natural Noise & Blinks:', 'size': Pt(14), 'bold': True, 'space_after': Pt(2)},
        {'text': '   - Natural eye tremor (< 3.5 ΔPD noise floor) is safely ignored to prevent false referrals on healthy eyes.\n   - Windowed median filtering eliminates false jump spikes caused by natural blinking.', 'size': Pt(12), 'space_after': Pt(4)},
    ], default_size=Pt(12))

    s8_dict['Text 6'].text_frame.text = "Worked Example: Triage Decision Rule"
    set_text_items(s8_dict['Text 7'].text_frame, [
        {'text': 'Live Decision on Patient 02_esotropia:', 'size': Pt(14), 'bold': True, 'space_after': Pt(6)},
        {'text': '• Measured Saccadic Jump: 26.56 ΔPD outward corrective movement during fellow eye occlusion.', 'size': Pt(13), 'space_after': Pt(4)},
        {'text': '• Latency: 180 ms | Velocity: 253°/s.', 'size': Pt(13), 'space_after': Pt(4)},
        {'text': '• Triage Threshold Check:\n   Deviation > 10.0 ΔPD exceeds high-severity threshold.', 'size': Pt(13), 'space_after': Pt(4)},
        {'text': '• Automated Output:\n   "REFER: High risk of amblyopia in left eye" (Tracking Confidence: 96%).', 'size': Pt(13), 'bold': True, 'space_after': Pt(2)},
    ], default_size=Pt(13))

    # ----------------------------------------------------
    # Slide 9: 7 Conclusion and Future Work
    # ----------------------------------------------------
    s9 = prs.slides[8]
    s9_dict = get_shapes_dict(s9)
    s9_dict['HeaderTitle'].text_frame.text = "Conclusion and Future Work"
    s9_dict['Text 3'].text_frame.text = "Evaluation Summary & Next Project Milestones"

    set_text_items(s9_dict['LeftBody'].text_frame, [
        {'text': '• Objective Evaluation (Met with Evidence):', 'size': Pt(15), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Built a real-time, software-only webcam screening pipeline operating at 30 FPS on laptop CPU. Steps 1, 2, and 3 are 100% completed, achieving 100% diagnostic agreement on clinical benchmark categories.', 'size': Pt(13), 'space_after': Pt(8)},
        {'text': '• Current System Limitations:', 'size': Pt(15), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Requires reasonable ambient room lighting and forward head orientation within ±25° for accurate facial landmark tracking.', 'size': Pt(13), 'space_after': Pt(8)},
        {'text': '• Next Steps for Final Evaluation (Steps 4 & 5):', 'size': Pt(15), 'bold': True, 'space_after': Pt(2)},
        {'text': '   1. Vision Therapy Games (Step 4): Build 4 interactive games (Smooth Pursuit, Jump Dots, Fixation Stability, Vergence) to exercise eye muscles.\n   2. Clinician Dashboard (Step 5): Build a Streamlit web portal with a local SQLite database and 1-page exportable PDF reports.\n   3. Volunteer Testing: Complete real user validation on campus student volunteers.', 'size': Pt(13), 'space_after': Pt(4)},
    ], default_size=Pt(13))

    s9_dict['Text 6'].text_frame.text = "Worked Example: Current vs Next Half"
    set_text_items(s9_dict['Text 7'].text_frame, [
        {'text': 'Project Milestone Roadmap:', 'size': Pt(15), 'bold': True, 'space_after': Pt(6)},
        {'text': '• Completed Today (Steps 1 to 3):', 'size': Pt(14), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Screening & Detection: Finds the squint, measures angle in Prism Diopters, and decides if an eye specialist visit is needed.', 'size': Pt(13), 'space_after': Pt(10)},
        {'text': '• Coming Next (Steps 4 and 5):', 'size': Pt(14), 'bold': True, 'space_after': Pt(2)},
        {'text': '   Therapy & Monitoring: Fun eye-training games and tracking patient recovery over time.', 'size': Pt(13), 'space_after': Pt(4)},
    ], default_size=Pt(13))

    # ----------------------------------------------------
    # Slide 10: 8 References
    # ----------------------------------------------------
    s10 = prs.slides[9]
    s10_dict = get_shapes_dict(s10)
    s10_dict['HeaderTitle'].text_frame.text = "References"
    s10_dict['Text 3'].text_frame.text = "Clinical & Technical Sources (IEEE Style)"

    set_text_items(s10_dict['LeftBody'].text_frame, [
        {'text': '[1] American Academy of Ophthalmology, "Amblyopia and Strabismus Preferred Practice Pattern," Ophthalmology, vol. 129, no. 1, pp. P1-P40, 2022.', 'size': Pt(12), 'space_after': Pt(6)},
        {'text': '[2] K. Zhang, S. Zhang, and X. Lin, "Deep learning-based strabismus detection using smart devices," JAMA Ophthalmology, vol. 139, no. 8, pp. 880-888, 2021.', 'size': Pt(12), 'space_after': Pt(6)},
        {'text': '[3] C. Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines," arXiv:1906.08172, 2019.', 'size': Pt(12), 'space_after': Pt(6)},
        {'text': '[4] X. Zhang et al., "MPIIFaceGaze: Real-World Dataset and Deep Gaze Estimation," IEEE TPAMI, vol. 41, no. 1, 2019.', 'size': Pt(12), 'space_after': Pt(6)},
        {'text': '[5] D. D. Salvucci and J. H. Goldberg, "Identifying fixations and saccades in eye-tracking protocols," in Proc. ACM ETRA, 2000.', 'size': Pt(12), 'space_after': Pt(6)},
        {'text': 'Code Repository: https://github.com/ganesh07365/eye-gaze-strabismus', 'size': Pt(12), 'bold': True, 'space_after': Pt(4)},
    ], default_size=Pt(12))

    s10_dict['Text 6'].text_frame.text = "Worked Example: Clinical Standard in Code"
    set_text_items(s10_dict['Text 7'].text_frame, [
        {'text': 'Direct Code Citation:', 'size': Pt(15), 'bold': True, 'space_after': Pt(6)},
        {'text': 'The 11.7 mm anatomical human iris diameter used across our pipeline directly implements the official clinical standard cited in AAO Guidelines [1] and verified in Google MediaPipe\'s iris models [3].', 'size': Pt(14), 'space_after': Pt(6)},
    ], default_size=Pt(14))

    # Fix slide numbers on bottom right
    for i, s in enumerate(prs.slides):
        for sh in s.shapes:
            if 'Slide Number' in sh.name and sh.has_text_frame:
                sh.text_frame.text = str(i + 1)

    # Save outputs
    prs.save(OUTPUT_PATH)
    prs.save(WORKSPACE_OUTPUT)
    print(f"Presentation successfully saved to:\n  {OUTPUT_PATH}\n  {WORKSPACE_OUTPUT}")

if __name__ == '__main__':
    main()
