import matplotlib
matplotlib.use('TkAgg')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os 

# Rolling statistics
roll_counts = defaultdict(int)
roll_history = []

def show_histogram():
    plt.clf()
    rolls = list(range(1, 7))
    freqs = [roll_counts[r] for r in rolls]
    plt.bar(rolls, freqs, tick_label=rolls)
    plt.xlabel("Die Value")
    plt.ylabel("Frequency")
    plt.title("Live Dice Roll Distribution")
    plt.pause(0.001)
    plt.gcf().canvas.flush_events()

# Initialize EasyOCR reader once
import easyocr
import torch
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
reader = easyocr.Reader(['en'], gpu=False)

# Load EfficientNet model

# Find the latest EfficientNet checkpoint in the checkpoints directory
import glob
checkpoint_files = sorted(glob.glob(os.path.join(os.getcwd(), 'checkpoints', 'efficientnet_dice_classifier_*.pt')))
if not checkpoint_files:
    raise FileNotFoundError("No EfficientNet checkpoint found in /checkpoints. Please train the model first.")
CHECKPOINT_PATH = checkpoint_files[-1]
model = efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 6)
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
class_names = checkpoint['class_names']

# Preprocessing for model
model_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def detect_dice_number(frame):
    """Detects the number on a die using OCR (EasyOCR). Returns the detected integer or 0 if not found."""
    # Convert to RGB for EasyOCR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb_frame, detail=1, paragraph=False)
    detected_number = 0
    best_conf = 0
    best_bbox = None
    for (bbox, text, conf) in results:
        text = text.strip()
        # Only consider 1-digit numbers (1-6) for standard dice
        if text.isdigit() and 1 <= int(text) <= 6:
            if conf > best_conf:
                detected_number = int(text)
                best_conf = conf
                best_bbox = bbox
    # Highlight the detected number in the frame
    if best_bbox is not None:
        pts = np.array(best_bbox, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0,255,255), thickness=3)
    return detected_number

# TODO: Is this _really_ the right size of the camera?
def main():
    W=640
    H=480
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    plt.ion()
    plt.show(block=False)
    plt.pause(0.1)  # Give time for the window to appear
    last_roll = None
    import os
    from datetime import datetime
    # Create top-level folder with datetime stamp under data/auto-labels
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_root = os.path.join(os.getcwd(), 'data', 'auto-labels', run_timestamp)
    os.makedirs(save_root, exist_ok=True)
    try:
        last_detected_roll = None
        detecting = False
        detection_frames = []
        detection_frame_imgs = []
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480))
            if not ret:
                print("Error: Failed to capture frame.")
                break

            key = cv2.waitKey(1) & 0xFF
            if detecting:
                detection_frames.append(frame.copy())
                if len(detection_frames) == 5:
                    ocr_results = [detect_dice_number(f) for f in detection_frames]
                    # Model predictions
                    model_preds = []
                    for f in detection_frames:
                        with torch.no_grad():
                            input_tensor = model_transform(f).unsqueeze(0)
                            out = model(input_tensor)
                            pred = out.argmax(1).item() + 1  # class indices 0-5 -> dice 1-6
                            model_preds.append(pred)
                    # Decide label: if all OCR fail, unknown. If OCR and model disagree, unknown.
                    ocr_valid = [r for r in ocr_results if r > 0]
                    model_valid = [p for p in model_preds if p > 0]
                    # Use the most common OCR and model prediction
                    from collections import Counter
                    ocr_label = Counter(ocr_valid).most_common(1)[0][0] if ocr_valid else None
                    model_label = Counter(model_valid).most_common(1)[0][0] if model_valid else None
                    # If OCR can't recognize, unknown
                    if ocr_label is None:
                        label = "unknown"
                    # If OCR and model disagree, unknown
                    elif ocr_label != model_label:
                        label = "unknown"
                    else:
                        label = ocr_label
                        roll_counts[label] += 1
                        roll_history.append(label)
                        last_detected_roll = label
                    # Save only the middle frame (index 2) to the appropriate folder
                    label_folder = os.path.join(save_root, str(label))
                    os.makedirs(label_folder, exist_ok=True)
                    img_filename = os.path.join(label_folder, f"img_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
                    cv2.imwrite(img_filename, detection_frames[2])
                    detection_frames = []
                    detecting = False
            if key == ord('d') and not detecting:
                detecting = True
                detection_frames = []
            show_histogram()
            cv2.putText(frame, f"Detected: {last_detected_roll if last_detected_roll else '-'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Dice Detection', frame)
            # Press 'q' to quit
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()