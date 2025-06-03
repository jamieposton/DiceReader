import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

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

# Initialize EasyOCR reader once
import easyocr
reader = easyocr.Reader(['en'], gpu=False)

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
    last_roll = None
    try:
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480))
            if not ret:
                print("Error: Failed to capture frame.")
                break
            roll = detect_dice_number(frame)
            if roll > 0:
                roll_counts[roll] += 1
                roll_history.append(roll)
                last_roll = roll
            show_histogram()
            cv2.putText(frame, f"Detected: {last_roll if last_roll else '-'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Dice Detection', frame)
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()