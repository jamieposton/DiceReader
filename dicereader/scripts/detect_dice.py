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

# TODO: don't try to detect dice with pips: use text.
def detect_dice_number(frame):
    """Simplified placeholder: replace with a ML model later"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Count dots (pip detection)
    dot_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 300:  # reasonable dot size
            dot_count += 1
            (x, y), r = cv2.minEnclosingCircle(cnt)
            cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)

    return min(dot_count, 6)  # cap to 6, useful if noise miscounts

# TODO: Is this _really_ the right size of the camera?
def main():
    W=640
    H=480
    cap = cv2.VideoCapture(0)
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