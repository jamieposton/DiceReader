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

def main():
    cap = cv2.VideoCapture(0)
    plt.ion()