from dicereader.domain.dumper import Dumper
from dicereader.domain.camera import Camera
from dicereader.domain.model import Model, OCRModel

import sys
import tempfile
import cv2
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime


# We'll instantiate the model globally so it's loaded only once
model = None
ocr_model = None

# Global histogram counter
dice_histogram = Counter()

# Status log file for OBS
STATUS_LOG_PATH = "/mnt/c/Users/tiger/OneDrive/Pictures/DiceRoller/status.txt"

def log_status(message):
    print(message)
    with open(STATUS_LOG_PATH, "w") as f:
        f.write(message + "\n")

def detect_dice(frame):
    """
    Detect dice in the given frame using the Model class.
    For now, assumes the whole frame is a single die.
    """
    global model, ocr_model
    # Choose which model to use: set USE_OCR = True to use OCRModel
    USE_OCR = True
    if USE_OCR:
        if ocr_model is None:
            ocr_model = OCRModel()
        category, score = ocr_model.predict(frame)
    else:
        if model is None:
            model = Model(model_location=None)
        category, score = model.predict(frame)
    detection = [{"dice_type": "die", "dice_roll": category, "confidence": score}]
    print("Detected dice:", detection)
    return detection

def record_dice(results, histogram_path="histogram.png"):
    """
    Record the detected dice results and update the histogram.
    Saves the histogram as a PNG for OBS integration.
    """
    global dice_histogram
    # Assume results is a list of dicts with 'dice_roll' key
    for res in results:
        dice_roll = res.get("dice_roll", None)
        if dice_roll is not None:
            dice_histogram[str(dice_roll)] += 1
    # Plot and save histogram
    if dice_histogram:
        plt.figure(figsize=(6,4))
        items = sorted(dice_histogram.items(), key=lambda x: x[0])
        labels, values = zip(*items)
        plt.bar(labels, values, color='skyblue')
        plt.xlabel('Dice Number')
        plt.ylabel('Count')
        plt.title('Dice Roll Histogram')
        plt.tight_layout()
        plt.savefig(histogram_path)
        plt.close()

def split_frames(frame):
    """
    Split the frame into individual dice images using contour detection.
    Returns a list of cropped dice images.
    """
    import cv2
    dice_images = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out small blobs (noise)
        if w * h > 500:  # Adjust this threshold as needed
            dice_img = frame[y:y+h, x:x+w]
            dice_images.append(dice_img)
    return dice_images

def main():

    if len(sys.argv) > 2:
        pi_address = sys.argv[1]
        run_name = sys.argv[2]
    elif len(sys.argv) > 1:
        pi_address = sys.argv[1]
        run_name = datetime.now().strftime('%Y%m%d')
    else:
        print("Incorrect usage. Please provide the PI_ADDRESS as an argument.")
        print("Usage: python orchestrator.py PI_ADDRESS RUN_NAME")
        exit()

    dumper = Dumper(pi_address=pi_address)
    camera = Camera(pi_address=pi_address)

    if not dumper.alive():
        print("Error: Dumper is not responding. Please check the ip address.")
        exit()

    log_status(f"Orchestrator is running... Using PI_ADDRESS: {pi_address}")
    log_status(f"Run name: {run_name}")

    loop_count = 1
    while True:
        try:
            log_status(f"\n*** Starting loop {loop_count} ***")
            log_status("Lowering dice tray...")
            dumper.lower_dice_tray()

            log_status("Dumping dice...")
            dumper.dump_dice()

            log_status("Capturing image from camera...")
            frame = camera.get_image()


            log_status("Splitting frame into dice blobs...")
            dice_images = split_frames(frame)
            log_status(f"Found {len(dice_images)} dice blobs.")
            results = []
            for idx, dice_img in enumerate(dice_images):
                log_status(f"Detecting dice for blob {idx+1}...")
                detected = detect_dice(dice_img)
                results.extend(detected)

            log_status("Recording dice results and updating histogram...")
            record_dice(results, histogram_path="/mnt/c/Users/tiger/OneDrive/Pictures/DiceRoller/histogram.png")

            log_status("Saving image...")
            camera.save_image(frame, info=results, loop_number=loop_count, top_level_folder_name=run_name)

            # TODO: Depending on the timing of detection and raising the dice tray, we might be able to do this inbetween
            # the image capture and detection steps or something to save time.

            log_status("Raising dice tray...")
            success = dumper.raise_dice_tray()
            if not success:
                log_status("Failed to raise dice tray. Exiting loop.")
                break

            log_status("Sweeping dice...")
            dumper.sweep_dice()
            loop_count += 1

        except KeyboardInterrupt:
            log_status("Orchestrator stopped by user.")
            break
        except Exception as e:
            log_status("Error occurred. Halp.")
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()