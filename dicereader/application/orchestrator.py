

from dicereader.domain.dumper import Dumper
from dicereader.domain.camera import Camera
from dicereader.domain.model import Model
import sys
import tempfile
import cv2


# We'll instantiate the model globally so it's loaded only once
model = None

def detect_dice(frame):
    """
    Detect dice in the given frame using the Model class.
    For now, assumes the whole frame is a single die.
    """
    global model
    if model is None:
        # You may want to change this to your actual model checkpoint
        model = Model(model_location=None)
    category, score = model.predict(frame)
    detection = [{"dice_type": "die", "dice_roll": category, "confidence": score}]
    print("Detected dice:", detection)
    return detection

def record_dice(results):
    """
    Record the detected dice results.
    Placeholder function while we're just collecting data.
    """
    pass

def split_frames(frame):
    """
    Split the frame into individual dice images.
    This is a placeholder function. The actual implementation will depend on how the dice are detected.
    """
    # For now, just return the original frame as a single "dice" image
    return frame

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

    print(f"Orchestrator is running... Using PI_ADDRESS: {pi_address}")
    print(f"Run name: {run_name}")

    loop_count = 1
    while True:
        try:
            print(f"\n*** Starting loop {loop_count} ***")
            print("Lowering dice tray...")
            dumper.lower_dice_tray()

            print("Dumping dice...")
            dumper.dump_dice()

            print("Capturing image from camera...")
            frame = camera.get_image()

            print("Detecting dice...")
            results = detect_dice(frame)

            print("Recording dice results...")
            record_dice(results) # This should also be forwarded to the stream and a database somehow

            print("Saving image...")
            camera.save_image(frame, info=results, loop_number=loop_count, run_name=run_name)

            # TODO: Depending on the timing of detection and raising the dice tray, we might be able to do this inbetween
            # the image capture and detection steps or something to save time.

            print("Raising dice tray...")
            success = dumper.raise_dice_tray()
            if not success:
                print("Error: Failed to raise dice tray (possibly 500 error). Exiting loop.")
                break

            print("Sweeping dice...")
            dumper.sweep_dice()
            loop_count += 1

        except KeyboardInterrupt:
            print("Orchestrator stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()