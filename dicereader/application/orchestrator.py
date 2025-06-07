
from dicereader.domain.dumper import Dumper
from dicereader.domain.camera import Camera
import sys

def detect_dice(frame):
    """
    Detect dice in the given frame.
    Placeholder function while we're just collecting data.
    """
    detection = [{"dice_type": "unknown", "dice_roll": "unknown", "confidence": 0.0}]
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
    if len(sys.argv) > 1:
        pi_address = sys.argv[1]
    else:
        print("Incorrect usage. Please provide the PI_ADDRESS as an argument.")
        print("Usage: python orchestrator.py <PI_ADDRESS>")
        exit()

    dumper = Dumper(pi_address=pi_address)
    camera = Camera(pi_address=pi_address)
    print(f"Orchestrator is running... Using PI_ADDRESS: {pi_address}")

    while True:
        try:
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
            camera.save_image(frame, info=results)

            # TODO: Depending on the timing of detection and raising the dice tray, we might be able to do this inbetween
            # the image capture and detection steps or something to save time.
            print("Raising dice tray...")
            dumper.raise_dice_tray()

            print("Sweeping dice...")
            dumper.sweep_dice()

        except KeyboardInterrupt:
            print("Orchestrator stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()