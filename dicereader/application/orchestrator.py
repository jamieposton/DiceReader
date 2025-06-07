
import detect_dice
import record_dice
from dicereader.domain.dumper import Dumper
from dicereader.domain.camera import Camera
import sys

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
            # Assuming we're starting with the dice in the the dump trough
            dumper.lower_dice_tray()

            dumper.dump_dice()


            # Get image from camera
            frame = camera.get_image()

            results = detect_dice(frame)

            record_dice(results) # This should also be forwarded to the stream and a database somehow

            # Save image using camera domain, with dice roll results
            camera.save_image(frame, info=results)

            dumper.raise_dice_tray()

            dumper.sweep_dice()

        except KeyboardInterrupt:
            print("Orchestrator stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()