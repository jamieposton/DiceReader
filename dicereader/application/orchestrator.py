
import detect_dice
import record_dice
from utilities import set_up_camera_stream, save_image
from dicereader.domain.dumper import Dumper
import sys

def main():
    if len(sys.argv) > 1:
        pi_address = sys.argv[1]
    else:
        print("Incorrect usage. Please provide the PI_ADDRESS as an argument.")
        print("Usage: python orchestrator.py <PI_ADDRESS>")
        exit()

    dumper = Dumper(pi_address=pi_address)
    print(f"Orchestrator is running... Using PI_ADDRESS: {pi_address}")

    set_up_camera_stream()
    while(True):
        try:
            # Assuming we're starting with the dice in the the dump trough
            dumper.lower_dice_tray()

            dumper.dump_dice()

            results = detect_dice()

            record_dice(results) # This should also be forwarded to the stream and a database somehow

            save_image(results)

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