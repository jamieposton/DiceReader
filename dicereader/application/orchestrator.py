import detect_dice
import record_dice
from utilities import set_up_camera_stream, save_image

# TODO: move these utility functions to a separate module
def lower_dice_tray():
    # Post to pi_address/move_tray_to_bottom
    # return result of post in bool

def raise_dice_tray():
    # Post to pi_address/move_tray_to_top
    # return result of post in bool

def sweep_dice():
    # Post to pi_address/sweep_dice_into_trough
    # return result of post in bool

def dump_dice():
    # Post to pi_address/dump_trough
    # return result of post in bool

def main():
    print("Orchestrator is running...")

    set_up_camera_stream()
    while(True):
        try:
            # Assuming we're starting with the dice in the the dump trough
            lower_dice_tray()

            dump_dice()

            results = detect_dice()

            record_dice(results) # This should also be forwarded to the stream and a database somehow

            save_image(results)

            raise_dice_tray()

            sweep_dice()

        except KeyboardInterrupt:
            print("Orchestrator stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()