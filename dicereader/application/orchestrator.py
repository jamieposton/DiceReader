def main():
    print("Orchestrator is running...")

    set_up_camera_stream()
    while(True):
        try:
            # Assuming we're starting with the dice in the the dump trough
            reset_dice_tray()

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