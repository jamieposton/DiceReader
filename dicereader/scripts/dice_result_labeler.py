import os
import cv2
import shutil
from dicereader.domain.model import DIE_TYPE_OPTIONS

class DiceResultLabeler:
    def __init__(self, die_type_labels, result_options=None, unknown_label='unknown'):
        self.die_type_labels = die_type_labels
        self.result_options = result_options or {}
        self.unknown_label = unknown_label

    def label_image(self, img_path, die_type):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image: {img_path}. Skipping.")
            return self.unknown_label
        options = self.result_options.get(die_type, [])
        key_label_str = ', '.join([str(opt) for opt in options])
        window_title = f"Label: {img_path} (Options: {key_label_str}, u={self.unknown_label}, q=quit)"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_title, 100, 100)
        cv2.imshow(window_title, img)
        cv2.waitKey(1)  # Ensure image is rendered
        print(f"\nLabeling image: {img_path}")
        print(f"Die type: {die_type}")
        print(f"Options: {key_label_str}")
        print(f"Type the result and press Enter, or 'u' for unknown, 'q' to quit.")
        # Keep window open until label is entered
        import threading
        def keep_window_open():
            while True:
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                cv2.waitKey(100)
        t = threading.Thread(target=keep_window_open)
        t.daemon = True
        t.start()
        while True:
            label = input("Label: ").strip()
            if label == 'q':
                cv2.destroyAllWindows()
                return None
            if label == 'u':
                cv2.destroyAllWindows()
                return self.unknown_label
            if label in [str(opt) for opt in options]:
                cv2.destroyAllWindows()
                return label
            print(f"Invalid input. Please enter one of: {key_label_str}, 'u', or 'q'.")
        window_title = f"Label: {img_path} ({key_label_str}, u={self.unknown_label}, q=quit)"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_title, 100, 100)
        cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE, 0)
        cv2.imshow(window_title, img)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in keymap:
                cv2.destroyAllWindows()
                return keymap[key]
            if key == ord('u'):
                cv2.destroyAllWindows()
                return self.unknown_label
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None

    def label_folder(self, input_root: str, output_root: str):
        os.makedirs(output_root, exist_ok=True)
        for die_type in sorted(os.listdir(input_root)):
            die_type_dir = os.path.join(input_root, die_type)
            if not os.path.isdir(die_type_dir):
                continue
            print(f"Labeling die type: {die_type}")
            for img_file in sorted(os.listdir(die_type_dir)):
                img_path = os.path.join(die_type_dir, img_file)
                result = self.label_image(img_path, die_type)
                if result is None:
                    print("Quitting...")
                    return
                label_dir = os.path.join(output_root, die_type, result)
                os.makedirs(label_dir, exist_ok=True)
                shutil.copy(img_path, label_dir)
                print(f"Labeled {img_file} as {result} for {die_type}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Manually label dice result images by die type.')
    parser.add_argument('--input-folder', type=str, required=True, help='Path to the folder of die type folders to label.')
    parser.add_argument('--output-folder', type=str, required=True, help='Path to save labeled images.')
    args = parser.parse_args()

    die_type_labels = ['D4', 'D6', 'D8', 'D10', 'P', 'D12', 'D20']
    labeler = DiceResultLabeler(die_type_labels, result_options=DIE_TYPE_OPTIONS)
    labeler.label_folder(args.input_folder, args.output_folder)
