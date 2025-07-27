import os
import cv2
import shutil
from typing import List, Optional

class ImageLabeler:
    def __init__(self, labels: List[str], unknown_label: str = 'unknown', keymap: Optional[dict] = None):
        self.labels = labels
        self.unknown_label = unknown_label
        # keymap: maps key (int) to label string
        if keymap is None:
            # Default: 1-7 keys for die types
            self.keymap = {ord(str(i+1)): label for i, label in enumerate(labels)}
        else:
            self.keymap = keymap

    def label_image(self, img_path: str) -> Optional[str]:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image: {img_path}. Skipping.")
            return self.unknown_label
        key_label_str = ', '.join([f"{i+1}={label}" for i, label in enumerate(self.labels)])
        window_title = f"Label: {img_path} ({key_label_str}, u={self.unknown_label}, q=quit)"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_title, 100, 100)
        cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE, 0)
        cv2.imshow(window_title, img)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in self.keymap:
                cv2.destroyAllWindows()
                return self.keymap[key]
            if key == ord('u'):
                cv2.destroyAllWindows()
                return self.unknown_label
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None

    def label_folder(self, input_folder: str, output_root: str):
        os.makedirs(output_root, exist_ok=True)
        # Iterate over each die type subfolder
        for die_type in sorted(os.listdir(input_folder)):
            die_type_dir = os.path.join(input_folder, die_type)
            if not os.path.isdir(die_type_dir):
                continue
            print(f"Labeling die type folder: {die_type}")
            for img_file in sorted(os.listdir(die_type_dir)):
                img_path = os.path.join(die_type_dir, img_file)
                label = self.label_image(img_path)
                if label is None:
                    print("Quitting...")
                    return
                label_dir = os.path.join(output_root, label)
                os.makedirs(label_dir, exist_ok=True)
                shutil.copy(img_path, label_dir)
                os.remove(img_path)
                print(f"Labeled {img_file} as {label} and deleted from input folder")

# Example usage for die type labeling:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Manually label die type images.')
    parser.add_argument('--input-folder', type=str, required=True, help='Path to the folder of images to label.')
    parser.add_argument('--output-folder', type=str, required=True, help='Path to save labeled images.')
    args = parser.parse_args()

    die_type_labels = ['D4', 'D6', 'D8', 'D10', 'P', 'D12', 'D20']  # P for Percentile
    labeler = ImageLabeler(labels=die_type_labels)
    labeler.label_folder(args.input_folder, args.output_folder)
