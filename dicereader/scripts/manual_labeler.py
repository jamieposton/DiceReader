import os
import cv2
import shutil
from datetime import datetime

# Get the most recent run folder (timestamped)
def get_latest_run_folder(base_dir):
    auto_labels_dir = os.path.join(base_dir, 'data', 'auto-labels')
    if not os.path.exists(auto_labels_dir):
        raise Exception(f"No auto-labels directory found at {auto_labels_dir}.")
    folders = [f for f in os.listdir(auto_labels_dir) if os.path.isdir(os.path.join(auto_labels_dir, f))]
    if not folders:
        raise Exception("No run folders found in auto-labels directory.")
    folders.sort()
    return folders[-1]

def confirm_label(img_path, label):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load image: {img_path}. Skipping.")
        return False
    window_title = f'Label: {label} | {img_path} (y=confirm, n=reject, q=quit)'
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_title, img)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            cv2.destroyAllWindows()
            return True
        elif key == ord('n'):
            cv2.destroyAllWindows()
            return False
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None

def manual_label(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load image: {img_path}. Skipping.")
        return 'unknown'
    window_title = f'Unknown: {img_path} (1-6=label, u=unknown, q=quit)'
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_title, img)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in [ord(str(i)) for i in range(1, 7)]:
            cv2.destroyAllWindows()
            return str(chr(key))
        elif key == ord('u'):
            cv2.destroyAllWindows()
            return 'unknown'
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None

def main():

    import argparse
    parser = argparse.ArgumentParser(description='Manually label dice images.')
    parser.add_argument('--input-folder', type=str, default=None, help='Path to the folder of images to label (timestamped run folder).')
    args = parser.parse_args()

    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    auto_labels_root = os.path.join(data_dir, 'auto-labels')
    curated_root = os.path.join(data_dir, 'curated-labels')
    os.makedirs(auto_labels_root, exist_ok=True)
    os.makedirs(curated_root, exist_ok=True)


    # If input-folder is specified, only process that run folder
    run_folders = []
    if args.input_folder:
        run_path = os.path.abspath(args.input_folder)
        run_folder = os.path.basename(run_path.rstrip('/'))
        run_folders = [(run_folder, run_path)]
    else:
        # Process all run folders in auto-labels
        run_folders = []
        for folder in sorted(os.listdir(auto_labels_root)):
            folder_path = os.path.join(auto_labels_root, folder)
            if os.path.isdir(folder_path):
                run_folders.append((folder, folder_path))

    for run_folder, run_path in run_folders:
        curated_run = os.path.join(curated_root, run_folder)
        # If the curated run folder already exists, skip labeling for this run
        if os.path.exists(curated_run):
            print(f"Curated folder for this run already exists: {curated_run}\nSkipping all labeling for this run.")
            continue
        os.makedirs(curated_run, exist_ok=True)

        # 1. Confirm or reject existing labels (except unknown), skipping already curated images
        for label in os.listdir(run_path):
            label_path = os.path.join(run_path, label)
            if not os.path.isdir(label_path) or label == 'unknown':
                continue
            curated_label_path = os.path.join(curated_run, label)
            os.makedirs(curated_label_path, exist_ok=True)
            curated_files = set(os.listdir(curated_label_path)) if os.path.exists(curated_label_path) else set()
            for img_file in os.listdir(label_path):
                if img_file in curated_files:
                    # Already labeled, skip
                    continue
                img_path = os.path.join(label_path, img_file)
                result = confirm_label(img_path, label)
                if result is None:
                    print("Quitting...")
                    return
                elif result:
                    shutil.copy(img_path, curated_label_path)
                else:
                    # Move to unknown
                    unknown_path = os.path.join(run_path, 'unknown')
                    os.makedirs(unknown_path, exist_ok=True)
                    shutil.move(img_path, os.path.join(unknown_path, img_file))

        # 2. Manually label unknowns, skipping already curated images
        unknown_path = os.path.join(run_path, 'unknown')
        if os.path.exists(unknown_path):
            for img_file in os.listdir(unknown_path):
                # Check if this image is already in any curated label folder for this run
                already_curated = False
                for label in [str(i) for i in range(1, 7)] + ['unknown']:
                    curated_label_path = os.path.join(curated_run, label)
                    if os.path.exists(os.path.join(curated_label_path, img_file)):
                        already_curated = True
                        break
                if already_curated:
                    continue
                img_path = os.path.join(unknown_path, img_file)
                label = manual_label(img_path)
                if label is None:
                    print("Quitting...")
                    return
                if label != 'unknown':
                    curated_label_path = os.path.join(curated_run, label)
                    os.makedirs(curated_label_path, exist_ok=True)
                    shutil.copy(img_path, curated_label_path)

        print(f"Curated dataset saved to: {curated_run}")

if __name__ == "__main__":
    main()
