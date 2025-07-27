import easyocr
from torchvision.models import efficientnet_b0
import numpy as np
import torch
import torch.nn as nn
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image
import cv2
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import Counter
from datetime import datetime


DIE_TYPE_OPTIONS = {
    "D4": list(range(1, 5)),
    "D6": list(range(1, 7)),
    "D8": list(range(1, 9)),
    "D10": list(range(1, 11)),
    "P": [i for i in range(10, 100, 10)] + [00],  # Percentile die: '00', '10', ..., '90'
    "D12": list(range(1, 13)),
    "D20": list(range(1, 21)),
}


class EfficientNetDieTypeModel:
    """
    EfficientNet-based model for die type prediction.
    """
    def __init__(self, model_location, device='cpu'):

        checkpoint = torch.load(model_location, map_location=device)
        self.class_names = checkpoint.get('class_names', ['D4', 'D6', 'D8', 'D10', 'P', 'D12', 'D20', 'unknown'])
        self.device = device
        self.model = efficientnet_b0()
        num_classes = len(self.class_names)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image):
        """
        image: numpy array (OpenCV BGR)
        Returns die type label.
        """
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            class_id = probs.argmax().item()
            confidence = probs[0, class_id].item()
            label = self.class_names[class_id]
        return label


class EfficientNetDieResultModel:
    def __init__(self, die_type_model_path, die_result_model_paths, device='cpu'):
        self.device = device
        # Die type model
        self.die_type_model = EfficientNetDieTypeModel(model_location=die_type_model_path, device=device)
        # Die result models: dict of die_type -> model
        self.die_result_models = {}
        self.result_class_names = {}
        for die_type, model_path in die_result_model_paths.items():
            checkpoint = torch.load(model_path, map_location=device)
            model = efficientnet_b0()
            num_classes = len(checkpoint['class_names'])
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.to(device)
            self.die_result_models[die_type] = model
            self.result_class_names[die_type] = checkpoint['class_names']

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, frame):
        """
        Predict die type and result for each blob in the frame.
        Returns list of (result, die_type, image)
        """
        dice_images = split_frames(frame)
        results = []
        for image in dice_images:
            die_type = self.die_type_model.predict(image)
            model = self.die_result_models.get(die_type)
            class_names = self.result_class_names.get(die_type)
            if model is None or class_names is None:
                results.append(("unknown", die_type, image))
                continue
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                class_id = probs.argmax().item()
                confidence = probs[0, class_id].item()
                result = class_names[class_id]
            results.append((result, die_type, image))
        return results


class OCRModel:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.die_type_model = EfficientNetDieTypeModel(
            model_location=os.path.join(os.getcwd(), 'checkpoints', 'die_type_models', 'efficientnet_dice_classifier_20250726_204313.pt'),
            device='cpu'
        )

    def predict(self, frame):
        """
        image: numpy array (OpenCV BGR)
        Rotates the image multiple ways and uses OCR to predict the dice number.
        Returns the list of the most common non-None prediction and its vote count.
        """
        dice_images = split_frames(frame)
        results = []
        print(f"Found {len(dice_images)} blobs.")
        for i, image in enumerate(dice_images):
            print(f"Detecting dice for blob {i+1}...")
            die_type = self.die_type_model.predict(image)
            valid_options = DIE_TYPE_OPTIONS.get(die_type)
            votes = []
            angles = [0, 90, 180, 270]
            for angle in angles:
                rot_img = rotate_image(image, angle)
                result = self.reader.readtext(rot_img, detail=0)
                # Filter to digits only
                digits = [s for s in result if s.isdigit()]
                # Keep only valid options
                if valid_options:
                    digits = [s for s in digits if (s.isdigit() and (int(s) in valid_options or s in valid_options))]
                if digits:
                    votes.extend(digits)
            if votes:
                # Majority vote
                vote_counts = Counter(votes)
                best, count = vote_counts.most_common(1)[0]
                results.append((best, die_type, image))
                print(f"Blob {i+1}: Detected number {best} for a {die_type} with confidence {count / len(angles):.2f}")
            else:
                print(f"No digits detected for blob {i+1}{die_type}.")
                results.append(("unknown", die_type, image))
        return results

def split_frames(self, frame):
    """
    Split the frame into individual dice images using contour detection.
    Returns a list of cropped dice images.
    """
    dice_images = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_h, frame_w = frame.shape[:2]
    max_blob_area = 0.2 * frame_w * frame_h  # blobs must be less than 20% of the frame area
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        blob_area = w * h
        # Filter out small blobs (noise) and blobs that are too large
        if 500 < blob_area < max_blob_area:
            aspect_ratio = w / h if h != 0 else 0
            # Only keep near-square blobs (aspect ratio between 0.7 and 1.3)
            if 0.7 <= aspect_ratio <= 1.3:
                dice_img = frame[y:y+h, x:x+w]
                dice_images.append(dice_img)
    return dice_images


def rotate_image(image, angle):
    """Rotate image by angle degrees."""
    import cv2
    if angle == 0:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


