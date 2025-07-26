import easyocr
from torchvision.models import efficientnet_b0
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
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

class DieTypeModel:
    """
    CNN-based model for die type prediction.
    """
    def __init__(self, model_location=None, num_classes=7):
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.fc1 = nn.Linear(32 * 25 * 25, 128)
                self.fc2 = nn.Linear(128, num_classes)
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        self.model = SimpleCNN(num_classes)
        if model_location is not None:
            self.model.load_state_dict(torch.load(model_location, map_location='cpu'))
        self.model.eval()
        self.class_names = ['D4', 'D6', 'D8', 'D10', 'P', 'D12', 'D20']
        self.transform = T.Compose([
            T.Resize((100, 100)),
            T.ToTensor(),
        ])

    def predict(self, image):
        """
        image: numpy array (OpenCV BGR)
        Returns die type label and confidence.
        """
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            class_id = probs.argmax().item()
            confidence = probs[0, class_id].item()
            label = self.class_names[class_id]
        return label, confidence

    def train(self, data_dir, epochs=10, batch_size=32, lr=0.001, save_path=None):
        """
        Train the CNN on labeled images in data_dir.
        data_dir: path to root folder with subfolders named by class (die type), each containing images.
        """

        transform = self.transform
        dataset = ImageFolder(data_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            avg_loss = running_loss / len(dataset)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        self.model.eval()
        if save_path:
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")


class OCRModel:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.die_type_model = EfficientNetDieTypeModel(
            model_location=os.path.join(os.getcwd(), 'checkpoints', 'efficientnet_dice_classifier_20250725_214613.pt'),
            device='cpu'
        )

    def predict(self, frame):
        """
        image: numpy array (OpenCV BGR)
        Rotates the image multiple ways and uses OCR to predict the dice number.
        Returns the list of the most common non-None prediction and its vote count.
        """
        dice_images = self.split_frames(frame)
        results = []
        print(f"Found {len(dice_images)} blobs.")
        for i, image in enumerate(dice_images):
            print(f"Detecting dice for blob {i+1}...")
            die_type = self.die_type_model.predict(image)
            valid_options = DIE_TYPE_OPTIONS.get(die_type)
            votes = []
            angles = [0, 90, 180, 270]
            for angle in angles:
                rot_img = self.rotate_image(image, angle)
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


    @staticmethod
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