import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image
import cv2

class Model:
    def __init__(self, model_location):
        # self.model = torch.jit.load(model_location)
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=self.weights)
        self.preprocessor = self.weights.transforms()
        self.model.eval()

    def predict(self, image):
        # image is a numpy array (OpenCV BGR)
        if isinstance(image, str):
            # fallback for old usage
            image = read_image(image)
            batch = self.preprocessor(image).unsqueeze(0)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            batch = self.preprocessor(pil_img).unsqueeze(0)
        prediction = self.model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = self.weights.meta["categories"][class_id]
        return category_name, score


class OCRModel:
    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=False)

    def predict(self, image):
        """
        image: numpy array (OpenCV BGR)
        Rotates the image multiple ways and uses OCR to predict the dice number.
        Returns the most common non-None prediction and its vote count.
        """
        import cv2
        votes = []
        angles = [0, 90, 180, 270]
        for angle in angles:
            rot_img = self.rotate_image(image, angle)
            result = self.reader.readtext(rot_img, detail=0)
            # Filter to digits only
            digits = [s for s in result if s.isdigit()]
            if digits:
                votes.extend(digits)
        if votes:
            # Majority vote
            from collections import Counter
            vote_counts = Counter(votes)
            best, count = vote_counts.most_common(1)[0]
            return best, count / len(angles)
        else:
            return None, 0.0

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