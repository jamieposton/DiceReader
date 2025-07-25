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
