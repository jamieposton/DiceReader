import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image

class Model:
    def __init__(self, model_location):
        # self.model = torch.jit.load(model_location)
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=self.weights)
        self.preprocessor = self.weights.transforms()
        self.model.eval()

    def predict(self, path_to_image):
        image = read_image(path_to_image)
        batch = self.preprocessor(image).unsqueeze(0)
        prediction = self.model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = self.weights.meta["categories"][class_id]
        return category_name, score
