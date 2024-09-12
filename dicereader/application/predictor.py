import cv2
from dicereader.domain.model import Model


class Predictor:
    """Predictor of Dice Rolls."""

    def __init__(self, model_location: str):
        self.model = Model(model_location)

    def predict(self, path_to_image_file):
        return self.model.predict(path_to_image_file)  # Not a prediction, but it should be an int