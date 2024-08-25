import cv2

class Predictor:
    """Predictor of Dice Rolls."""

    def predict(self, path_to_image_file):
        image = cv2.imread(path_to_image_file)
        return image.shape[0]  # Not a prediction, but it should be an int