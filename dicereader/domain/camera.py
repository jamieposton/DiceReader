# camera.py - domain logic for camera operations
import requests
import os
import cv2
import numpy as np

class Camera:
    def __init__(self, pi_address="http://127.0.0.1:5000"):
        self.pi_address = pi_address

    def get_image(self):
        """Get an image from the remote camera via GET /get_image. Returns a numpy array (OpenCV image)."""
        try:
            resp = requests.get(f"{self.pi_address}/get_image")
            resp.raise_for_status()
            # Assume the response is a JPEG image
            img_array = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("Failed to decode image from response.")
            return img
        except Exception as e:
            print(f"Error getting image from camera: {e}")
            return None

    def save_image(self, frame, info):
        """
        Save image to a folder structure based on info dict.
        info: dict with keys 'dice_type' and 'dice_roll'.
        The image will be saved to: <dice_type>/<dice_roll>/img_TIMESTAMP.jpg
        """
        #TODO: This makes less sense when we're going to split the image into bounding boxes.
        from datetime import datetime
        dice_type = info.get('dice_type', 'unknown_type')
        dice_roll = str(info.get('dice_roll', 'unknown'))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        folder = os.path.join(dice_type, dice_roll)
        os.makedirs(folder, exist_ok=True)
        filename = f"img_{timestamp}.jpg"
        path = os.path.join(folder, filename)
        cv2.imwrite(path, frame)
        return path

    def get_camera_status(self):
        """Get camera status from GET /camera_status."""
        try:
            resp = requests.get(f"{self.pi_address}/camera_status")
            resp.raise_for_status()
            return resp.json()  # Assuming the status is returned as JSON
        except Exception as e:
            print(f"Error getting camera status: {e}")
            return None
