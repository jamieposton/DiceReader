# camera.py - domain logic for camera operations
import random
import string
import requests
import os
import cv2
import numpy as np
from datetime import datetime

class Camera:
    def __init__(self, pi_address):
        self.pi_address = pi_address

    def get_image(self):
        """Get an image from the remote camera via GET /get_image. Returns a numpy array (OpenCV image)."""
        try:
            resp = requests.get(f"http://{self.pi_address}/get_image")
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

    def save_blob_image(self, result_tuple, loop_number, top_level_folder_name='unlabeled_data'):
        """
        Save a dice blob image to a folder structure based on its label.
        result_tuple: (label, confidence, image)
        The image will be saved to: data/<top_level_folder_name>/<label>/img_TIMESTAMP_RANDOM.jpg
        Ensures filename uniqueness with microseconds and a random suffix.
        """
        label, confidence, image = result_tuple
        img_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        rand_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        folder = os.path.join('data', top_level_folder_name, label)
        os.makedirs(folder, exist_ok=True)
        filename = f"img_{img_timestamp}_{rand_suffix}_loop_{loop_number}.jpg"
        path = os.path.join(folder, filename)
        cv2.imwrite(path, image)
        return path

    def get_camera_status(self):
        """Get camera status from GET /camera_status."""
        try:
            resp = requests.get(f"http://{self.pi_address}/camera_status")
            resp.raise_for_status()
            return resp.json()  # Assuming the status is returned as JSON
        except Exception as e:
            print(f"Error getting camera status: {e}")
            return None
