# dumper.py - utility functions for dice tray and trough control
import requests

class Dumper:
    def __init__(self, pi_address):
        self.pi_address = pi_address

    def lower_dice_tray(self):
        try:
            resp = requests.post(f"http://{self.pi_address}/move_tray_to_bottom")
            return resp.status_code == 200
        except Exception as e:
            print(f"Error lowering dice tray: {e}")
            return False

    def raise_dice_tray(self):
        try:
            resp = requests.post(f"http://{self.pi_address}/move_tray_to_top")
            return resp.status_code == 200
        except Exception as e:
            print(f"Error raising dice tray: {e}")
            return False

    def sweep_dice(self):
        try:
            resp = requests.post(f"http://{self.pi_address}/sweep_dice_into_trough")
            return resp.status_code == 200
        except Exception as e:
            print(f"Error sweeping dice: {e}")
            return False

    def dump_dice(self):
        try:
            resp = requests.post(f"http://{self.pi_address}/dump_trough")
            return resp.status_code == 200
        except Exception as e:
            print(f"Error dumping dice: {e}")
            return False
