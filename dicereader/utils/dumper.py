
# dumper.py - utility functions for dice tray and trough control
import requests

PI_ADDRESS = "http://127.0.0.1:5000"  # Change to your Pi's address as needed

def lower_dice_tray():
    try:
        resp = requests.post(f"{PI_ADDRESS}/move_tray_to_bottom")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error lowering dice tray: {e}")
        return False

def raise_dice_tray():
    try:
        resp = requests.post(f"{PI_ADDRESS}/move_tray_to_top")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error raising dice tray: {e}")
        return False

def sweep_dice():
    try:
        resp = requests.post(f"{PI_ADDRESS}/sweep_dice_into_trough")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error sweeping dice: {e}")
        return False

def dump_dice():
    try:
        resp = requests.post(f"{PI_ADDRESS}/dump_trough")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error dumping dice: {e}")
        return False
