import cv2
import easyocr
from pathlib import Path
from datetime import datetime, timezone

print("Initializing camera")
cam = cv2.VideoCapture("/dev/video4")

print("Initializing OCR Reader")
reader = easyocr.Reader(['en'])

cv2.namedWindow("test")

img_counter = 0
timestamp = str(datetime.now(tz=timezone.utc))
base_path = Path("data") / Path(timestamp)
base_path.mkdir(parents=True)
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "data/{}/opencv_frame_{}.png".format(timestamp, img_counter)
        print("Reader found the text:")
        print(reader.readtext(frame))
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
