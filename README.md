# Setup

You should be able to start a virtual environment using Poetry using this command:
```shell
pip install poetry
poetry shell
```

Then you can install the packages using this:
```shell
poetry install
```

## TODO
 - Attempt OCR on dice
 - Depending on that, write function to find center bounding box and print result of center bounding box
 - Hook into dice roller to activate camera after dumping but before raising
 - Evaluate accuracy over successive runs
 - Potentially switch out models depending on accuracy, or train custom model.
   - The OCR might do well at detecting characters, but needs help with identifying the characters.

# Troubleshooting

## `error: (-2:Unspecified error) The function is not implemented.`
As far as I can tell, this is an issue with poetry using a different version of cv2
that is uncompiled?

To fix this, you can try the suggestions to install libgtk2.0-dev and pkg-config.
If that doesn't work, try uninstalling and reinstalling opencv-python with pip:

```shell
pip uninstall opencv-python
pip install opencv-python
```

## Script using wrong camera, or failed to find frame

This could be because you have two cameras plugged into your computer.
For me, this was a built-in webcam and a usb camera that wasn't set as the default.

To switch cameras, try running `ls /dev/` before and after plugging in your camera.
You should be able to see a list of `video0`, `video1`, and so on.
Your USB camera should be the video that appears in that list after plugging it in.
Once you've figured that out, you can use that when calling `cv2.VideoCapture` instead
of whatevers already in the script.