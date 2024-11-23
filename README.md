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