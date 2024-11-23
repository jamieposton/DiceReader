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