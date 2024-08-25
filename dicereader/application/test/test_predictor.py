from dicereader.application import predictor

def test_making_a_prediction():
    current_predictor = predictor.Predictor()
    prediction = current_predictor.predict("path_to_file.jpeg")
    assert isinstance(prediction, int)
