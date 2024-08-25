from dicereader.application import predictor

def test_making_a_prediction():
    current_predictor = predictor.Predictor()
    prediction = current_predictor.predict("data/2024-08-25 03:50:01.700701+00:00/opencv_frame_0.png")
    assert isinstance(prediction, int)
