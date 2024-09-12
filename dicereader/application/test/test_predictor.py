from dicereader.application import predictor

def test_making_a_prediction():
    current_predictor = predictor.Predictor("fake_model_location")
    prediction = current_predictor.predict("./data/PXL_20240912_213252176.jpg")
    assert isinstance(prediction, int)
