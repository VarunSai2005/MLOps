from model import predict_result


def test_model_predicts_pass():
    assert predict_result(75) == "Pass"


def test_model_predicts_fail():
    assert predict_result(25) == "Fail"
