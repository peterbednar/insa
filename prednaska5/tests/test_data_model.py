import copy
import pytest
from pydantic import ValidationError
from uui_iris_predictor.data_model import RpcRequest, RpcResponse

TEST_REQUEST_1 = {
    "jsonrpc": "2.0",
    "method": "pipeline/predict",
    "params": {
        "data": [
            {"sepal_length": 6.1, "sepal_width": 3.0, "petal_length": 4.6, "petal_width": 1.4}
        ]
    },
    "id": 1
}

TEST_RESPONSE_1 = {
    "jsonrpc": "2.0",
    "result": {
        "predictions": [{"label": 1}],
        "pipeline": "0.1.0"
    },
    "id": 1
}

def test_request_1():
    RpcRequest(**TEST_REQUEST_1)

def test_request_2():
    request = copy.deepcopy(TEST_REQUEST_1)
    request["jsonrpc"] = "1.0"

    with pytest.raises(ValidationError):
        RpcRequest(**request)

def test_request_3():
    request = copy.deepcopy(TEST_REQUEST_1)
    request["method"] = "unknown"

    with pytest.raises(ValidationError):
        RpcRequest(**request)

def test_request_4():
    request = copy.deepcopy(TEST_REQUEST_1)
    del request["params"]

    with pytest.raises(ValidationError):
        RpcRequest(**request)

def test_response_1():
    RpcResponse(**TEST_RESPONSE_1)
