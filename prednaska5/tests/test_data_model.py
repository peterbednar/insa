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

def test_request():
    RpcRequest(**TEST_REQUEST_1)

def test_response():
    RpcResponse(**TEST_RESPONSE_1)
