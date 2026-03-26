import pytest
import copy
from fastapi.testclient import TestClient
from uui_iris_predictor.endpoint import app
from uui_iris_predictor.version import __version__
from uui_iris_predictor.config import PIPELINE_VERSION
from uui_iris_predictor.train_pipeline import run_training
from uui_iris_predictor.data_manager import pipeline_exists

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
        "predictions": [{"label": 1, "proba": {"0":0.020317443545718548, "1":0.796773079482379,  "2": 0.1829094769719025}}],
        "pipeline": PIPELINE_VERSION
    },
    "id": 1
}

@pytest.fixture(scope="module", autouse=True)
def setup_module():
    if not pipeline_exists():
        run_training()

client = TestClient(app)

def test_info():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"version" : __version__, "pipeline_version": PIPELINE_VERSION}

def test_rpc_call_1():
    response = client.post("/api/v2/rpc", json=TEST_REQUEST_1)

    assert response.status_code == 200
    assert response.json() == TEST_RESPONSE_1

TEST_REQUEST_2 = {
    "jsonrpc": "2.0",
    "method": "pipeline/predict",
    "params": {
        "data": [
            {"sepal_length": 6.1, "sepal_width": None, "petal_length": 4.6, "petal_width": 1.4}
        ]
    },
    "id": 1
}

def test_rpc_call_2():
    response = client.post("/api/v2/rpc", json=TEST_REQUEST_2)

    assert response.status_code == 422

def test_metrics():
    client.post("/api/v2/rpc", json=TEST_REQUEST_1)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert len(response.text.splitlines()) == 95
