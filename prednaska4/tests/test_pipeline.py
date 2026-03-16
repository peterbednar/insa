import pytest
import numpy as np
import pandas as pd
from uui_iris_predictor.train_pipeline import run_training
from uui_iris_predictor.data_manager import load_pipeline

@pytest.fixture(scope="module", autouse=True)
def setup_module():
    run_training()

def test_pipeline_score():
    pipe = load_pipeline()

    X = pd.read_csv("tests/data/test1_input.csv")
    y = pd.read_csv("tests/data/test1_output.csv")

    y_pred = pipe.predict(X)

    assert np.array_equal(y["label"].to_numpy(), y_pred)
