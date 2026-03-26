import numpy as np
import pandas as pd
from uui_iris_predictor.processors import MeanInputer

TEST_DATA_1 = {
    "x1": [1, 2, np.nan, 3, 4, np.nan, 5, 6, np.nan],
    "x2": [1, np.nan, 2, np.nan, 3, np.nan, 4, np.nan, 5]
}

def test_mean_inputer():
    df = pd.DataFrame.from_dict(TEST_DATA_1)
    mean_inputer = MeanInputer(variables=["x1"])

    mean_inputer.fit(df)
    tdf = mean_inputer.transform(df)

    assert tdf["x1"].isnull().sum() == 0
    assert tdf["x2"].isnull().sum() == df["x2"].isnull().sum()

    mean = df["x1"].mean()
    x1_nan_index = df[df["x1"].isna()].index
    x1_nan_values = tdf["x1"][x1_nan_index].to_numpy()

    assert np.all(np.equal(x1_nan_values, mean))
    # assert np.all(np.isclose(x1_nan_values, mean))
