import numpy as np
from sklearn import datasets
import requests
import time

BATCH_SIZE = 5

def predict(data):
    id = 0
    random_state = np.random.RandomState(1234)
    print("Press CTRL+C to quit")
    while True:
        sample = X.sample(n=BATCH_SIZE, random_state=random_state)

        request = {
            "jsonrpc": "2.0",
            "method": "pipeline/predict",
                "params": {
                    "data": sample.to_dict(orient="records")
                },
            "id": id
        }

        response = requests.post("http://localhost:8000/api/v2/rpc", json=request)
        print(f"{id}: status code {response.status_code}")

        id += 1
        time.sleep(5)

if __name__ == "__main__":
    X, _ = datasets.load_iris(return_X_y=True, as_frame=True)
    X = X.rename(columns={"sepal length (cm)": "sepal_length",
                          "sepal width (cm)": "sepal_width",
                          "petal length (cm)": "petal_length",
                          "petal width (cm)": "petal_width"})
    predict(X)
