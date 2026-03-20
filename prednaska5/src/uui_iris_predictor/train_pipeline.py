from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from uui_iris_predictor.processors import MeanInputer
from uui_iris_predictor.data_manager import save_pipeline

def run_training():

    pipe = Pipeline(
        [("input_missing", MeanInputer(variables=["sepal_length", "sepal_width"])),
         ("scale", StandardScaler()),
         ("pca", PCA()),
         ("logistic", LogisticRegression())]
    )

    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    X = X.rename(columns={"sepal length (cm)": "sepal_length",
                          "sepal width (cm)": "sepal_width",
                          "petal length (cm)": "petal_length",
                          "petal width (cm)": "petal_width"})
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)

    pipe.fit(X_train, y_train)

    y_test_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Logistic regression test accuracy: {test_acc:.4f}")

    save_pipeline(pipe)

if __name__ == "__main__":
    run_training()