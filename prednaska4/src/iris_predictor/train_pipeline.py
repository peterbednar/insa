from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from iris_predictor.processors import MeanInputer
from iris_predictor.data_manager import save_pipeline

def run_training():

    pipe = Pipeline(
        [("imput-missing", MeanInputer(variables=["sepal length (cm)", "sepal width (cm)"])),
         ("scale", StandardScaler()),
         ("pca", PCA()),
         ("logistic", LogisticRegression())]
    )

    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipe.fit(X_train, y_train)

    y_test_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Logistic regression test accuracy: {test_acc:.4f}")
    save_pipeline(pipe)

if __name__ == "__main__":
    run_training()