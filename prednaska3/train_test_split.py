from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

preprocess = Pipeline(
    [("scale", StandardScaler()),
     ("pca", PCA())]
)

X, y = datasets.load_iris(return_X_y=True)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

X_train_val = preprocess.fit_transform(X_train_val, y_train_val)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=1234)

log_regression = LogisticRegression()
log_regression.fit(X_train, y_train)

y_val_pred = log_regression.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Logistic regression validation accuracy: {val_acc:.4f}")

# ...

best_model = log_regression
best_model.fit(X_train_val, y_train_val)

X_test = preprocess.transform(X_test)

y_test_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Best model test accuracy: {test_acc:.4f}")
