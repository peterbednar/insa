import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

preprocess = Pipeline(
    [("scale", StandardScaler()),
     ("pca", PCA())]
)

X, y = datasets.load_iris(return_X_y=True)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

X_train_val = preprocess.fit_transform(X_train_val, y_train_val)

random_forest = RandomForestClassifier(
    n_estimators=20,
    random_state=np.random.RandomState(1234)
)

scores = cross_val_score(random_forest, X, y, cv=5)
print(f"Random forest cross-validation accuracy: {scores.mean():.4f}+/-{scores.std():.4f}")

# ...

best_model = random_forest
best_model.fit(X_train_val, y_train_val)

X_test = preprocess.transform(X_test)

y_test_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Best model test accuracy: {test_acc:.4f}")
