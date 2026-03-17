from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pipe = Pipeline(
    [("scale", StandardScaler()),
     ("feature_selection", SelectFromModel(LogisticRegression(penalty="l1", solver="saga"), max_features=2)),
     ("model", RandomForestClassifier(random_state=1234))]
)

X, y = datasets.load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)

pipe.fit(X_train, y_train)

y_test_pred = pipe.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Model test accuracy: {test_acc:.4f}")

feature_selector = pipe.named_steps["feature_selection"]
selected_features = X.columns[feature_selector.get_support()].to_list()

print(f"Selected features: {selected_features}") # ['petal length (cm)', 'petal width (cm)']

# Production code

SELECTED_FEATURES = ['petal length (cm)', 'petal width (cm)']

prod_steps = [(name, step) for (name, step) in pipe.steps if name != "feature_selection"]
prod_pipe = Pipeline(prod_steps)

X_train = X_train[SELECTED_FEATURES]
X_test = X_test[SELECTED_FEATURES]

prod_pipe.fit(X_train, y_train)

y_test_pred = prod_pipe.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Production model test accuracy: {test_acc:.4f}")
