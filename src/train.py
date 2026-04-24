import pandas as pd
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


def model_training(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
) -> pd.Series:
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])

    param_grid = {
        "knn__n_neighbors": [3, 5, 7, 9],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    return pd.Series(y_pred, dtype=str)
