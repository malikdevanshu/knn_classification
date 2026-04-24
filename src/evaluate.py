import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix  # type: ignore


def predictions(y_pred: pd.Series, y_test: pd.Series) -> None:
    acc = accuracy_score(y_test, y_pred)
    con_mat = confusion_matrix(y_test, y_pred)
    print(acc)
    print(f"\n {con_mat} ")
