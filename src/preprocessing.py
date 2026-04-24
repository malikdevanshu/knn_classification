import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore


def feature_target_split(
    data: pd.DataFrame, target_col: str = "Label"
) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.DataFrame(data)
    features = data.drop(target_col, axis=1)
    target = data[target_col]
    return features, target


def splitting(
    features: pd.DataFrame, target: pd.Series, state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=state
    )
    return (X_train, X_test, y_train, y_test)
