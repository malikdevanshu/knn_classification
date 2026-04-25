import pandas as pd
import pytest

from src.preprocessing import feature_target_split, splitting
from src.train import model_training


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature1": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11,
                12,
                13,
                14,
                15,
                16,
                12,
                12,
                12,
                34,
                23,
                23,
                24,
                23,
                12,
                13,
                24,
                23,
                20,
            ],
            "feature2": [
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                11,
                12,
                13,
                14,
                15,
                16,
                12,
                12,
                12,
                34,
                23,
                23,
                24,
                23,
                12,
                13,
                24,
                23,
                20,
            ],
            "Label": [
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
            ],
        }
    )


@pytest.fixture
def split_data(sample_df: pd.DataFrame) -> tuple:
    features, target = feature_target_split(sample_df)
    return splitting(features, target, state=42)


# test feature target split shape


def feature_target_shape(sample_df: pd.DataFrame) -> None:
    features, target = feature_target_split(sample_df)
    assert features.shape[1] == sample_df.shape[1] - 1
    assert len(target) == len(sample_df)


def test_feature_target_split_no_target_in_features(sample_df: pd.DataFrame) -> None:
    """The target column must not appear in the features DataFrame."""
    features, target = feature_target_split(sample_df)
    assert "Label" not in features.columns


# test the train test split


def test_train_test_split(split_data: tuple) -> None:
    X_train, X_test, y_train, y_test = split_data
    assert len(X_train) + len(X_test) == 29
    assert len(y_train) + len(y_test) == 29


def test_model_tarining(split_data: tuple) -> None:
    X_train, X_test, y_train, y_test = split_data
    preds = model_training(X_train, X_test, y_train)
    assert isinstance(preds, pd.Series)


def test_pred_test_shape(split_data: tuple) -> None:
    X_train, X_test, y_train, y_test = split_data
    preds = model_training(X_train=X_train, X_test=X_test, y_train=y_train)
    assert len(X_test) == len(preds)
