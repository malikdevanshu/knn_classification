from src.config import load_config
from src.data_loader import load_data
from src.evaluate import predictions
from src.preprocessing import feature_target_split, splitting
from src.train import model_training


def main() -> None:
    config = load_config()
    data_path = config["paths"]["data_path"]
    state = config["random"]["random_state"]
    df = load_data(data_path)
    features, target = feature_target_split(df)
    X_train, X_test, y_train, y_test = splitting(features, target, int(state))
    predict = model_training(X_train=X_train, X_test=X_test, y_train=y_train)

    predictions(y_test, predict)


if __name__ == "__main__":
    main()
