from src.data_loader import load_config, load_data
from src.preprocessing import feature_target_split, splitting
from src.train import model_training
from src.evaluate import predictions
def main():
    config = load_config()
    data_path = config["paths"]["data_path"]
    state = config["random"]["random_state"]
    df = load_data(data_path)
    features, target = feature_target_split(df)
    X_train, X_test, y_train, y_test = splitting(features, target, state)
    predict = model_training(X_train=X_train, X_test=X_test, y_train=y_train)

    eval = predictions(y_test, predict)


if __name__ == "__main__":
    main()
 