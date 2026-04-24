from src.config_loader import load_config
import pandas as pd
def main():
    config = load_config()
    data_path = config["paths"]["data_path"]
    output_path = config["paths"]["output_path"]
    df = pd.read_csv(data_path)
    df.head().to_csv(output_path)


if __name__ == "__main__":
    main()
