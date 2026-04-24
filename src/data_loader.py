from pathlib import Path
import yaml
import sys
import pandas as pd

def load_config():
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config

def load_data(path):
        if not Path(path).exists():
            print(f"Error path not exists : {path}")
            sys.exit(1)
        try:  
            df = pd.read_csv(path)   
        except FileNotFoundError:
             print("Error could not found file.")
             sys.exit(1)
        else:
             print("OK loaded file :")     
             
        return df       