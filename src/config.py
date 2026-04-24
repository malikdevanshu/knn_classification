from pathlib import Path
from typing import Dict, Union
import yaml


def load_config() -> Dict[str, Union[str, int]]:
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config or {}
