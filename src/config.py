# src/config.py
import yaml

from .dataclass import Config


MLFLOW_URI = "http://localhost:5000"


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config.model_validate(config_dict)
