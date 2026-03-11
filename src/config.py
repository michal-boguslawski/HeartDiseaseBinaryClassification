# src/config.py
import yaml

from .dataclass import Config


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config.model_validate(config_dict)
