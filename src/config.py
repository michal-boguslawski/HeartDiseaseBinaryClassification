# src/config.py

import yaml
from dataclasses import dataclass


@dataclass
class Config:
    # General
    target: str
    random_state: int
    test_size: float

    # Model
    model_name: str
    model_params: dict

    # Feature selection
    use_feature_selection: bool
    k_best: int


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)
