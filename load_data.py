import yaml
import numpy as np

from .datasets.sonar.load import load_sonar_dataset


def load_dataset(dataset_name: str, config_path: str, read_label: bool = True) -> np.ndarray:
    if dataset_name == "sonar":
        return load_sonar_dataset(config_path, read_label)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")