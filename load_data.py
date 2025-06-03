import yaml
import numpy as np

from datasets.sonar.load import load_sonar_dataset
from datasets.cmhs.load import load_cmhs_dataset


def load_dataset(dataset_name: str, config_path: str) -> np.ndarray:
    if dataset_name == "sonar": # 208 x 60
        return load_sonar_dataset(config_path)
    if dataset_name == "cmhs": # 2205 x 43680
        return load_cmhs_dataset(config_path)
    raise ValueError(f"Dataset {dataset_name} not supported")