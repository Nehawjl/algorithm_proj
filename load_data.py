import yaml
import numpy as np
import pandas as pd
import os


def _read_dataset_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Dataset config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Parsing YAML file error: {config_path}\nError: {e}")

    required_fields = ['file', 'format', 'n_row', 'n_col']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Config '{config_path}' lacks field(s): '{field}'")

    if config['format'] != 'csv':
        raise ValueError(f"Dataset format '{config['format']}' not supported")

    return config
    

def _read_csv_data(file_path: str, read_label: bool) -> np.ndarray:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        df = pd.read_csv(file_path, header=None, sep=',')
    except Exception as e:
        raise IOError(f"Error reading CSV file '{file_path}': {e}")

    result_df = df if read_label else df.iloc[:, :-1]
    return result_df.to_numpy()


def load_sonar_dataset(config_path: str, read_label: bool = True) -> np.ndarray:
    config = _read_dataset_config(config_path)
    config_dir = os.path.dirname(config_path)
    data_file_path = os.path.join(config_dir, config['file'])

    print(f"Reading data from {data_file_path}...")
    data_matrix = _read_csv_data(data_file_path, read_label)
    print(f"Data read done with shape {data_matrix.shape}")

    expected_rows, expected_cols = config['n_row'], config['n_col'] if read_label else config['n_col'] - 1
    actual_rows, actual_cols = data_matrix.shape
    if actual_rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows but actual {actual_rows} rows")
    if actual_cols != expected_cols:
        raise ValueError(f"Expected {expected_cols} cols but actual {actual_cols} cols")
    return data_matrix


def load_dataset(dataset_name: str, config_path: str, read_label: bool = True) -> np.ndarray:
    if dataset_name == "sonar":
        return load_sonar_dataset(config_path, read_label)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")