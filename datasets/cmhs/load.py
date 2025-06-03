import pandas as pd
import os
import yaml
import numpy as np


def _read_dataset_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Dataset config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Parsing YAML file error: {config_path}\nError: {e}")

    required_fields = ['files', 'format', 'n_row', 'n_col']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Config '{config_path}' lacks field(s): '{field}'")

    if config['format'] != 'txt':
        raise ValueError(f"Dataset format '{config['format']}' not supported")

    return config
    
def _read_txt_data(file_path: str) -> np.ndarray:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        df = pd.read_csv(file_path, header=None, sep='\t')
    except Exception as e:
        raise IOError(f"Error reading txt file '{file_path}': {e}")

    return df.to_numpy()


def load_cmhs_dataset(config_path: str) -> np.ndarray:
    config = _read_dataset_config(config_path)
    config_dir = os.path.dirname(config_path)
    
    data_files = config['files']
    
    feature_matrices = []
    for file_name in data_files:
        data_file_path = os.path.join(config_dir, file_name)
        print(f"Reading data from {data_file_path}...")
        feature_matrix = _read_txt_data(data_file_path)
        print(f"File {file_name} read with shape {feature_matrix.shape}")
        feature_matrices.append(feature_matrix)
    
    data_matrix = np.hstack(feature_matrices)
    print(f"Combined data matrix shape: {data_matrix.shape}")
    
    expected_rows, expected_cols = config['n_row'], config['n_col']
    actual_rows, actual_cols = data_matrix.shape
    
    if actual_rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows but actual {actual_rows} rows")
    if actual_cols != expected_cols:
        raise ValueError(f"Expected {expected_cols} cols but actual {actual_cols} cols")
    return data_matrix