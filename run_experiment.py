import os
import time
import logging
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from load_data import load_dataset
from css_solver import CSSProblemSolver
from svd import svd_error


EXPERIMENT_CONFIG = {
    "dataset_names": ["cmhs", "sonar"],
    # "dataset_names": ["cmhs"],
    "dataset_dir": "datasets",
    "k_values": [5, 10, 15, 20, 25, 30],
    # "k_values": [15, 25],
    # "k_values": [30],
    "t_factor": 2, # t = t_factor * k
    "stochastic_runs": 5,
    "algorithms": [
        {"name": "random",     "is_stochastic": True,  "needs_t": False},
        {"name": "greedy_rec", "is_stochastic": False, "needs_t": False},
        {"name": "lscss",      "is_stochastic": True,  "needs_t": True},
        {"name": "lscss_qr",   "is_stochastic": True,  "needs_t": True},
    ],
    "log_dir": "logs",
    "tensorboard_log_dir": "runs"
}


def setup_logger(log_dir, experiment_name="experiment"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def run_experiment(config):
    logger = setup_logger(config["log_dir"], "css_experiment")
    writer = SummaryWriter(log_dir=os.path.join(config["tensorboard_log_dir"], datetime.now().strftime("%Y%m%d_%H%M%S")))

    all_results = []
    solver = CSSProblemSolver()

    logger.info("Starting experiment batch with configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("-" * 50)

    for dataset_name in config["dataset_names"]:
        logger.info(f"Processing dataset: {dataset_name}")
        try:
            config_path = os.path.join(config["dataset_dir"], dataset_name, "detail.yaml")
            data_matrix = load_dataset(dataset_name, config_path)
            logger.info(f"Loaded dataset {dataset_name} with shape: {data_matrix.shape}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            continue

        for k in config["k_values"]:
            logger.info(f"  Processing k = {k} for dataset {dataset_name}")
            t = config["t_factor"] * k

            try:
                start_time_svd = time.perf_counter()
                baseline_error = svd_error(data_matrix, k)
                runtime_svd_s = time.perf_counter() - start_time_svd
                logger.info(f"    SVD Baseline (k={k}): Error = {baseline_error:.6f}, Time = {runtime_svd_s:.4f}s")
                if baseline_error == 0:
                    logger.warning(f"    SVD Baseline error is 0 for k={k}. Skipping ratio calculations for this k.")
            except Exception as e:
                logger.error(f"    Error calculating SVD baseline for k={k}: {e}")
                continue

            for algo_config in config["algorithms"]:
                algo_name = algo_config["name"]
                logger.info(f"    Running algorithm: {algo_name} (k={k})")

                solve_args = [data_matrix, k]
                if algo_config["needs_t"]:
                    solve_args.append(t)

                runtimes_s = []
                errors = []
                error_ratios = []

                num_actual_runs = config["stochastic_runs"] if algo_config["is_stochastic"] else 1

                for run_idx in range(num_actual_runs):
                    try:
                        start_time_algo = time.perf_counter()
                        selected_indices = solver.solve(algo_name, *solve_args)
                        runtime_algo_s = time.perf_counter() - start_time_algo
                        runtimes_s.append(runtime_algo_s)

                        current_error = solver.get_objective(data_matrix, selected_indices)
                        errors.append(current_error)

                        if baseline_error != 0:
                            current_error_ratio = current_error / baseline_error
                            error_ratios.append(current_error_ratio)
                        else:
                            error_ratios.append(np.nan if current_error != 0 else 1.0)

                        if algo_config["is_stochastic"]:
                            logger.debug(f"      Run {run_idx+1}/{num_actual_runs} for {algo_name}: Error={current_error:.6f}, Time={runtime_algo_s:.4f}s, Ratio={error_ratios[-1] if error_ratios else 'N/A':.4f}")

                    except Exception as e:
                        logger.error(f"      Error running {algo_name} (k={k}), run {run_idx+1}: {e}")
                        errors.append(np.nan)
                        runtimes_s.append(np.nan)
                        error_ratios.append(np.nan)
                        break

                mean_runtime_s = np.nanmean(runtimes_s)
                mean_error = np.nanmean(errors)
                
                if error_ratios:
                    mean_error_ratio = np.nanmean(error_ratios)
                    median_error_ratio = np.nanmedian(error_ratios)
                    std_dev_error_ratio = np.nanstd(error_ratios) # 可选
                else:
                    mean_error_ratio = np.nan
                    median_error_ratio = np.nan


                logger.info(f"      Algorithm: {algo_name} (k={k}) - Results:")
                logger.info(f"        Mean Runtime: {mean_runtime_s:.4f}s")
                logger.info(f"        Mean Error: {mean_error:.6f}")
                if baseline_error != 0:
                    logger.info(f"        Mean Error Ratio: {mean_error_ratio:.4f}")
                    logger.info(f"        Median Error Ratio: {median_error_ratio:.4f}")
                    logger.info(f"        Std Dev Error Ratio: {std_dev_error_ratio:.4f}")
                    if algo_config["is_stochastic"]:
                        logger.info(f"        All Error Ratios for boxplot: {error_ratios}")
                else:
                    logger.info(f"        Error Ratio: N/A (baseline error is zero)")


                writer.add_scalar(f"{dataset_name}/{algo_name}/mean_runtime_s", mean_runtime_s, k)
                writer.add_scalar(f"{dataset_name}/{algo_name}/mean_error", mean_error, k)
                if baseline_error != 0:
                    writer.add_scalar(f"{dataset_name}/{algo_name}/mean_error_ratio", mean_error_ratio, k)
                    writer.add_scalar(f"{dataset_name}/{algo_name}/median_error_ratio", median_error_ratio, k)
                
                # 对于随机算法，可以考虑记录所有运行的误差比，用于后续生成箱线图
                # hparams 记录超参数和最终指标
                if algo_config["is_stochastic"] and error_ratios and not np.isnan(median_error_ratio):
                    # Tensorboard的hparams比较复杂，如果需要可以添加。
                    # 这里主要记录每个k下的中位数误差比，用于后续箱线图的Y轴数据点。
                    # 为了绘制箱线图，我们需要的是所有运行的 error_ratios 列表。
                    pass


                result_entry = {
                    "dataset_name": dataset_name,
                    "k": k,
                    "algorithm": algo_name,
                    "is_stochastic": algo_config["is_stochastic"],
                    "baseline_error_svd": baseline_error,
                    "mean_algorithm_error": mean_error,
                    "mean_error_ratio": mean_error_ratio,
                    "median_error_ratio": median_error_ratio,
                    "all_error_ratios_for_boxplot": error_ratios if algo_config["is_stochastic"] else [mean_error_ratio],
                    "mean_runtime_s": mean_runtime_s,
                    "all_runtimes_s": runtimes_s,
                }
                all_results.append(result_entry)

    logger.info("-" * 50)
    logger.info("Experiment batch finished.")
    logger.info(f"Total results collected: {len(all_results)}")

    writer.close()
    return all_results

if __name__ == "__main__":
    results = run_experiment(EXPERIMENT_CONFIG)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = os.path.join(EXPERIMENT_CONFIG["log_dir"], f"experiment_results_{timestamp}.json")
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_filename}")
