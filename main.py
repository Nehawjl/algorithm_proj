import os
import cProfile, pstats

from load_data import load_dataset
from css_solver import CSSProblemSolver
from svd import svd_error


dataset_name = "cmhs" # 2205 x 43680
dataset_dir = "datasets"

k = 5
t = 2 * k


if __name__ == "__main__":
    # dataset_name = "sonar" # 208 x 60

    config_path = os.path.join(dataset_dir, dataset_name, "detail.yaml")
    data_matrix = load_dataset(dataset_name, config_path)


    baseline = svd_error(data_matrix, k)
    print(f"baseline: {baseline}")

    solver = CSSProblemSolver()
    profiler = cProfile.Profile()

    profiler.enable()
    selected_indices_lscss = solver.solve('lscss_qr', data_matrix, k, t)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

    error_lscss = solver.get_objective(data_matrix, selected_indices_lscss)
    print(f"selected: {selected_indices_lscss}\nerror: {error_lscss}\nerror ratio: {error_lscss/baseline}")