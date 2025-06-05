# CS240 Algorithm Project
This repository is our experiment crafts containing implementation and analysing of Column Subset Selection algorithms.

## Compared algorithms
The implementations strictly follow details proposed in the original work.
- [Greedy](https://ieeexplore.ieee.org/document/6137220)
- [LSCSS](https://proceedings.neurips.cc/paper_files/paper/2024/file/f9a50cf037f5ca2f687e3cd70b572c6f-Paper-Conference.pdf)
- LSCSS-QR -- Our enhanced method using QR decomposition for efficient objective value calculation and subset matrix maintenance.

## Structure
```
.
├── css
│   ├── bf_css.py                                # brute-force (only for small dataset)
│   ├── greedy_css.py                            # greedy algorithm
│   ├── __init__.py
│   ├── lscss.py                                 # LSCSS
│   ├── lscss_qr.py                              # LSCSS-QR
│   ├── random_css.py
│   └── utility.py
├── css.ipynb                                    # usage example
├── css_solver.py
├── datasets
│   ├── cmhs                                     # please download manually from the source website in `detail.yaml`
│   |   ├── load.py
│   │   ├── detail.yaml
│   │   └── ...
│   ├── __init__.py
│   └── sonar
│       ├── detail.yaml
│       ├── load.py
│       └── sonar.csv
├── __init__.py
├── load_data.py
├── logs                                         # results
│   ├── experiment_results_20250604_033443.json  # measured results
├── main.py
├── output_plots
│   ├── CMHS_error_ratio_vs_k.pdf
│   ├── CMHS_runtime_vs_k.pdf
├── plot.py
├── README.md
├── run_experiment.py
├── runs                                         # tensorboard record
└── svd.py                                       # baseline
```

## Usage
- run `python run_experiment.py`. the result will be logged under `./logs`
- set the result json file path in `plot.py` and run `python plot.py`, the result will be plotted under `./output_plots`

## Accuracy and Runtime on CHMS
![Error Ratio](output_plots/CMHS_error_ratio_vs_k.pdf)
![Runtime](output_plots/CMHS_error_ratio_vs_k.pdf)
