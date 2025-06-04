import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator

# --- Configuration ---
RESULTS_FILE_PATH = "logs/experiment_results_20250604_033443.json"
OUTPUT_DIR = "output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define consistent colors and display names for algorithms
ALGORITHM_STYLES = {
    # "random": {"color": "#F8766D", "marker": "o", "linestyle": "-", "display_name": "Random"},
    "greedy_rec": {"color": "#00BFC4", "marker": "s", "linestyle": "-", "display_name": "Greedy"},
    "lscss": {"color": "#7CAE00", "marker": "^", "linestyle": "--", "display_name": "LSCSS"},
    "lscss_qr": {"color": "#C77CFF", "marker": "D", "linestyle": "-.", "display_name": r'LSCSS-QR'}, # Changed linestyle for better distinction
}

# --- Global Plotting Style Configuration ---
# Using a context manager for style is also an option if you prefer not to set globally
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif', # Using a serif font often preferred in academic settings
    'font.serif': ['Times New Roman', 'DejaVu Serif'], # Fallback to DejaVu Serif
    'axes.labelsize': 14,       # X and Y axis labels
    'xtick.labelsize': 12,      # X-axis tick labels
    'ytick.labelsize': 12,      # Y-axis tick labels
    'legend.fontsize': 11,
    'legend.title_fontsize': 12,
    'figure.titlesize': 16,     # Main figure title (rarely used with subplots)
    'axes.titlesize': 16,       # Subplot (Axes) title
    'grid.linestyle': ':',      # Dotted grid lines
    'grid.alpha': 0.7,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
})

# --- Helper Function to Prepare Data for Plotting ---
def prepare_plot_data(df_results, metric_list_column_name, target_metric_name):
    """
    Expands the list of metrics (e.g., error ratios from multiple runs)
    into individual rows for plotting. This version retains robustness.
    """
    plot_data_list = []
    # Filter upfront for algorithms we care about
    df_filtered_algos = df_results[df_results["algorithm"].isin(ALGORITHM_STYLES.keys())].copy()

    for _, row in df_filtered_algos.iterrows():
        metric_values = row[metric_list_column_name]
        # Ensure k is numeric early on
        k_value = pd.to_numeric(row["k"], errors='coerce')
        if pd.isna(k_value):
            continue # Skip if k cannot be converted to a number

        shared_data = {
            "dataset_name": row["dataset_name"],
            "k": k_value,
            "algorithm": row["algorithm"],
            "is_stochastic": row["is_stochastic"],
            "baseline_error_svd": row.get("baseline_error_svd", np.nan)
        }

        if isinstance(metric_values, list):
            for val in metric_values:
                if pd.notna(val):
                    entry = shared_data.copy()
                    entry[target_metric_name] = val
                    plot_data_list.append(entry)
        elif pd.notna(metric_values): # Handles single scalar value case
            entry = shared_data.copy()
            entry[target_metric_name] = metric_values
            plot_data_list.append(entry)
        # If metric_values is None or NaN (and not a list), it's skipped implicitly

    return pd.DataFrame(plot_data_list)

# --- Plotting Function ---
def plot_metric_comparison(df_plot_ready, dataset_name, metric_col, y_label, base_title, output_filename, show_individual_points=False):
    """
    Generates a median line chart with shaded min-max range for a given metric.
    """
    fig, ax = plt.subplots(figsize=(10, 6)) # Adjusted figsize for better aspect ratio

    df_dataset = df_plot_ready[df_plot_ready['dataset_name'] == dataset_name].copy()

    if df_dataset.empty:
        print(f"No data to plot for {dataset_name} and metric {metric_col}.")
        plt.close(fig)
        return

    # Ensure 'k' is numeric for correct sorting and plotting
    # This should already be handled by prepare_plot_data, but as a safeguard:
    df_dataset['k'] = pd.to_numeric(df_dataset['k'])
    
    # Filter out any NaN k values that might have slipped through (e.g. if prepare_plot_data was bypassed)
    df_dataset.dropna(subset=['k'], inplace=True)
    if df_dataset.empty:
        print(f"No valid 'k' values to plot for {dataset_name} and metric {metric_col} after NaN drop.")
        plt.close(fig)
        return
        
    algorithms = sorted([algo for algo in df_dataset['algorithm'].unique() if algo in ALGORITHM_STYLES])
    k_values = sorted(df_dataset['k'].unique())

    legend_handles = []
    legend_labels = []

    for algo_name in algorithms:
        df_algo = df_dataset[df_dataset['algorithm'] == algo_name]
        style = ALGORITHM_STYLES[algo_name] # Assumes algo_name is in ALGORITHM_STYLES
        
        summary_data = df_algo.groupby('k')[metric_col].agg(['median', 'min', 'max']).reset_index()
        summary_data = summary_data.sort_values(by='k')

        if summary_data.empty:
            continue

        line, = ax.plot(
            summary_data['k'],
            summary_data['median'],
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            color=style.get("color", "gray"),
            label=style.get("display_name", algo_name), # Use label for automatic legend handles
            markersize=plt.rcParams['lines.markersize'], # Use global markersize
            linewidth=plt.rcParams['lines.linewidth'],   # Use global linewidth
            zorder=3
        )
        # Collect handles and labels directly if preferred, but ax.get_legend_handles_labels() is also good
        # if display_name not in legend_labels: # This check is implicitly handled by ax.plot's label argument
        #     legend_handles.append(line)
        #     legend_labels.append(display_name)

        ax.fill_between(
            summary_data['k'],
            summary_data['min'],
            summary_data['max'],
            color=style.get("color", "gray"),
            alpha=0.2,
            where=summary_data['min'] < summary_data['max'],
            interpolate=False, # Step-like fill suitable for discrete k
            zorder=1
        )

        if show_individual_points:
            stochastic_points_df = df_algo.groupby('k').filter(lambda x: x[metric_col].nunique() > 1)
            if not stochastic_points_df.empty:
                sns.stripplot(
                    x='k', y=metric_col, data=stochastic_points_df, ax=ax,
                    color=style.get("color", "gray"), alpha=0.35, jitter=0.1, s=4,
                    zorder=2, label='_nolegend_'
                )

    # Aesthetics and Final Touches
    ax.set_xlabel("#k") # Revised X-axis label
    ax.set_ylabel(y_label)
    plot_title = f"{base_title} on {dataset.capitalize()} Dataset"
    ax.set_title(plot_title)

    if not k_values: # No k_values to plot
        print(f"Warning: No k_values found for dataset {dataset_name}, metric {metric_col}.")
        plt.close(fig)
        return

    ax.set_xticks(k_values)
    # ax.set_xticklabels([str(int(k)) if k.is_integer() else str(k) for k in k_values])
    ax.set_xticklabels([str(int(k)) for k in k_values])
    
    # Ensure x-axis major ticks are integers if all k_values are integers.
    # if all(float(k_val).is_integer() for k_val in k_values):
    #      ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=max(1, len(k_values))))

    # Add some padding to the x-axis so markers aren't flush with the spines
    ax.margins(x=0.04) # e.g., 4% padding on each side of the x-axis

    # Create and place the legend
    # Get handles and labels from what was plotted on ax
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Place legend outside the plot to the right
        # The rect argument in fig.tight_layout will make space
        # ax.legend(handles, labels, title="Algorithm", loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 1))
    
    # Adjust layout to make space for legend outside if needed
    # rect=[left, bottom, right, top] in figure coordinates
    # fig.tight_layout(rect=[0, 0, 0.85, 1]) # Leave 15% on the right for the legend
    fig.tight_layout(rect=[0, 0, 1, 1]) # Leave 15% on the right for the legend

    plt.savefig(output_filename, dpi=300, bbox_inches='tight') # bbox_inches='tight' is important
    print(f"Saved plot: {output_filename}")
    plt.close(fig)

# --- Main Script ---
if __name__ == "__main__":
    try:
        with open(RESULTS_FILE_PATH, 'r') as f:
            results_raw = json.load(f)
        df_results_raw = pd.DataFrame(results_raw)
    except FileNotFoundError:
        print(f"ERROR: Results file not found at {RESULTS_FILE_PATH}")
        print("Please run the experiment script first or update the RESULTS_FILE_PATH.")
        exit(1)
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {RESULTS_FILE_PATH}. File might be corrupted.")
        exit(1)

    if df_results_raw.empty:
        print("No results found in the JSON file.")
        exit(0)

    # Ensure 'k' is numeric in the raw DataFrame before passing to prepare_plot_data
    if 'k' in df_results_raw.columns:
        df_results_raw['k'] = pd.to_numeric(df_results_raw['k'], errors='coerce')
        # Optionally, drop rows where k became NaN if that's critical
        # df_results_raw.dropna(subset=['k'], inplace=True)
    else:
        print("ERROR: 'k' column not found in the results data.")
        exit(1)

    df_plot_error_ratio = prepare_plot_data(df_results_raw, 'all_error_ratios_for_boxplot', 'error_ratio_run')
    df_plot_runtime = prepare_plot_data(df_results_raw, 'all_runtimes_s', 'runtime_s_run')

    unique_datasets = df_results_raw['dataset_name'].unique()

    for dataset in unique_datasets:
        print(f"\nGenerating plots for dataset: {dataset}...")

        # For Error Ratio
        df_er_dataset_check = df_plot_error_ratio[df_plot_error_ratio['dataset_name'] == dataset]
        if not df_er_dataset_check.empty and not df_er_dataset_check['error_ratio_run'].isnull().all():
            plot_metric_comparison(
                df_plot_ready=df_plot_error_ratio,
                dataset_name=dataset,
                metric_col='error_ratio_run',
                # y_label="Error Ratio (Algorithm Error / SVD Error)", # Kept descriptive label
                y_label="Error Ratio", # Kept descriptive label
                base_title="Error Ratio vs. #k", # Base title, dataset name added in function
                output_filename=os.path.join(OUTPUT_DIR, f"{dataset}_error_ratio_vs_k.pdf"), # More descriptive filename
                show_individual_points=False # Set to True to see individual points for stochastic methods
            )
        else:
            print(f"Skipping error ratio plot for {dataset} due to missing or all-NaN data.")

        # For Runtime
        df_rt_dataset_check = df_plot_runtime[df_plot_runtime['dataset_name'] == dataset]
        if not df_rt_dataset_check.empty and not df_rt_dataset_check['runtime_s_run'].isnull().all():
             plot_metric_comparison(
                df_plot_ready=df_plot_runtime,
                dataset_name=dataset,
                metric_col='runtime_s_run',
                y_label="Runtime (s)",
                base_title="Runtime vs. #k", # Base title
                output_filename=os.path.join(OUTPUT_DIR, f"{dataset}_runtime_vs_k.pdf"), # More descriptive filename
                show_individual_points=False
            )
        else:
            print(f"Skipping runtime plot for {dataset} due to missing or all-NaN data.")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")