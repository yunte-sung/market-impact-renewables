# sliding window DML estimate of the effect of wind on day-ahead prices
# run from repo root: uv run python -m double_machine_learning.DML_overall_wind     -> UK original
# ZONE=DE_LU N_ITER=5 uv run python -m double_machine_learning.DML_overall_wind   -> Germany, quick test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d

from analysis_settings import get_analysis_settings
from data_pipeline import schema
from double_machine_learning.DML_utils_wind import residualize_data, fit_residualized_model

# Settings for the chosen data
S = get_analysis_settings()
c = S["cols"]
PENETRATION = c[schema.WIND_PENETRATION]
CURRENCY = S["currency"]
S["output_dir"].mkdir(parents=True, exist_ok=True)
results_path = S["output_dir"] / f"results_wind{S["file_suffix"]}.csv"
print(f"Zone {S['zone']}: {S['n_iterations']} bootstraps per window -> {results_path}")


# Load data
df = pd.read_csv(S["data_path"])
df[S["index_col"]] = pd.to_datetime(df[S["index_col"]])
df.set_index(S["index_col"], inplace=True)

# Scale the relevant columns (MW -> GW)
df[c[schema.LOAD_FORECAST]] = df[c[schema.LOAD_FORECAST]] / 1000
df[c[schema.WIND_FORECAST]] = df[c[schema.WIND_FORECAST]] / 1000
df[c[schema.SOLAR_FORECAST]] = df[c[schema.SOLAR_FORECAST]] / 1000


# Sort the DataFrame by wind penetration
df = df.sort_values(by=PENETRATION)

# Parameters for sliding window
window_size = 10000
step_size = 1000
n_iterations = S["n_iterations"]  # Number of bootstraps

# Store CATE and corresponding penetration levels for each window
results = []

# Total number of windows
total_windows = (len(df) - window_size) // step_size + 1

# Bootstrap for one window (windows are run in parallel across CPU cores)
def bootstrap_window(window_data):
    cates = []
    for _ in range(n_iterations):
        random_subset = window_data.sample(n=min(len(window_data), window_size), replace=True)
        df_residualized = residualize_data(random_subset, cols=c)
        res = fit_residualized_model(df_residualized)
        cates.append(res[0])  # Extract the first (and only) coefficient
    return cates


# Sliding window analysis
starts = range(0, len(df) - window_size + 1, step_size)
window_cates = Parallel(n_jobs=-1, return_as='generator')(
    delayed(bootstrap_window)(df.iloc[start:start + window_size]) for start in starts
)

for start, cates in tqdm(zip(starts, window_cates), total=total_windows, desc='Windows'):
    # Calculate the mean wind penetration for the current window
    mean_wind_penetration = df.iloc[start:start + window_size][PENETRATION].mean()

    # Append the mean wind penetration and CATE to results
    for cate in cates:
        results.append({
            'mean_wind_penetration': mean_wind_penetration,
            'cate': cate
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save raw results
results_df.to_csv(results_path, index=False)
results_dataset = pd.read_csv(results_path)

# Group by mean wind penetration and calculate mean and quantiles for CATE
mean_cate_df = results_dataset.groupby('mean_wind_penetration')['cate'].agg(['mean']).reset_index()
mean_cate_df['lower_ci'] = results_dataset.groupby('mean_wind_penetration')['cate'].quantile(0.1).values
mean_cate_df['upper_ci'] = results_dataset.groupby('mean_wind_penetration')['cate'].quantile(0.9).values

# Apply Gaussian smoothing
sigma_smoothing = np.var(mean_cate_df['mean'])
print(f"Smoothing sigma: {sigma_smoothing:.2f} windows")
mean_cate_df['smoothed_mean'] = gaussian_filter1d(mean_cate_df['mean'], sigma=sigma_smoothing)
mean_cate_df['smoothed_lower_ci'] = gaussian_filter1d(mean_cate_df['lower_ci'], sigma=sigma_smoothing)
mean_cate_df['smoothed_upper_ci'] = gaussian_filter1d(mean_cate_df['upper_ci'], sigma=sigma_smoothing)

# Plotting the smoothed mean CATE against mean wind penetration with 80% CI
plt.figure(figsize=(6, 6), dpi=300)
plt.plot(mean_cate_df['smoothed_mean'], mean_cate_df['mean_wind_penetration'], marker='', linestyle='-',
         color='b', label='Mean', lw=3)
plt.fill_betweenx(mean_cate_df['mean_wind_penetration'], mean_cate_df['smoothed_lower_ci'],
                  mean_cate_df['smoothed_upper_ci'],
                  color='c', alpha=.3, label='80% CI')
plt.ylabel('Forecasted wind penetration (%)', fontsize=16)  # Now this is on y-axis
plt.xlabel(f'CATE ({CURRENCY}/MWh)', fontsize=16)  # Now this is on x-axis
plt.axvline(0, color='gray', linestyle='-', lw=.8)  # Add a vertical line at x=0
plt.grid(axis='x', linestyle='-', alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
if S["zone"] == "UK":
    plt.xlim(-9, 0)  # fixed range from the original UK figure; other zones use the data's range
plt.savefig(S["output_dir"] / "cate_wind.png", dpi=300)
plt.show()

plt.figure(figsize=(6, 6), dpi=300)
plt.scatter(results_df['cate'], results_df['mean_wind_penetration'], alpha=0.2, color='c',
            label='Individual CATE estimates')
plt.axvline(0, color='k', linestyle='-', lw=1, alpha=0.5)  # Add a vertical line at x=0
plt.xlabel(f'CATE ({CURRENCY}/MWh)', fontsize=16)
plt.ylabel('Forecasted wind penetration (%)', fontsize=16)
plt.grid(axis='x', linestyle='-', alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(S["output_dir"] / "cate_wind_individual.png", dpi=300)
plt.show()
