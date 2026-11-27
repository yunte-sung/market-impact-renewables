# Main figure: causal effect (CATE) of wind next to the observed average price, by wind penetration.
# Needs the results from DML_overall_wind.py for the same zone.
# Run from repo root: uv run python -m double_machine_learning.dml_vs_mean_plot_wind -> UK (original)
# ZONE=DE_LU uv run python -m double_machine_learning.dml_vs_mean_plot_wind   -> Germany

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
import matplotlib as mpl

from analysis_settings import get_analysis_settings, smooth, filter_period
from data_pipeline import schema

# Settings for the chosen zone
S = get_analysis_settings()
c = S["cols"]
PENETRATION = c[schema.WIND_PENETRATION]
PRICE = c[schema.PRICE]
CURRENCY = S["currency"]

# Load dataset
df = pd.read_csv(S["data_path"])
df[S["index_col"]] = pd.to_datetime(df[S["index_col"]])
df.set_index(S["index_col"], inplace=True)
df = filter_period(df, S)
print(f"Period {S['period_start']} to {S['period_end']}: {len(df)} hours")

# Extract necessary columns
df = df[[PENETRATION, PRICE]]

# Sort the dataset by penetration for smoother sliding window
df = df.sort_values(PENETRATION)

# Define window size and step size
window_size = 10000
step_size = 1000

# Initialize lists to store results
mean_penetrations = []
mean_prices = []
ci_lower = []
ci_upper = []


# Bootstrap function to calculate mean and CI
def apply_bootstrap(data, n_resamples=1000):
    # Perform bootstrap resampling
    res = bootstrap((data,), np.mean, n_resamples=n_resamples, method='basic', confidence_level=0.80)
    return res.confidence_interval.low, res.confidence_interval.high, np.mean(data)


# Slide over the dataset with the specified window and step
for i in range(0, len(df) - window_size + 1, step_size):
    # Select the current window of 10,000 observations
    window = df.iloc[i:i + window_size]

    # Extract the data for bootstrap
    penetration_data = window[PENETRATION].values
    price_data = window[PRICE].values

    # Perform bootstrap for wind penetration and price
    penetration_ci_low, penetration_ci_high, penetration_mean = apply_bootstrap(penetration_data)
    price_ci_low, price_ci_high, price_mean = apply_bootstrap(price_data)

    # Store the results
    mean_penetrations.append(penetration_mean)
    mean_prices.append(price_mean)
    ci_lower.append(price_ci_low)
    ci_upper.append(price_ci_high)

# Load wind results
results_dataset_wind = pd.read_csv(S["output_dir"] / f"results_wind{S['file_suffix']}.csv")

# Group by mean wind penetration and calculate mean and quantiles for CATE
mean_cate_df_wind = results_dataset_wind.groupby('mean_wind_penetration')['cate'].agg(['mean']).reset_index()
mean_cate_df_wind['lower_ci'] = results_dataset_wind.groupby('mean_wind_penetration')['cate'].quantile(0.1).values
mean_cate_df_wind['upper_ci'] = results_dataset_wind.groupby('mean_wind_penetration')['cate'].quantile(0.9).values

# Apply Gaussian smoothing (sigma 0 = none, see analysis_settings)
sigma_smoothing = S["plot_smoothing_sigma"]
if sigma_smoothing is None: # original rule, kept for the UK
    sigma_smoothing = 1.5 * np.std(mean_cate_df_wind['mean'])
print(f"Smoothing sigma: {sigma_smoothing:.2f} windows")
mean_cate_df_wind['smoothed_mean'] = smooth(mean_cate_df_wind['mean'], sigma_smoothing)
mean_cate_df_wind['smoothed_lower_ci'] = smooth(mean_cate_df_wind['lower_ci'], sigma_smoothing)
mean_cate_df_wind['smoothed_upper_ci'] = smooth(mean_cate_df_wind['upper_ci'], sigma_smoothing)

# Set Helvetica font globally
plt.rcParams.update({
    'font.family': 'Helvetica',
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'axes.labelsize': 16,
})

# Create a plot with shared y-axis for penetration
fig, ax1 = plt.subplots(figsize=(6, 6), dpi=300)
ax1.plot(mean_cate_df_wind['smoothed_mean'], mean_cate_df_wind['mean_wind_penetration'],
         color='b', label='Causal estimate', lw=4)
ax1.fill_betweenx(mean_cate_df_wind['mean_wind_penetration'],
                  mean_cate_df_wind['smoothed_lower_ci'],
                  mean_cate_df_wind['smoothed_upper_ci'],
                  color='c', alpha=.3)  # Orange fill
ax1.set_xlabel(f'CATE [{CURRENCY}/MWh]', color='b', fontsize=16, labelpad=10)
ax1.set_ylabel('Forecasted penetration [%]', fontsize=16, labelpad=10)
ax1.tick_params(axis='x', labelcolor='b', labelsize=14)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax2 = ax1.twiny()  # Create a secondary x-axis sharing the same y-axis
ax2.plot(mean_prices, mean_penetrations, color='k', label='Observational mean', lw=1, alpha=.5, ls='--')
ax2.fill_betweenx(mean_penetrations, ci_lower, ci_upper, color='gray', alpha=0.1)  # Crimson fill
ax2.set_xlabel(f'Price [{CURRENCY}/MWh]', color='gray', fontsize=16, labelpad=10)
ax2.tick_params(axis='x', labelcolor='gray', labelsize=14)
fig.legend(fontsize=14, ncol=1, bbox_to_anchor=(0.58, 0.365), frameon=True)
plt.tight_layout()
plt.savefig(S["output_dir"] / 'mean_vs_cate_wind.png', format='png', dpi=600)
plt.show()
