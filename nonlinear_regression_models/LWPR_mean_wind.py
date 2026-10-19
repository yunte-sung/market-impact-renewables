# The original `from LWPR_mean.utils_mean import *` pointed to a module that was never shared.
# The helpers below reconstruct it from the LWPR code in LWPR_quantile_wind.py.
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


# Generate polynomial features (copied from LWPR_quantile_wind.py)
def polynomial_features(x, order=2):
    n = x.size(0)
    features = [torch.ones(n)]
    if x.shape[1] == 1:
        for i in range(1, order + 1):
            features.append(x[:, 0] ** (i))
    if x.shape[1] == 2:
        for i in range(1, order + 1):
            for j in range(i + 1):
                features.append((x[:, 0] ** (i - j)) * (x[:, 1] ** j))
    return torch.cat([f.unsqueeze(1) for f in features], dim=1)


# Tri-cube weight function (copied from LWPR_quantile_wind.py)
def tri_cube(u):
    return torch.where(u < 1, (1 - u ** 3) ** 3, torch.zeros_like(u))


# Perform weighted least squares regression (copied from LWPR_quantile_wind.py)
def weighted_least_squares(X, y, weights):
    weighted_X = X * weights.unsqueeze(1)
    XTWX = weighted_X.T @ X
    XTWy = weighted_X.T @ y
    beta = torch.linalg.pinv(XTWX) @ XTWy
    return beta.squeeze()


# Calculate weights for all points based on a fitting point and bandwidth (copied from LWPR_quantile_wind.py)
def calculate_weights(data_points, fitting_point, bandwidth):
    if fitting_point.dim() == 1:
        fitting_point = fitting_point.unsqueeze(0)
    distances = torch.sqrt(torch.sum((data_points - fitting_point) ** 2, axis=1))
    return tri_cube(distances / bandwidth)


# Calculate adaptive bandwidth based on the percentile of distances (copied from LWPR_quantile_wind.py)
def adaptive_bandwidth(data_points, fitting_point, percentile=30):
    if fitting_point.dim() == 1:
        fitting_point = fitting_point.unsqueeze(0)
    distances = torch.sqrt(torch.sum((data_points - fitting_point) ** 2, axis=1))
    return torch.quantile(distances, percentile / 100.0)


# Scale covariates to [-1, 1] and convert to tensors
def process_data(data, covariates, response):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    features = torch.tensor(scaler.fit_transform(data[covariates]), dtype=torch.float32)
    target = torch.tensor(data[response].values, dtype=torch.float32)
    return features, target, scaler


# Local weighted quadratic regression evaluated on an n x n grid of fitting points in [-1, 1]^2
def fit_model(features, target, n_fitting_points):
    grid = torch.linspace(-1, 1, n_fitting_points)
    grid_x, grid_y = torch.meshgrid(grid, grid, indexing='ij')
    fitting_points = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=1)
    X_poly_base = polynomial_features(features)
    predicted_values = torch.zeros(len(fitting_points))
    for i, fp in enumerate(tqdm(fitting_points, desc="Fitting points (mean)")):
        bandwidth = adaptive_bandwidth(features, fp, percentile=30)
        weights = calculate_weights(features, fp, bandwidth)
        beta = weighted_least_squares(X_poly_base, target, weights)
        predicted_values[i] = (polynomial_features(fp.unsqueeze(0)) @ beta).squeeze()
    return fitting_points, predicted_values


import matplotlib as mpl
import matplotlib.colors as mcolors

# Set plot aesthetics
mpl.rcParams.update({
    'font.family': 'Helvetica',  # Use Helvetica for clarity
    'axes.labelsize': 16,        # Larger axis labels
    'axes.titlesize': 16,        # Title font size
    'xtick.labelsize': 14,       # X-axis tick label size
    'ytick.labelsize': 14,       # Y-axis tick label size
    'legend.fontsize': 14,       # Legend font size
    'axes.grid': True,           # Enable grid
    'grid.alpha': 0.1,           # Subtle grid lines
    'grid.linestyle': '--',      # Dashed grid lines for a softer look
    'figure.dpi': 300,           # High DPI for high-quality figure
    'savefig.dpi': 300           # High DPI for saving the figure
})

df = pd.read_csv('data/market_impact_renewables.csv')

# Define covariates and response
covariates = ['forecasted_wind_penetration', 'Hour']
response = 'Day Ahead APX Price'

# Process data
input_variables, target_variable, scaler = process_data(data=df, covariates=covariates, response=response)

# Fit models
fitting_points, predicted_values = fit_model(features=input_variables, target=target_variable, n_fitting_points=24)

# Plot results
n_grid_points = 24

fitting_points = fitting_points.cpu()  # Transfer to CPU
predicted_values = predicted_values.cpu()  # Transfer to CPU

grid_y, grid_x = np.meshgrid(np.linspace(-1, 1, n_grid_points), np.linspace(-1, 1, n_grid_points))
grid_z = griddata(fitting_points.numpy(), predicted_values.numpy(), (grid_x, grid_y), method='cubic')

original_grid_x, original_grid_y = scaler.inverse_transform(np.column_stack([grid_x.ravel(), grid_y.ravel()])).T
original_grid_x = original_grid_x.reshape(grid_x.shape)
original_grid_y = original_grid_y.reshape(grid_y.shape)

# colors_wind = ["#f0f0f0", "#1f77b4"]  # Light gray to blue
# colors_wind = ["#add8e6", "#1f77b4"]  # Light blue to dark blue
colors_wind = ["#e0ffff", "#1f77b4"]  # Light cyan to bright blue
n_bins = 100  # Number of bins for colormap
# cmap_wind = mcolors.LinearSegmentedColormap.from_list("wind_cmap", colors_wind, N=n_bins)
cmap_wind = 'plasma'
# cmap_wind = 'Blues'

fig = plt.figure(figsize=(8, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(original_grid_x, original_grid_y, grid_z, cmap=cmap_wind, edgecolor='k', linewidth=.5)
contours = ax.contourf(original_grid_x, original_grid_y, grid_z, zdir='z', offset=np.min(grid_z), cmap=cmap_wind, alpha=0.7)
contours = ax.contourf(original_grid_x, original_grid_y, grid_z, zdir='x', offset=np.min(original_grid_x), cmap=cmap_wind, alpha=0.7)
contours = ax.contourf(original_grid_x, original_grid_y, grid_z, zdir='y', offset=np.max(original_grid_y), cmap=cmap_wind, alpha=0.7)

cbar = fig.colorbar(surface, orientation='horizontal', shrink=0.5, aspect=50, pad=0.05)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xlabel('Forecasted penetration [%]', labelpad=10)
ax.set_ylabel('Hour of the day', labelpad=10)
ax.set_zlabel('Price [GBP/MWh]', labelpad=10)
# ax.set_zlim(-80, 200)
plt.tight_layout()
plt.savefig(f'outputs/LWPR_wind.png',
            format='png', dpi=600)
plt.show()
