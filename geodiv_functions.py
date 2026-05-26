# functions for calculating and visualising Geodiversity metrics

# Author: Mark Lindsay https://orcid.org/0000-0003-2614-2069

from scipy.ndimage import convolve
from scipy.signal import fftconvolve
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import f
import plotly.graph_objects as go
from adjustText import adjust_text


# Moran's I is a measure of spatial autocorrelation, quantifying how similar values are to their neighbors in a spatial dataset. 
# It ranges from -1 (perfect dispersion) to +1 (perfect clustering), with 0 indicating randomness.
# In the context of 3D geodiversity models, Moran's I can help identify whether high or low density values are spatially
# clustered, dispersed, or randomly distributed within the subsurface volume. 
# This insight helps understand geophysical and geological structure.

# define function to calculate Moran's I for 3D data

# refer to https://en.wikipedia.org/wiki/Spatial_weight_matrix#/media/File:Chess_connectivity.svg for different neighborhood types

def radius_ball_kernel(radius=3, sigma=None, normalize=True):
    """
    Create a 3D isotropic, distance-weighted spherical neighborhood kernel.
    
    Parameters
    ----------
    radius : int
        Radius of the neighborhood (in voxels).
    sigma : float or None
        Standard deviation for Gaussian weighting. If None, sigma = radius / 2.
    normalize : bool
        If True, normalize weights to sum to 1.
    """
    if sigma is None:
        sigma = radius / 2.0

    size = 2 * radius + 1
    center = radius

    k = np.zeros((size, size, size), dtype=float)

    for i in range(size):
        for j in range(size):
            for l in range(size):
                dx = i - center
                dy = j - center
                dz = l - center
                d = np.sqrt(dx*dx + dy*dy + dz*dz)

                if 0 < d <= radius:
                    k[i, j, l] = np.exp(-(d*d) / (2.0 * sigma * sigma))

    if normalize and k.sum() > 0:
        k /= k.sum()

    return k


def get_neighborhood_kernels(normalize=True, radius_ball=3):
    """Returns a dictionary of kernels for different neighborhood scales."""
    # 6-node (Rook)
    k6 = np.zeros((3, 3, 3))
    k6[1,1,0] = k6[1,1,2] = k6[1,0,1] = k6[1,2,1] = k6[0,1,1] = k6[2,1,1] = 1
    if normalize:
        k6 /= k6.sum()
    
    # 18-node (Face + Edge)
    k18 = np.ones((3, 3, 3))
    # Remove the 8 corners of the 3x3x3 cube
    for i in [0, 2]:
        for j in [0, 2]:
            for k in [0, 2]:
                k18[i, j, k] = 0
    k18[1, 1, 1] = 0 # Remove center
    if normalize:
        k18 /= k18.sum()
    
    # 26-node (Queen)
    k26 = np.ones((3, 3, 3))
    k26[1, 1, 1] = 0
    if normalize:
        k26 /= k26.sum()
    
    # 2-ring (5x5x5 cube)
    k124 = np.ones((5, 5, 5))
    k124[2, 2, 2] = 0
    if normalize:
        k124 /= k124.sum()

    # Radius-3 distance-weighted spherical kernel
    k_r3 = radius_ball_kernel(radius=radius_ball, sigma=None, normalize=normalize)
    
    return {
        "moran_6": k6,
        "moran_18": k18,
        "moran_26": k26,
        "moran_2ring": k124,
        "moran_r3": k_r3
    }


# this function calculates Moran's I for a given 3D model and a specified neighborhood kernel. It uses FFT-based convolution for efficiency, especially with larger kernels.
def calculate_moran_i(model_3d, kernel):
    z = model_3d - np.mean(model_3d)
    
    # FFT convolution is O(N log N), much faster for the 5x5x5 kernel
    z_lag = fftconvolve(z, kernel, mode='same')
    
    N = model_3d.size
    # S0 is the sum of all weights in the connectivity matrix
    # On a grid, this is roughly (N * sum of kernel) minus edge effects
    # We calculate it precisely by convolving a matrix of ones
    s0_matrix = fftconvolve(np.ones_like(z), kernel, mode='same')
    S0 = np.sum(s0_matrix)
    
    numerator = N * np.sum(z * z_lag)
    denominator = S0 * np.sum(z**2)
    
    return numerator / denominator

# This function allows users to extract Moran's I values for multiple neighborhood selections in a flexible way. It can handle "all" kernels, a single kernel, or a list of kernels, and returns a dictionary of results.
# This function is called within the main metric extraction function to compute Moran's I for each model in the ensemble based on the user's choice of neighborhood analysis.
def extract_morans_i(model_3d, neighborhood_selection="all"):
    """
    neighborhood_selection: "all", a single string (e.g., "moran_26"), 
                             or a list (e.g., ["moran_6", "moran_2ring"])
    """
    all_kernels = get_neighborhood_kernels()
    results = {}
    
    # Fine which kernels to run
    if neighborhood_selection == "all":
        to_run = all_kernels.keys()
    elif isinstance(neighborhood_selection, list):
        to_run = [k for k in neighborhood_selection if k in all_kernels]
    elif neighborhood_selection in all_kernels:
        to_run = [neighborhood_selection]
    else:
        raise ValueError(f"Selection {neighborhood_selection} not recognized.")
    
    # Calculate chosen metrics
    for name in to_run:
        results[name] = calculate_moran_i(model_3d, all_kernels[name])
        
    return results

# Below is an older version of the Moran's I function using spatial convolution. Fast for small models, but for larger models
# and larger kernels, the FFT-based convolution is much faster. The above function is more flexible and can handle any kernel size.

def calculate_morans_i_3d(model_3d, kernel=6):
    """
    model_3d: 3D numpy array (x, y, z) representing density
    """
    # 1. Mean-center the density values
    z = model_3d - np.mean(model_3d)
    
    if kernel == 6:
    # 2. Define the neighborhood kernel (6-neighbor Rook contiguity)
    # This kernel sums the values of the 6 immediate face-sharing neighbors
        kernel = np.array([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ])
    elif kernel == 18:
        kernel = np.array([
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        ])
        
    elif kernel == 26:
        kernel = np.array([
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        ])
         
    # 3. Calculate Spatial Lag using convolution
    # lag_i = sum of neighbors' z-values
    z_lag = convolve(z, kernel, mode='constant', cval=0.0)
    
    # 4. Compute Moran's I components
    N = model_3d.size
    S0 = np.sum(convolve(np.ones_like(z), kernel, mode='constant', cval=0.0))
    
    numerator = N * np.sum(z * z_lag)
    denominator = S0 * np.sum(z**2)
    
    return numerator / denominator

# define function to extract metrics from 4D ensemble tensor
# neighborhood_selection defines Moran's I analysis:
# can be "all", a single string (e.g., "moran_26"), or a list (e.g., ["moran_6", "moran_2ring"])

def extract_metrics_4d(ensemble, neighborhood_selection="all"):
    """
    ensemble: np.array of shape (n_models, x_dim, y_dim, z_dim)
    Values are density.
    """
    n_models, nx, ny, nz = ensemble.shape
    
    # Create coordinate grids once (x, y, z indices for every voxel)
    # This represents the "location" of each density value
    x_idx, y_idx, z_idx = np.indices((nx, ny, nz))
    coords = np.stack([x_idx, y_idx, z_idx], axis=-1) # Shape: (nx, ny, nz, 3)

    metrics_list = []

    for i in range(n_models):
        rho = ensemble[i]
        total_rho = np.sum(rho)
        
        # 1. Statistical Metrics
        m_rho = np.mean(rho)
        v_rho = np.var(rho)
        s_rho = skew(rho.flatten())
        
        # 2. Geometric Centroid (Center of Mass in voxel units)
        # We multiply the coordinate grids by the density values
        centroid = np.array([
            np.sum(x_idx * rho) / total_rho,
            np.sum(y_idx * rho) / total_rho,
            np.sum(z_idx * rho) / total_rho
        ])
        
        # 3. Inertia Tensor (Spatial Spread)
        # Shift coords so origin is at the centroid
        rel_coords = coords - centroid
        
        # Flatten for matrix multiplication
        rel_coords_flat = rel_coords.reshape(-1, 3)
        rho_flat = rho.flatten()
        
        # Covariance-like matrix of the density distribution
        inertia_tensor = (rel_coords_flat.T * rho_flat) @ rel_coords_flat / total_rho
        
        # Eigenvalues describe the length of the 3 principal axes of the density mass
        evals = np.linalg.eigvals(inertia_tensor)
        evals = np.sort(evals)[::-1]
        
        # Moran's I for spatial autocorrelation
        morans_i = extract_morans_i(rho, neighborhood_selection=neighborhood_selection)

        # Create base metrics dictionary
        metrics_dict = {
            'model_index': i,
            'mean_rho': m_rho,
            'var_rho': v_rho,
            'skew_rho': s_rho,
            'centroid_z': centroid[2],  # Depth of the anomaly
            'axis_major': evals[0],     # Primary elongation
            'sphericity': evals[2] / evals[0] if evals[0] > 0 else 0,
            'max_rho': np.max(rho),
            'min_rho': np.min(rho),
        }
        
        # Add each Moran's I as individual columns
        for neighborhood_name, morans_value in morans_i.items():
            metrics_dict[neighborhood_name] = morans_value
        
        metrics_list.append(metrics_dict)

    return pd.DataFrame(metrics_list).set_index('model_index')


# define function to perform PCA and calculate Hotelling's T^2

def analyze_ensemble_pca(features_df):
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)
    
    # PCA
    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    
    # Hotelling's T^2 calculation
    # Normalized distance of each model from the ensemble mean
    ev = pca.explained_variance_
    t2 = np.sum((scores**2) / ev, axis=1)
    
    results = features_df.copy()
    results['PC1'] = scores[:, 0]
    results['PC2'] = scores[:, 1]
    results['T2'] = t2
    
    return results, scores, pca,

# define function to plot PCA results with scree plot and biplot

def plot_pca_results(pca, scores, feature_names):
    """
    pca: The fitted sklearn PCA object
    scores: The transformed PCA scores (PCA.transform(X))
    feature_names: list of metric names (e.g., features_df.columns)
    """
    
    # Scree Plot
    plt.figure(figsize=(8, 5))
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    
    plt.bar(range(1, len(exp_var)+1), exp_var, alpha=0.5, align='center', label='Individual variance')
    plt.step(range(1, len(cum_var)+1), cum_var, where='mid', label='Cumulative variance')
    
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.title('Scree Plot: Variance per Component')
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # PCA Biplot
    plt.figure(figsize=(10, 8))
    
    # Plot the models as scores
    sns.scatterplot(x=scores[:, 0], y=scores[:, 1], alpha=0.6, s=60, edgecolors='w')
    
    # Determine scale for loading arrows
    # Arrows are scaled so they fit comfortably within the scores plot
    scale_factor = np.max(np.abs(scores[:, 0:2])) / np.max(np.abs(pca.components_[0:2, :])) * 0.8
    
    # Plot loading vectors
    for i, name in enumerate(feature_names):
        # Coordinates for the arrow (PC1 loading, PC2 loading)
        x = pca.components_[0, i] * scale_factor
        y = pca.components_[1, i] * scale_factor
        
        plt.arrow(0, 0, x, y, color='red', alpha=0.8, head_width=0.15, lw=1.5)
        plt.text(x * 1.1, y * 1.1, name, color='darkred', ha='center', va='center', fontweight='bold')

    plt.axhline(0, color='black', linewidth=1, alpha=0.3)
    plt.axvline(0, color='black', linewidth=1, alpha=0.3)
    plt.xlabel(f'PC1 ({exp_var[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({exp_var[1]:.1%} variance)')
    plt.title('PCA Biplot: Models (Dots) vs. Metrics (Vectors)')
    plt.grid(True, linestyle=':', alpha=0.5)
    adjust_text(plt.gca().texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.show()


# define function to plot Hotelling's T^2 control chart using matplot lib



def plot_t2_control_chart(t2_values, n_samples, p_features, alpha=0.05):
    """
    t2_values: Array of T2 scores
    n_samples: Number of models (rows)
    p_features: Number of metrics (columns)
    alpha: Significance level (0.05 for 95% limit)
    """
    # Calculate the F-distribution threshold
    # Formula: ((n-1)*p / (n-p)) * F_alpha(p, n-p)
    f_threshold = f.ppf(1 - alpha, p_features, n_samples - p_features)
    threshold = (p_features * (n_samples - 1) / (n_samples - p_features)) * f_threshold

    plt.figure(figsize=(10, 5))
    plt.stem(range(len(t2_values)), t2_values, markerfmt='o', basefmt=" ")
    plt.axhline(threshold, color='red', linestyle='--', label=f'{int((1-alpha)*100)}% Confidence Limit')
    
    # Highlight outliers
    outliers = np.where(t2_values > threshold)[0]
    plt.scatter(outliers, t2_values[outliers], color='red', s=100, zorder=5)

    plt.xlabel('Model Index')
    plt.ylabel('Hotelling\'s $T^2$ Value')
    plt.title('Hotelling\'s $T^2$ Control Chart')
    plt.legend()
    plt.show()


# # define function to plot and interactive Hotelling's T^2 control chart using plotly


def plot_t2_plotly(t2_values, n_samples, p_features, alpha=0.05):
    """
    t2_values: Array of T2 scores from your PCA
    n_samples: Number of models in the ensemble
    p_features: Number of metrics used for PCA
    alpha: Significance level (0.05 for 95% limit)
    """
    # 1. Calculate the Statistical Threshold (Upper Control Limit)
    # Formula: ((n-1)*p / (n-p)) * F_alpha(p, n-p)
    f_val = f.ppf(1 - alpha, p_features, n_samples - p_features)
    ucl = (p_features * (n_samples - 1) / (n_samples - p_features)) * f_val

    # 2. Identify outliers for specialized styling
    is_outlier = t2_values > ucl
    colors = ['#EF553B' if val else '#636EFA' for val in is_outlier]

    # 3. Create the Plotly Figure
    fig = go.Figure()

    # Add the "stems" (vertical lines)
    for i in range(len(t2_values)):
        fig.add_trace(go.Scatter(
            x=[i, i],
            y=[0, t2_values[i]],
            mode='lines',
            line=dict(color='lightgrey', width=1),
            hoverinfo='none',
            showlegend=False
        ))

    # Add the points
    fig.add_trace(go.Scatter(
        x=np.arange(len(t2_values)),
        y=t2_values,
        mode='markers',
        marker=dict(
            color=colors,
            size=10,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        text=[f"Model Index: {i}<br>T² Value: {v:.2f}" for i, v in enumerate(t2_values)],
        hoverinfo='text',
        name='Models'
    ))

    # Add the Threshold Line
    fig.add_shape(
        type="line",
        x0=0, x1=len(t2_values)-1,
        y0=ucl, y1=ucl,
        line=dict(color="Red", width=2, dash="dash"),
    )

    # Add annotation for the threshold
    fig.add_annotation(
        x=len(t2_values)-1,
        y=ucl,
        text=f"{int((1-alpha)*100)}% Confidence Limit",
        showarrow=False,
        yshift=10,
        font=dict(color="Red")
    )

    # Update layout
    fig.update_layout(
        title=f"Hotelling's T² Control Chart (α={alpha})",
        xaxis_title="Model Index",
        yaxis_title="T² Statistic",
        template="plotly_white",
        hovermode="closest"
    )

    fig.show()

