import numpy as np
import pandas as pd
import laspy
import rasterio
from rasterio.transform import rowcol
import matplotlib.pyplot as plt
from matplotlib import patheffects
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, norm
from sklearn.linear_model import LinearRegression
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def load_validation_data(filepath, sep=None):
    """
    Load validation data from ASCII file (CSV, TXT, DAT).
    Auto-detects separator if not specified.

    Parameters
    ----------
    filepath : str
        Path to ASCII validation file
    sep : str, optional
        Delimiter (space, comma, tab). Auto-detected if None.

    Returns
    -------
    validation_xyz : ndarray (N, 3)
        X, Y, Z columns
    filename : str
        Input filename for logging
    """
    filepath = Path(filepath)
    filename = filepath.name

    if sep is None:
        with open(filepath, 'r') as f:
            first_line = f.readline()
            if ',' in first_line:
                sep = ','
            elif '\t' in first_line:
                sep = '\t'
            else:
                sep = '\s+'

    engine = 'python' if sep == '\s+' else None
    df = pd.read_csv(filepath, sep=sep, header=None, engine=engine)
    validation_xyz = df.iloc[:, :3].to_numpy(dtype=np.float64)

    print(f"Loaded validation data: {filename} | {validation_xyz.shape[0]} points")
    return validation_xyz, filename


def load_point_cloud(source):
    """
    Load point cloud from LAS file or numpy array.

    Parameters
    ----------
    source : str or ndarray
        Path to LAS file or numpy array (N, ≥3)

    Returns
    -------
    pc_xyz : ndarray (N, 3)
        X, Y, Z columns
    source_crs : int
        EPSG code of source CRS
    """
    if isinstance(source, str):
        las = laspy.read(source)
        # column_stack avoids building a (3, N) array then transposing it
        pc_xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
        source_crs = las.header.parse_crs().to_epsg()
        print(f"Loaded point cloud from LAS: {source} | {pc_xyz.shape[0]} points | EPSG:{source_crs}")
    elif isinstance(source, np.ndarray):
        pc_xyz = source[:, :3].astype(np.float64)
        source_crs = None
        print(f"Loaded point cloud array: {pc_xyz.shape[0]} points")
    else:
        raise ValueError("source must be LAS filepath or numpy array")

    return pc_xyz, source_crs


def transform_to_epsg4326(coords, source_epsg):
    """
    Transform coordinates to EPSG:4326 (WGS84 geographic).

    Parameters
    ----------
    coords : ndarray (N, 2)
        X, Y coordinates in source CRS
    source_epsg : int
        EPSG code of source CRS

    Returns
    -------
    coords_4326 : ndarray (N, 2)
        Lon, lat in EPSG:4326
    """
    transformer = Transformer.from_crs(source_epsg, 4326, always_xy=True)
    coords_4326 = np.column_stack(transformer.transform(coords[:, 0], coords[:, 1]))
    return coords_4326


def read_datum_offsets(geopath, coords_epsg4326):
    """
    Read ellipsoid-to-MSL offsets from GeoTIFF at given coordinates.

    Vectorized: computes all row/col indices in one call, reads the band
    once, and does a single fancy-indexing lookup instead of looping in
    Python per point. This is the dominant cost in the original pipeline
    for large point clouds (it scales with N raster lookups) — vectorizing
    it removes essentially all Python-level per-point overhead.

    Parameters
    ----------
    geopath : str
        Path to GeoTIFF datum file
    coords_epsg4326 : ndarray (N, 2)
        Lon, lat coordinates (EPSG:4326)

    Returns
    -------
    offsets : ndarray (N,)
        Ellipsoid-to-MSL undulation values (meters)
    """
    n_total = len(coords_epsg4326)
    offsets = np.full(n_total, np.nan, dtype=np.float64)

    with rasterio.open(geopath) as src:
        lon = coords_epsg4326[:, 0]
        lat = coords_epsg4326[:, 1]

        # Vectorized affine inverse transform -> row/col arrays in one call
        rows, cols = rowcol(src.transform, lon, lat)
        rows = np.asarray(rows)
        cols = np.asarray(cols)

        in_bounds = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)

        band = src.read(1)  # single read of the full band, not per-point
        nodata = src.nodata

        valid_rows = rows[in_bounds]
        valid_cols = cols[in_bounds]
        sampled = band[valid_rows, valid_cols].astype(np.float64)

        if nodata is not None:
            sampled = np.where(sampled == nodata, np.nan, sampled)

        offsets[in_bounds] = sampled

    n_valid = np.sum(~np.isnan(offsets))
    print(f"Read datum offsets: {n_valid}/{n_total} points within GeoTIFF bounds")

    if n_valid < n_total * 0.8:
        warnings.warn(f"Only {n_valid}/{n_total} points have valid datum offsets")

    return offsets


def ellipsoid_to_msl(pc_xyz, datum_offsets):
    """
    Convert Z coordinates from ellipsoid to MSL using datum offsets.

    Parameters
    ----------
    pc_xyz : ndarray (N, 3)
        Point cloud with Z in ellipsoid reference
    datum_offsets : ndarray (N,)
        Ellipsoid-to-MSL undulation

    Returns
    -------
    pc_msl : ndarray (N, 3)
        Point cloud with Z in MSL reference
    """
    pc_msl = pc_xyz.copy()
    valid_mask = ~np.isnan(datum_offsets)
    pc_msl[valid_mask, 2] = pc_xyz[valid_mask, 2] - datum_offsets[valid_mask]
    pc_msl[~valid_mask, 2] = np.nan

    n_valid = np.sum(valid_mask)
    print(f"Converted to MSL: {n_valid}/{len(pc_msl)} points valid")

    return pc_msl


def match_validation_to_cloud(validation_xyz, cloud_xyz, max_distance=None):
    """
    Match validation points to nearest cloud points using KDTree.

    Parameters
    ----------
    validation_xyz : ndarray (N_val, 3)
        Validation point locations
    cloud_xyz : ndarray (N_cloud, 3)
        Point cloud locations
    max_distance : float, optional
        Maximum distance (meters) for match acceptance. None = accept all nearest neighbors.

    Returns
    -------
    match_data : dict
        'cloud_idx': indices of matched cloud points
        'distances': euclidean distances
        'valid_mask': boolean mask of accepted matches
    """
    tree = cKDTree(cloud_xyz[:, :2])
    # workers=-1 parallelizes the query across all available CPU cores
    distances, indices = tree.query(validation_xyz[:, :2], k=1, workers=-1)

    if max_distance is not None:
        valid_mask = distances <= max_distance
    else:
        valid_mask = np.ones(len(distances), dtype=bool)

    n_matched = np.sum(valid_mask)
    print(f"Spatial matching: {n_matched}/{len(distances)} validation points matched "
          f"(threshold: {max_distance if max_distance else 'None'} m)")

    return {
        'cloud_idx': indices,
        'distances': distances,
        'valid_mask': valid_mask
    }


def filter_by_depth_range(validation_xyz, cloud_xyz_matched, match_data,
                          min_depth=0, max_depth=None, msl_reference=0):
    """
    Filter matched points by depth range (positive-down convention).

    Parameters
    ----------
    validation_xyz : ndarray (N_val, 3)
        Validation data
    cloud_xyz_matched : ndarray (N_val, 3)
        Matched cloud data (indexed by match_data)
    match_data : dict
        Output from match_validation_to_cloud()
    min_depth : float, default=0
        Minimum depth (meters, positive downward)
    max_depth : float, optional
        Maximum depth (meters). None = no upper limit.
    msl_reference : float, default=0
        MSL elevation (meters) for depth calculation

    Returns
    -------
    filter_mask : ndarray (N_val,)
        Boolean mask for points within depth range
    depths : ndarray (N_val,)
        Calculated depths (positive down)
    """
    valid_mask = match_data['valid_mask']
    depths = np.full(len(validation_xyz), np.nan)

    depths[valid_mask] = msl_reference - validation_xyz[valid_mask, 2]

    filter_mask = np.zeros(len(depths), dtype=bool)

    if max_depth is None:
        filter_mask[valid_mask] = (depths[valid_mask] >= min_depth)
    else:
        filter_mask[valid_mask] = (depths[valid_mask] >= min_depth) & (depths[valid_mask] <= max_depth)

    n_in_range = np.sum(filter_mask)
    print(f"Depth range filter ({min_depth}-{max_depth if max_depth else '∞'} m): "
          f"{n_in_range}/{np.sum(valid_mask)} points retained")

    return filter_mask, depths


def calculate_metrics(validation_z, cloud_z):
    """
    Calculate error metrics between validation and cloud Z values.

    Parameters
    ----------
    validation_z : ndarray
        Validation Z values
    cloud_z : ndarray
        Matched cloud Z values

    Returns
    -------
    metrics : dict
        Keys: mae, rmse, mbe, r2, pearson_r, pearson_p, n_points
    """
    residuals = validation_z - cloud_z

    mae = np.nanmean(np.abs(residuals))
    rmse = np.sqrt(np.nanmean(residuals**2))
    mbe = np.nanmean(residuals)

    valid_idx = ~np.isnan(residuals)
    y_val = validation_z[valid_idx]
    y_pred = cloud_z[valid_idx]

    ss_res = np.sum((y_val - y_pred)**2)
    ss_tot = np.sum((y_val - np.mean(y_val))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    pearson_r, pearson_p = pearsonr(y_val, y_pred)

    metrics = {
        'n_points': np.sum(valid_idx),
        'mae': mae,
        'rmse': rmse,
        'mbe': mbe,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
    }

    print(f"\nMetrics ({metrics['n_points']} points):")
    print(f"  MAE:  {mae:.4f} m")
    print(f"  RMSE: {rmse:.4f} m")
    print(f"  MBE:  {mbe:.4f} m")
    print(f"  R²:   {r2:.4f}")
    print(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})")

    return metrics


def plot_scatter_with_regression(validation_z, cloud_z, output_path=None):
    """
    Create scatter plot with 1:1 line and regression fit.

    Parameters
    ----------
    validation_z : ndarray
        Validation Z values
    cloud_z : ndarray
        Cloud Z values
    output_path : str, optional
        Path to save PNG. If None, just returns figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    valid_idx = ~np.isnan(validation_z) & ~np.isnan(cloud_z)
    x = validation_z[valid_idx]
    y = cloud_z[valid_idx]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(x, y, alpha=0.6, s=30, edgecolors='k', linewidth=0.5, label='Data points')

    combined = np.concatenate([x, y])
    z_min, z_max = combined.min(), combined.max()
    z_range = z_max - z_min
    z_min -= 0.05 * z_range
    z_max += 0.05 * z_range
    ax.plot([z_min, z_max], [z_min, z_max], 'k--', linewidth=2, label='1:1 Reference', alpha=0.7)

    lr = LinearRegression()
    lr.fit(x.reshape(-1, 1), y)
    x_line = np.array([z_min, z_max])
    y_line = lr.predict(x_line.reshape(-1, 1))

    r2 = lr.score(x.reshape(-1, 1), y)
    slope = lr.coef_[0]
    intercept = lr.intercept_

    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'Fit: y={slope:.3f}x+{intercept:.3f}')

    ax.set_xlabel('Validation Data (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('SfM Bathymetry Data (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'SfM Bathymetry vs Validation Data (R² = {r2:.4f})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved scatter plot: {output_path}")

    return fig


def plot_residuals_vs_depth(depths, validation_z, cloud_z, output_path=None):
    """
    Plot residuals versus depth.

    Parameters
    ----------
    depths : ndarray
        Depth values (positive down)
    validation_z : ndarray
        Validation Z
    cloud_z : ndarray
        Cloud Z
    output_path : str, optional
        Path to save PNG

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    residuals = validation_z - cloud_z

    valid_idx = ~np.isnan(depths) & ~np.isnan(residuals)
    d = depths[valid_idx]
    r = residuals[valid_idx]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = np.abs(r)
    scatter = ax.scatter(d, r, c=colors, cmap='viridis', s=40, alpha=0.7, edgecolors='k', linewidth=0.5)

    ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Zero residual', alpha=0.7)

    z = np.polyfit(d, r, 1)
    p = np.poly1d(z)
    d_sort = np.sort(d)
    ax.plot(d_sort, p(d_sort), 'b-', linewidth=2.5, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')

    ax.set_xlabel('Depth (m, positive downward)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residuals: Validation - SfM (m)', fontsize=12, fontweight='bold')
    ax.set_title('Residuals vs Depth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('|Residual| (m)', fontsize=11)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved residuals plot: {output_path}")

    return fig


def plot_error_distribution(validation_z, cloud_z, output_path=None):
    """
    Plot histogram of residuals with normal distribution overlay.

    Parameters
    ----------
    validation_z : ndarray
        Validation Z
    cloud_z : ndarray
        Cloud Z
    output_path : str, optional
        Path to save PNG

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    residuals = validation_z - cloud_z
    residuals = residuals[~np.isnan(residuals)]

    mu = np.mean(residuals)
    sigma = np.std(residuals)
    skewness = np.mean(((residuals - mu) / sigma) ** 3)

    fig, ax = plt.subplots(figsize=(10, 7))

    n, bins, patches = ax.hist(residuals, bins=30, density=True, alpha=0.7,
                                color='skyblue', edgecolor='black', linewidth=1.2, label='Observed')

    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, norm.pdf(x, mu, sigma), 'b-', linewidth=2.5, label='Normal fit')

    ax.axvline(mu, color='r', linestyle='--', linewidth=2, label=f'Mean (MBE) = {mu:.4f} m')
    ax.axvline(mu - sigma, color='orange', linestyle=':', linewidth=2, label=f'±1σ = {sigma:.4f} m')
    ax.axvline(mu + sigma, color='orange', linestyle=':', linewidth=2)

    ax.set_xlabel('Residuals: Validation - SfM (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(f'Error Distribution (RMSE = {np.sqrt(np.mean(residuals**2)):.4f} m, Skewness = {skewness:.3f})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved error distribution plot: {output_path}")

    return fig


def validate_bathymetry(pc_source, validation_file, datum_geotiff, pc_epsg=None,
                       min_depth=0, max_depth=None, max_match_distance=None,
                       msl_reference=0, output_dir=None):
    """
    Complete validation workflow: load, match, filter, calculate metrics, visualize.

    Parameters
    ----------
    pc_source : str or ndarray
        LAS filepath or numpy array of point cloud (N, ≥3)
        If LAS file: EPSG auto-detected from header
        If numpy array: pc_epsg must be provided
    validation_file : str
        Path to ASCII validation data (X, Y, Z)
    datum_geotiff : str
        Path to GeoTIFF datum file (EPSG:4326, contains ellipsoid-MSL undulation)
    pc_epsg : int, optional
        EPSG code of point cloud CRS. Auto-detected from LAS header if file input.
        Required if pc_source is numpy array.
    min_depth : float, default=0
        Minimum depth filter (m, positive down)
    max_depth : float, optional
        Maximum depth filter (m). None = no limit.
    max_match_distance : float, optional
        Spatial match threshold (m). None = accept all nearest neighbors.
    msl_reference : float, default=0
        MSL elevation for depth calculation
    output_dir : str, optional
        Directory to save results. If None, uses current directory.

    Returns
    -------
    results : dict
        'metrics': dict of error metrics
        'data': DataFrame with matched/filtered points
        'figures': dict of matplotlib figures
    """
    print("\n" + "="*70)
    print("SfM BATHYMETRY VALIDATION WORKFLOW")
    print("="*70 + "\n")

    output_dir = Path(output_dir) if output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading data...")
    pc_xyz, detected_epsg = load_point_cloud(pc_source)
    val_xyz, val_filename = load_validation_data(validation_file)

    # Auto-detect EPSG from LAS, or require user to specify for arrays
    if detected_epsg is not None:
        pc_epsg = detected_epsg
    elif pc_epsg is None:
        raise ValueError(
            "pc_epsg must be specified when using numpy array input. "
            "Detected EPSG is only available for LAS file inputs."
        )

    print("\n[2/7] Transforming to EPSG:4326...")
    pc_4326 = transform_to_epsg4326(pc_xyz[:, :2], pc_epsg)
    val_4326 = transform_to_epsg4326(val_xyz[:, :2], pc_epsg)

    print("\n[3/7] Reading datum offsets...")
    pc_offsets = read_datum_offsets(datum_geotiff, pc_4326)
    val_offsets = read_datum_offsets(datum_geotiff, val_4326)

    print("\n[4/7] Converting ellipsoid → MSL...")
    pc_msl = ellipsoid_to_msl(pc_xyz, pc_offsets)
    val_msl = ellipsoid_to_msl(val_xyz, val_offsets)

    print("\n[5/7] Spatial matching (nearest neighbor)...")
    match_data = match_validation_to_cloud(val_msl, pc_msl, max_distance=max_match_distance)

    cloud_matched = pc_msl[match_data['cloud_idx']]

    print("\n[6/7] Filtering by depth range...")
    depth_filter, depths = filter_by_depth_range(val_msl, cloud_matched, match_data,
                                                  min_depth=min_depth, max_depth=max_depth,
                                                  msl_reference=msl_reference)

    final_mask = depth_filter & match_data['valid_mask']

    val_final = val_msl[final_mask, 2]
    cloud_final = cloud_matched[final_mask, 2]
    depths_final = depths[final_mask]

    if len(val_final) < 3:
        raise ValueError(f"Too few points after filtering ({len(val_final)}). "
                        "Check depth range and match distance settings.")

    print("\n[7/7] Calculating metrics and generating plots...")
    metrics = calculate_metrics(val_final, cloud_final)

    print("\nGenerating visualizations...")
    fig1 = plot_scatter_with_regression(val_final, cloud_final,
                                        output_dir / "scatter_regression.png")
    fig2 = plot_residuals_vs_depth(depths_final, val_final, cloud_final,
                                   output_dir / "residuals_vs_depth.png")
    fig3 = plot_error_distribution(val_final, cloud_final,
                                   output_dir / "error_distribution.png")
    
    print("\nSaving results...")
    results_df = pd.DataFrame({
        'validation_x': val_xyz[final_mask, 0],
        'validation_y': val_xyz[final_mask, 1],
        'validation_z_msl': val_final,
        'cloud_z_msl': cloud_final,
        'depth': depths_final,
        'residual': val_final - cloud_final,
        'match_distance': match_data['distances'][final_mask]
    })

    results_df.to_csv(output_dir / "validation_results.csv", index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    with open(output_dir / "summary_report.txt", 'w') as f:
        f.write("SfM BATHYMETRY VALIDATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        source_type = "LAS File" if isinstance(pc_source, str) else "NumPy Array"
        f.write(f"Point Cloud: {pc_source if isinstance(pc_source, str) else 'Array input'} ({source_type})\n")
        f.write(f"Validation Data: {val_filename}\n")
        f.write(f"Datum GeoTIFF: {datum_geotiff}\n")
        f.write(f"CRS: EPSG:{pc_epsg} "
                f"{'(auto-detected from LAS)' if isinstance(pc_source, str) else '(user-specified)'}\n\n")
        f.write(f"Processing Parameters:\n")
        f.write(f"  Depth Range: {min_depth} - {max_depth if max_depth else '∞'} m\n")
        f.write(f"  Match Distance Threshold: {max_match_distance if max_match_distance else 'None'} m\n")
        f.write(f"  MSL Reference: {msl_reference} m\n\n")
        f.write(f"Results:\n")
        f.write(f"  Total Validation Points: {len(val_xyz)}\n")
        f.write(f"  Total Cloud Points: {len(pc_xyz)}\n")
        f.write(f"  Matched Points: {np.sum(match_data['valid_mask'])}\n")
        f.write(f"  Final Points (after filtering): {metrics['n_points']}\n\n")
        f.write(f"Error Metrics:\n")
        f.write(f"  MAE:  {metrics['mae']:.4f} m\n")
        f.write(f"  RMSE: {metrics['rmse']:.4f} m\n")
        f.write(f"  MBE:  {metrics['mbe']:.4f} m\n")
        f.write(f"  R²:   {metrics['r2']:.4f}\n")
        f.write(f"  Pearson r: {metrics['pearson_r']:.4f} (p-value: {metrics['pearson_p']:.2e})\n")

    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - validation_results.csv")
    print(f"  - metrics.csv")
    print(f"  - summary_report.txt")
    print(f"  - scatter_regression.png")
    print(f"  - residuals_vs_depth.png")
    print(f"  - error_distribution.png")

    print("\n" + "="*70 + "\n")

    return {
        'metrics': metrics,
        'data': results_df,
        'figures': {'scatter': fig1, 'residuals': fig2, 'distribution': fig3}
    }