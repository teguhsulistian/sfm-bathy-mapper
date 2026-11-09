"""
bathy_ml.py
===========
Satellite/UAV-derived bathymetry (SDB) with machine learning.

Workflow
--------
1. Read training depth points (ASCII .txt / .csv, e.g. SBES or any x,y,depth file).
2. Sample UAV RGB GeoTIFF at the training points.
3. Build Stumpf-style log / band-ratio features:
       "R", "G", "B", "lnR", "lnG", "lnB", "ratio_BG", "ratio_BR", "ratio_GR"
4. Train SVR, Random Forest or XGBoost (user choice).
5. Validate against an external ASCII file (e.g. USV SBES) -> RMSE, MAE, R2, plots.
6. Predict the whole image and export a depth DEM as GeoTIFF.

Quick start (Jupyter)
---------------------
    import bathy_ml as bm

    res = bm.run_bathymetry(
        train_txt="train_sbes.txt",
        image_tif="ortho_rgb.tif",
        val_txt="val_usv_sbes.txt",
        model="xgboost",              # "svr" | "rf" | "xgboost"
        output_dem="depth_xgb.tif",
        train_cols=(0, 1, 2),         # x, y, depth (index or column names)
        val_cols=(0, 1, 2),
    )
    print(res["val_metrics"])

    # Compare all three models on the same data
    table = bm.compare_models("train_sbes.txt", "ortho_rgb.tif", "val_usv_sbes.txt")

Parallel processing
-------------------
All heavy steps accept `n_jobs` (joblib convention: -1 = all cores, 1 = serial):
point sampling (joblib), SVR tuning / CV (joblib), DEM prediction (ThreadPoolExecutor,
tile-parallel with a single writer) and compare_models(parallel_models=True).

Requirements
------------
    numpy, pandas, rasterio, scikit-learn, matplotlib, joblib, xgboost (only for "xgboost")
"""

from __future__ import annotations

import copy
import os
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs

__all__ = [
    "FEATURES",
    "read_ascii_points",
    "sample_raster_rgb",
    "build_features",
    "get_model",
    "compute_metrics",
    "plot_validation",
    "predict_dem",
    "save_model",
    "load_model",
    "apply_model",
    "save_results",
    "run_bathymetry",
    "compare_models",
]

def _ensure_dir(path: Optional[str]) -> None:
    """Create the parent folder of an output file if it does not exist yet."""
    folder = os.path.dirname(os.path.abspath(path)) if path else ""
    if folder:
        os.makedirs(folder, exist_ok=True)


# Feature order used for training AND prediction (must never change between the two)
FEATURES: List[str] = ["R", "G", "B", "lnR", "lnG", "lnB", "ratio_BG", "ratio_BR", "ratio_GR"]

# Stumpf constant: keeps the argument of ln() positive and the ratio stable
STUMPF_N: float = 1000.0

ColSpec = Sequence[Union[int, str]]


# ----------------------------------------------------------------------------
# 1. ASCII point reader
# ----------------------------------------------------------------------------
def read_ascii_points(
    path: str,
    cols: ColSpec = (0, 1, 2),
    sep: Optional[str] = None,
    z_is_depth: bool = True,
    z_offset: float = 0.0,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
) -> pd.DataFrame:
    """
    Read an ASCII x, y, z file and return a DataFrame with columns x, y, depth.

    Parameters
    ----------
    path : str
        Path to .txt / .csv / .xyz file.
    cols : (x, y, z)
        Column indices (int, 0-based) or column names (str, if the file has a header).
    sep : str or None
        Delimiter. None = auto-detect (comma, tab, space, semicolon).
    z_is_depth : bool
        True  -> the z column is already depth (positive downward).
        False -> the z column is elevation (negative below datum); depth = -z.
    z_offset : float
        Constant added to depth (e.g. tide correction to a common datum).
    min_depth, max_depth : float or None
        Keep only points inside this depth range (metres).
    """
    # Read without header first, then check whether the first row is numeric
    kw = dict(sep=sep, engine="python", header=None, comment="#", skipinitialspace=True)
    if sep is None:
        kw["sep"] = r"[,\t; ]+"
    df = pd.read_csv(path, **kw)

    first = pd.to_numeric(df.iloc[0], errors="coerce")
    if first.isna().any():  # first row contains text -> it is a header
        df = pd.read_csv(path, **{**kw, "header": 0})

    def _get(c):
        return df.iloc[:, c] if isinstance(c, (int, np.integer)) else df[c]

    out = pd.DataFrame(
        {
            "x": pd.to_numeric(_get(cols[0]), errors="coerce"),
            "y": pd.to_numeric(_get(cols[1]), errors="coerce"),
            "z": pd.to_numeric(_get(cols[2]), errors="coerce"),
        }
    ).dropna()

    out["depth"] = (out["z"] if z_is_depth else -out["z"]) + z_offset
    out = out.drop(columns="z")

    if min_depth is not None:
        out = out[out["depth"] >= min_depth]
    if max_depth is not None:
        out = out[out["depth"] <= max_depth]
    return out.reset_index(drop=True)


def _thin_points(df: pd.DataFrame, cell: float) -> pd.DataFrame:
    """Average points inside square cells (reduces spatial autocorrelation / density bias)."""
    key_x = np.floor(df["x"] / cell).astype(np.int64)
    key_y = np.floor(df["y"] / cell).astype(np.int64)
    return df.groupby([key_x, key_y], sort=False)[["x", "y", "depth"]].mean().reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Raster sampling
# ----------------------------------------------------------------------------
def _normalise(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Scale integer imagery to 0-1 so 8-bit and 16-bit images behave identically."""
    arr = arr.astype("float32")
    if dtype == "uint8":
        return arr / 255.0
    if dtype == "uint16":
        return arr / 65535.0
    return arr  # float imagery is assumed to be already scaled


def _sample_chunk(image_tif: str, x: np.ndarray, y: np.ndarray, window: int, aggregate: str = "mean") -> np.ndarray:
    """
    Sample R, G, B (0-1) for one chunk of points. Opens its own dataset handle so it is
    safe to call from separate threads or processes. Returns an (n, 3) float32 array,
    NaN where the point is outside the image or the pixel is nodata / black.
    """
    import rasterio
    from rasterio.windows import Window

    out = np.full((len(x), 3), np.nan, dtype="float32")
    with rasterio.open(image_tif) as src:
        if src.count < 3:
            raise ValueError("Image needs at least 3 bands (R, G, B).")
        dtype = src.dtypes[0]
        h, w = src.height, src.width
        rows, cols = rasterio.transform.rowcol(src.transform, x, y)
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

        if window <= 1:
            idx = np.where(inside)[0]
            if len(idx):
                vals = np.array(list(src.sample(list(zip(x[idx], y[idx])), indexes=[1, 2, 3])))
                vals = _normalise(vals, dtype)
                out[idx] = vals
                out[idx[~np.all(vals > 0, axis=1)]] = np.nan  # mask nodata / black pixels
        else:
            half = window // 2
            for i in np.where(inside)[0]:
                r0, c0 = max(rows[i] - half, 0), max(cols[i] - half, 0)
                r1, c1 = min(rows[i] + half + 1, h), min(cols[i] + half + 1, w)
                block = _normalise(src.read([1, 2, 3], window=Window(c0, r0, c1 - c0, r1 - r0)), dtype)
                valid = np.all(block > 0, axis=0)
                if valid.any():
                    vals = block[:, valid]
                    out[i] = np.median(vals, axis=1) if aggregate == "median" else vals.mean(axis=1)
    return out


def sample_raster_rgb(
    image_tif: str,
    x: np.ndarray,
    y: np.ndarray,
    window: int = 1,
    n_jobs: int = 1,
    backend: str = "loky",
    aggregate: str = "mean",
) -> pd.DataFrame:
    """
    Sample R, G, B (scaled 0-1) at point coordinates.

    Points must be in the same CRS as the GeoTIFF.
    window = 1 uses the single pixel; window = 3, 5, ... averages a window x window
    neighbourhood (reduces pixel noise).
    aggregate : "mean" or "median" of the valid pixels in the window (median ignores glint /
              foam outliers better).

    Parallel processing
    -------------------
    n_jobs  : number of workers (joblib convention: -1 = all cores, 1 = serial).
              The points are split into chunks, each worker opens its own raster handle.
    backend : "loky" (processes, best for the per-point Python loop used when window > 1)
              or "threading" (lighter, fine for window = 1).

    Returns a DataFrame (R, G, B) aligned with the input; invalid points are NaN.
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    n = len(x)
    workers = effective_n_jobs(n_jobs)
    # per-point windows are expensive, so parallelism pays off for fewer points
    min_points = 5000 if window <= 1 else 200

    if workers <= 1 or n < min_points:
        arr = _sample_chunk(image_tif, x, y, window, aggregate)
    else:
        # more chunks than workers -> better load balancing
        chunks = [c for c in np.array_split(np.arange(n), workers * 4) if len(c)]
        parts = Parallel(n_jobs=workers, backend=backend)(
            delayed(_sample_chunk)(image_tif, x[c], y[c], window, aggregate) for c in chunks
        )
        arr = np.vstack(parts)  # joblib keeps the submission order

    return pd.DataFrame(arr, columns=["R", "G", "B"])


# ----------------------------------------------------------------------------
# 3. Stumpf features
# ----------------------------------------------------------------------------
def build_features(R, G, B, n: float = STUMPF_N) -> pd.DataFrame:
    """
    Build the 9 features from R, G, B (0-1 scaled, strictly positive).

        lnX       = ln(n * X)
        ratio_BG  = ln(n*B) / ln(n*G)       (Stumpf et al., 2003)
        ratio_BR  = ln(n*B) / ln(n*R)
        ratio_GR  = ln(n*G) / ln(n*R)
    """
    R = np.asarray(R, dtype="float64")
    G = np.asarray(G, dtype="float64")
    B = np.asarray(B, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        lnR, lnG, lnB = np.log(n * R), np.log(n * G), np.log(n * B)
        feats = pd.DataFrame(
            {
                "R": R,
                "G": G,
                "B": B,
                "lnR": lnR,
                "lnG": lnG,
                "lnB": lnB,
                "ratio_BG": lnB / lnG,
                "ratio_BR": lnB / lnR,
                "ratio_GR": lnG / lnR,
            }
        )
    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats[FEATURES]


# ----------------------------------------------------------------------------
# 4. Models
# ----------------------------------------------------------------------------
_MODEL_ALIASES = {
    "svr": "svr",
    "svm": "svr",
    "rf": "rf",
    "randomforest": "rf",
    "random_forest": "rf",
    "xgb": "xgboost",
    "xgboost": "xgboost",
}

_DEFAULT_GRIDS = {
    "svr": {
        "model__regressor__C": [1, 10, 100],
        "model__regressor__gamma": ["scale", 0.01, 0.1],
        "model__regressor__epsilon": [0.05, 0.1, 0.2],
    },
    "rf": {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 3, 5],
    },
    "xgboost": {
        "n_estimators": [300, 600],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8, 1.0],
    },
}


def _canonical(name: str) -> str:
    key = name.lower().replace(" ", "").replace("-", "_")
    if key not in _MODEL_ALIASES:
        raise ValueError(f"Unknown model '{name}'. Choose 'svr', 'rf' or 'xgboost'.")
    return _MODEL_ALIASES[key]


def get_model(name: str, params: Optional[dict] = None, random_state: int = 42, n_jobs: int = -1):
    """
    Return an unfitted scikit-learn compatible regressor.

    name   : "svr" | "rf" | "xgboost"
    params : optional dict of estimator parameters overriding the defaults.
    n_jobs : threads used by Random Forest / XGBoost during fit and predict
             (SVR is single-threaded; parallelise it through the grid search / CV instead).
    """
    name = _canonical(name)
    params = dict(params or {})

    if name == "svr":
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        base = dict(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)
        base.update(params)
        # Scale inputs AND target: SVR is sensitive to both
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    TransformedTargetRegressor(regressor=SVR(**base), transformer=StandardScaler()),
                ),
            ]
        )

    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        base = dict(n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=n_jobs, random_state=random_state)
        base.update(params)
        return RandomForestRegressor(**base)

    # xgboost
    try:
        from xgboost import XGBRegressor
    except ImportError as e:  # pragma: no cover
        raise ImportError("xgboost is not installed. Run: pip install xgboost") from e

    base = dict(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=n_jobs,
        random_state=random_state,
    )
    base.update(params)
    return XGBRegressor(**base)


# ----------------------------------------------------------------------------
# 5. Statistics and plots
# ----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """RMSE, MAE, R2 (coefficient of determination), bias (mean error) and N."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
        "N": int(len(y_true)),
    }


def plot_validation(
    y_true,
    y_pred,
    title: str = "External validation",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Two-panel validation figure:
      (left)  observed vs predicted depth with 1:1 line, regression line and statistics
      (right) residual histogram (predicted - observed)
    """
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    m = compute_metrics(y_true, y_pred)
    resid = y_pred - y_true

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    hb = ax[0].hexbin(y_true, y_pred, gridsize=50, mincnt=1, cmap="viridis", bins="log")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax[0].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="1:1 line")
    if len(y_true) > 2 and np.ptp(y_true) > 0:
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        ax[0].plot([lo, hi], [slope * lo + intercept, slope * hi + intercept], "k-", lw=1, label="Regression")
    ax[0].set_xlabel("Observed depth (m)")
    ax[0].set_ylabel("Predicted depth (m)")
    ax[0].set_title(title)
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].legend(loc="lower right")
    ax[0].text(
        0.03,
        0.97,
        f"RMSE = {m['RMSE']:.3f} m\nMAE  = {m['MAE']:.3f} m\nR$^2$   = {m['R2']:.3f}\nBias = {m['Bias']:.3f} m\nN = {m['N']}",
        transform=ax[0].transAxes,
        va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
    )
    fig.colorbar(hb, ax=ax[0], label="log10(count)")

    ax[1].hist(resid, bins=40, color="steelblue", edgecolor="white")
    ax[1].axvline(0, color="r", ls="--")
    ax[1].set_xlabel("Residual: predicted - observed (m)")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Residual distribution")

    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ----------------------------------------------------------------------------
# 6. DEM export
# ----------------------------------------------------------------------------
def _single_thread_copy(model):
    """
    Deep copy of a fitted model with every internal thread pool set to 1.
    Used when the DEM is predicted block-parallel, to avoid oversubscribing the CPU
    (N block-threads x M estimator-threads).
    """
    m = copy.deepcopy(model)
    keys = {k: 1 for k in m.get_params(deep=True) if k.split("__")[-1] == "n_jobs"}
    if keys:
        m.set_params(**keys)
    inner = m.named_steps["model"] if hasattr(m, "named_steps") else m
    if hasattr(inner, "get_booster"):  # XGBoost keeps a separate thread setting in the booster
        try:
            inner.get_booster().set_param({"nthread": 1})
        except Exception:
            pass
    return m


def predict_dem(
    model,
    image_tif: str,
    output_tif: str,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    output_as_elevation: bool = False,
    z_offset: float = 0.0,
    water_mask_tif: Optional[str] = None,
    nodata: float = -9999.0,
    n_jobs: int = -1,
    tile_size: int = 512,
    verbose: bool = False,
    target_resolution: Optional[float] = None,
    min_valid_fraction: float = 0.5,
    aggregate: str = "mean",
) -> str:
    """
    Apply a trained model to the RGB GeoTIFF tile by tile and write a single-band
    float32 GeoTIFF.

    Parameters
    ----------
    min_depth, max_depth : clip predictions to this range (metres).
    output_as_elevation : if True the DEM stores -depth (negative below datum).
    z_offset : value subtracted from the predicted depth before writing.
    water_mask_tif : optional single-band raster (1 = water) on the same grid.
    n_jobs : worker threads (-1 = all cores, 1 = serial); only the main thread writes.
    tile_size : tile edge in pixels.
    target_resolution : output cell size in map units (metres). When given, R, G, B are
        AVERAGED over blocks of ~target_resolution before the features are computed, and
        the DEM is written on that coarser grid. This removes pixel-scale noise (texture,
        glint, ripples, sensor noise) and should match the footprint used for training
        (see `feature_scale_m` in run_bathymetry). None = predict at the image resolution.
    min_valid_fraction : a coarse cell needs at least this fraction of valid pixels.
    aggregate : "mean" or "median" used for the block averaging (must match training).
    """
    import rasterio
    from affine import Affine
    from rasterio.windows import Window

    workers = effective_n_jobs(n_jobs)
    mdl = _single_thread_copy(model) if workers > 1 else model

    with rasterio.open(image_tif) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        transform = src.transform
        pix = abs(src.res[0])

    # aggregation factor k: output cell = k x k input pixels
    k = max(1, int(round(target_resolution / pix))) if target_resolution else 1
    if verbose and k > 1:
        print(f"    DEM grid: {k} x {k} pixel blocks -> cell size {k * pix:.3f} m (input {pix:.3f} m)")
    tile = max(k, (tile_size // k) * k)  # tiles are multiples of k so blocks never straddle tiles
    out_h, out_w = -(-height // k), -(-width // k)

    def work(win: Window):
        """Read one tile, (optionally) block-average RGB, build features, predict."""
        with rasterio.open(image_tif) as src:
            rgb = _normalise(src.read([1, 2, 3], window=win), src.dtypes[0])
            valid = np.all(rgb > 0, axis=0) & (src.read_masks(1, window=win) > 0)
        if water_mask_tif:
            with rasterio.open(water_mask_tif) as ms:
                valid &= ms.read(1, window=win) == 1

        r0, c0 = int(win.row_off), int(win.col_off)
        if k > 1:
            h, w = valid.shape
            ph, pw = (-h) % k, (-w) % k
            if ph or pw:  # pad edge tiles to a multiple of k
                rgb = np.pad(rgb, ((0, 0), (0, ph), (0, pw)))
                valid = np.pad(valid, ((0, ph), (0, pw)))
            H, W = valid.shape
            cnt = valid.reshape(H // k, k, W // k, k).sum(axis=(1, 3))
            if aggregate == "median":
                # median of the valid pixels in every k x k block (invalid pixels -> NaN)
                nanrgb = np.where(valid[None], rgb, np.nan)
                blocks = nanrgb.reshape(3, H // k, k, W // k, k).transpose(0, 1, 3, 2, 4)
                blocks = blocks.reshape(3, -1, k * k)
                # Median only for blocks with >= 1 valid pixel: no all-NaN slices, so no
                # RuntimeWarning (warnings.catch_warnings is not thread-safe, so it cannot
                # be used reliably inside the worker threads).
                has = (cnt > 0).ravel()
                med = np.zeros((3, blocks.shape[1]), dtype="float32")
                if has.any():
                    med[:, has] = np.nanmedian(blocks[:, has], axis=-1)
                rgb = med.reshape(3, H // k, W // k)
            else:
                rgb = np.where(valid[None], rgb, 0.0)
                rgb = rgb.reshape(3, H // k, k, W // k, k).sum(axis=(2, 4)) / np.maximum(cnt, 1)
            valid = cnt >= max(1.0, min_valid_fraction * k * k)
        out_win = Window(c0 // k, r0 // k, valid.shape[1], valid.shape[0])

        block = np.full(valid.shape, nodata, dtype="float32")
        if valid.any():
            feats = build_features(rgb[0][valid], rgb[1][valid], rgb[2][valid])
            ok = feats.notna().all(axis=1).values
            pred = np.full(len(feats), np.nan, dtype="float64")
            if ok.any():
                pred[ok] = mdl.predict(feats[ok])
            if min_depth is not None or max_depth is not None:
                pred = np.clip(pred, min_depth, max_depth)
            pred = pred - z_offset
            if output_as_elevation:
                pred = -pred
            block[valid] = np.where(np.isfinite(pred), pred, nodata).astype("float32")
        return out_win, block

    profile.update(
        driver="GTiff", count=1, dtype="float32", nodata=nodata,
        width=out_w, height=out_h,
        # same as transform * Affine.scale(k, k) but without the deprecated '*' operator
        transform=Affine(transform.a * k, transform.b * k, transform.c,
                         transform.d * k, transform.e * k, transform.f),
        compress="deflate", tiled=True, blockxsize=256, blockysize=256,
    )
    profile.pop("photometric", None)

    windows = (
        Window(c0, r0, min(tile, width - c0), min(tile, height - r0))
        for r0 in range(0, height, tile)
        for c0 in range(0, width, tile)
    )
    n_tiles = ((height + tile - 1) // tile) * ((width + tile - 1) // tile)
    done_count, next_report = 0, 0.1

    def report():
        nonlocal done_count, next_report
        done_count += 1
        if verbose and done_count / n_tiles >= next_report:
            print(f"    DEM prediction: {100 * done_count / n_tiles:.0f}% ({done_count}/{n_tiles} tiles)")
            next_report += 0.1

    _ensure_dir(output_tif)
    with rasterio.open(output_tif, "w", **profile) as dst:
        if workers <= 1:
            for win in windows:
                w_, blk = work(win)
                dst.write(blk, 1, window=w_)
                report()
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                it = iter(windows)
                pending = set()

                def submit_next():
                    win = next(it, None)
                    if win is not None:
                        pending.add(ex.submit(work, win))

                for _ in range(workers * 3):
                    submit_next()
                while pending:
                    finished, _unused = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        pending.discard(fut)
                        w_, blk = fut.result()
                        dst.write(blk, 1, window=w_)
                        report()
                        submit_next()

    return output_tif


# ----------------------------------------------------------------------------
# 6b. Save / load / re-use a trained model
# ----------------------------------------------------------------------------
def _library_versions() -> Dict[str, str]:
    """Versions of the libraries that matter for unpickling a fitted model."""
    import sklearn

    v = {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    try:
        import xgboost

        v["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    return v


def save_model(path: str, model, model_name: str, metadata: Optional[dict] = None) -> str:
    """
    Save a fitted model together with everything needed to re-use it later:
    feature list, Stumpf constant, training depth range, metrics, image info and
    library versions. Stored as one joblib file (e.g. 'model_xgb.joblib').
    """
    import datetime

    import joblib

    bundle = {
        "model": model,
        "model_name": _canonical(model_name),
        "features": list(FEATURES),
        "stumpf_n": STUMPF_N,
        "versions": _library_versions(),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "meta": metadata or {},
    }
    _ensure_dir(path)
    joblib.dump(bundle, path, compress=3)
    return path


def load_model(path: str, check_versions: bool = True) -> dict:
    """
    Load a bundle written by save_model().

    Returns a dict with keys: model, model_name, features, stumpf_n, versions, created, meta.
    Warns when scikit-learn / xgboost versions differ from the ones used for training,
    because pickled models are only guaranteed to load with the same versions.
    """
    import joblib

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:  # plain estimator saved by hand
        bundle = {"model": bundle, "model_name": None, "features": list(FEATURES),
                  "stumpf_n": STUMPF_N, "versions": {}, "created": None, "meta": {}}
    if check_versions:
        now = _library_versions()
        for lib, ver in bundle.get("versions", {}).items():
            if lib in now and now[lib] != ver:
                warnings.warn(
                    f"{lib} version differs: model trained with {ver}, now {now[lib]}. "
                    "Results may change or loading may fail; ideally use the same version."
                )
    if list(bundle["features"]) != list(FEATURES):
        raise ValueError("Feature list of the saved model does not match this version of bathy_ml.")
    return bundle


# Uses: load_model, read_ascii_points, sample_raster_rgb, build_features, compute_metrics,
# plot_validation, predict_dem, save_results, _fmt, _library_versions.
# ---------------------------------------------------------------------------
def apply_model(
    model_path: str,
    image_tif: str,
    output_tif: Optional[str] = None,
    # --- external check at the new location (optional) ---
    val_txt: Optional[str] = None,
    val_cols: ColSpec = (0, 1, 2),
    sep: Optional[str] = None,
    z_is_depth: Optional[bool] = None,
    z_offset: float = 0.0,
    val_min_depth: Optional[float] = 0.0,
    val_max_depth: Optional[float] = None,
    # --- prediction settings ---
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    clip_to_training_range: bool = True,
    output_as_elevation: Optional[bool] = None,
    water_mask_tif: Optional[str] = None,
    n_jobs: int = -1,
    tile_size: int = 512,
    target_resolution: Optional[float] = None,
    aggregate: Optional[str] = None,
    # --- outputs ---
    plot: bool = True,
    plot_path: Optional[str] = None,
    stats_prefix: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Apply a saved model to a NEW RGB GeoTIFF (another site / flight) WITHOUT training data,
    and optionally check it against independent reference points (e.g. USV SBES) there.

    Parameters
    ----------
    model_path : joblib file written by run_bathymetry(save_model_path=...).
    image_tif  : new RGB GeoTIFF.
    output_tif : DEM to write. None = only run the external check.
    val_txt    : ASCII x, y, z file at the new location (same CRS as image_tif). Used ONLY
                 for checking, never for training or calibration.
    val_cols, sep : columns / delimiter of val_txt.
    z_is_depth : sign convention of val_txt z. None = same as the training run (from the model).
                 False = z is elevation (negative below datum), True = z is depth (positive).
    z_offset   : tide / datum correction of the NEW survey, added to the depth of val_txt and
                 subtracted from the DEM (same meaning as in run_bathymetry). It is NOT taken
                 from the training run, because the water level differs between surveys.
    val_min_depth, val_max_depth : keep only check points inside this depth range
                 (default 0.0 drops land / positive elevations).
    min_depth, max_depth : clip predictions (model depth, positive). None + clip_to_training_range
                 = the depth range seen in training.
    output_as_elevation : None = automatic: the DEM follows the convention of the input data
                 (z_is_depth False -> elevation, negative below datum; True -> positive depth).
                 True / False = force the sign of the DEM.
    target_resolution, aggregate : None = use feature_scale_m / aggregate of the training run.

    Returns a dict: dem_path, val_df, val_metrics, suggested_z_offset, settings, figure, stats_files.
    """
    import rasterio

    bundle = load_model(model_path)
    meta = bundle["meta"]
    model = bundle["model"]
    saved = meta.get("settings", {})
    if z_is_depth is None:
        z_is_depth = saved.get("z_is_depth", True)
    if output_as_elevation is None:
        output_as_elevation = not z_is_depth

    with rasterio.open(image_tif) as src:
        new_dtype, new_bands, pix = src.dtypes[0], src.count, abs(src.res[0])
    old = meta.get("image", {})
    if old.get("dtype") and old["dtype"] != new_dtype:
        warnings.warn(f"Image data type differs (training {old['dtype']}, new {new_dtype}); "
                      "pixel scaling is normalised, but check radiometry.")
    if new_bands < 3:
        raise ValueError("New image needs at least 3 bands (R, G, B).")

    # Feature scale and aggregation must be the same as in training
    trained_scale = saved.get("feature_scale_m")
    if target_resolution is None:
        target_resolution = trained_scale
    elif trained_scale and abs(target_resolution - trained_scale) > 1e-9:
        warnings.warn(f"target_resolution ({target_resolution}) differs from the feature scale used in "
                      f"training ({trained_scale}); predictions may be less reliable.")
    if aggregate is None:
        aggregate = saved.get("aggregate", "mean")
    if aggregate not in ("mean", "median"):
        raise ValueError("aggregate must be 'mean' or 'median'.")

    # Clip range (model depth, positive)
    rng = meta.get("depth_range")
    if clip_to_training_range and rng:
        min_depth = rng[0] if min_depth is None else min_depth
        max_depth = rng[1] if max_depth is None else max_depth

    if verbose:
        print(f"Model: {bundle['model_name']} (trained {bundle['created']}), "
              f"clip range: {min_depth} - {max_depth} m")
        print(f"       feature scale: {target_resolution} m, aggregate: {aggregate}, "
              f"z_is_depth: {z_is_depth}, z_offset: {z_offset}")
        print("NOTE: RGB-based models transfer best to the same camera, similar water clarity, "
              "bottom type and lighting. A different water level shifts the result: set z_offset.")

    result = {"dem_path": None, "val_df": None, "val_metrics": None,
              "suggested_z_offset": None, "figure": None, "stats_files": None}

    # ---------------- external check at the new location ----------------
    if val_txt:
        val = read_ascii_points(val_txt, val_cols, sep, z_is_depth, z_offset, val_min_depth, val_max_depth)
        if target_resolution:
            win = max(1, int(round(target_resolution / pix)))
            win += 1 - win % 2                       # centred window needs an odd size
        else:
            win = int(saved.get("pixel_window") or 1)
        rgb = sample_raster_rgb(image_tif, val["x"].values, val["y"].values, win, n_jobs, "loky", aggregate)
        feats = build_features(rgb["R"], rgb["G"], rgb["B"])
        val_df = pd.concat([val, feats], axis=1).dropna(subset=FEATURES + ["depth"]).reset_index(drop=True)
        if verbose:
            print(f"External check: {len(val_df)} of {len(val)} points inside the image with valid pixels")
        if len(val_df) < 3:
            warnings.warn("Fewer than 3 valid check points; statistics skipped. "
                          "Are the points in the same CRS and inside the image?")
        else:
            pred = model.predict(val_df[FEATURES])
            if min_depth is not None or max_depth is not None:
                pred = np.clip(pred, min_depth, max_depth)
            val_df["depth_pred"] = pred
            val_df["residual"] = pred - val_df["depth"]
            m = compute_metrics(val_df["depth"], pred)
            # mean(pred - depth_file) = Bias + z_offset  ->  the offset that would remove the bias
            result.update(val_df=val_df, val_metrics=m, suggested_z_offset=z_offset + m["Bias"])
            if verbose:
                print(f"    Result: {_fmt(m)}")
                print(f"    Mean bias {m['Bias']:+.3f} m. z_offset = {z_offset + m['Bias']:+.3f} would remove it, "
                      "but that calibrates on these points, so they would no longer be an independent check.")
            if plot or plot_path:
                result["figure"] = plot_validation(
                    val_df["depth"], pred,
                    title=f"{bundle['model_name'].upper()} - external check (new location)",
                    save_path=plot_path, show=plot)

    # ---------------- DEM ----------------
    if output_tif:
        result["dem_path"] = predict_dem(
            model, image_tif, output_tif,
            min_depth=min_depth, max_depth=max_depth,
            output_as_elevation=output_as_elevation, z_offset=z_offset,
            water_mask_tif=water_mask_tif, n_jobs=n_jobs, tile_size=tile_size, verbose=verbose,
            target_resolution=target_resolution, aggregate=aggregate,
        )
        if verbose:
            print(f"DEM written: {output_tif}")

    result["settings"] = {
        "model": bundle["model_name"], "model_file": model_path, "image": image_tif,
        "validation_file": val_txt, "dem": result["dem_path"],
        "feature_scale_m": target_resolution, "aggregate": aggregate,
        "z_is_depth": z_is_depth, "z_offset": z_offset, "output_as_elevation": output_as_elevation,
        "clip_range": [min_depth, max_depth], "suggested_z_offset": result["suggested_z_offset"],
        "versions": _library_versions(),
    }
    if stats_prefix:
        result["stats_files"] = save_results(
            {"model_name": bundle["model_name"], "settings": result["settings"],
             "val_metrics": result["val_metrics"], "val_df": result["val_df"]}, stats_prefix)
        if verbose:
            print("Statistics saved: " + ", ".join(result["stats_files"].values()))
    return result



# ----------------------------------------------------------------------------
# 6c. Save statistics / validation tables
# ----------------------------------------------------------------------------
def _metrics_table(res: dict) -> pd.DataFrame:
    """One row per evaluation set: Training, Spatial CV, Hold-out test, External validation."""
    rows = []
    for label, key in [
        ("Training", "train_metrics"),
        ("Spatial CV", "cv_metrics"),
        ("Hold-out test", "test_metrics"),
        ("External validation", "val_metrics"),
    ]:
        m = res.get(key)
        if m:
            rows.append({"model": res.get("model_name"), "dataset": label, **m})
    return pd.DataFrame(rows)


def save_results(res: dict, prefix: str, excel: bool = True, decimals: int = 4) -> Dict[str, str]:
    """
    Save the statistics of a run_bathymetry() result to disk.

    Files written (prefix = e.g. "out/rf_siteA"):
        <prefix>_metrics.csv            RMSE, MAE, R2, Bias, N for every evaluation set
        <prefix>_settings.json          all settings + metrics + library versions (reproducibility)
        <prefix>_validation_points.csv  x, y, observed, predicted, residual (external validation)
        <prefix>_test_points.csv        same for the hold-out test set (if test_size was used)
        <prefix>_results.xlsx           everything above as sheets (if excel=True and openpyxl is installed)

    The validation plots are saved separately through run_bathymetry(plot_path=...).
    Returns a dict {name: file path}.
    """
    import json

    folder = os.path.dirname(prefix)
    if folder:
        os.makedirs(folder, exist_ok=True)

    files: Dict[str, str] = {}
    metrics = _metrics_table(res).round(decimals)
    files["metrics"] = f"{prefix}_metrics.csv"
    metrics.to_csv(files["metrics"], index=False)

    settings = dict(res.get("settings", {}))
    settings["metrics"] = {k: res.get(k) for k in ("train_metrics", "cv_metrics", "test_metrics", "val_metrics")}
    files["settings"] = f"{prefix}_settings.json"
    with open(files["settings"], "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2, default=str)

    points = {}
    for key, label in (("val_df", "validation_points"), ("test_df", "test_points")):
        df = res.get(key)
        if df is not None and len(df):
            cols = [c for c in ("x", "y", "depth", "depth_pred", "residual") if c in df.columns]
            points[label] = df[cols].rename(columns={"depth": "depth_observed", "depth_pred": "depth_predicted"}).round(decimals)
            files[label] = f"{prefix}_{label}.csv"
            points[label].to_csv(files[label], index=False)

    if excel:
        try:
            xlsx = f"{prefix}_results.xlsx"
            with pd.ExcelWriter(xlsx) as xw:
                metrics.to_excel(xw, sheet_name="metrics", index=False)
                flat = {k: v for k, v in settings.items() if k != "metrics"}
                pd.DataFrame({"setting": list(flat), "value": [str(v) for v in flat.values()]}).to_excel(
                    xw, sheet_name="settings", index=False
                )
                for label, df in points.items():
                    df.head(1_000_000).to_excel(xw, sheet_name=label, index=False)
            files["excel"] = xlsx
        except ImportError:
            warnings.warn("Excel export skipped: install openpyxl (pip install openpyxl).")
    return files


# ----------------------------------------------------------------------------
# 7. Main function
# ----------------------------------------------------------------------------
def run_bathymetry(
    train_txt: str,
    image_tif: str,
    val_txt: Optional[str] = None,
    model: str = "xgboost",
    output_dem: Optional[str] = "bathymetry_dem.tif",
    # --- file reading ---
    train_cols: ColSpec = (0, 1, 2),
    val_cols: ColSpec = (0, 1, 2),
    sep: Optional[str] = None,
    z_is_depth: bool = True,
    z_offset: float = 0.0,
    # --- sample preparation ---
    min_depth: Optional[float] = 0.0,
    max_depth: Optional[float] = None,
    pixel_window: int = 1,
    feature_scale_m: Optional[float] = None,
    aggregate: str = "mean",
    thin_cell: Optional[float] = None,
    # --- modelling ---
    model_params: Optional[dict] = None,
    tune: bool = False,
    param_grid: Optional[dict] = None,
    cv_folds: int = 5,
    block_size: float = 50.0,
    random_state: int = 42,
    # --- train / test split (hold-out from the training file) ---
    test_size: Optional[float] = None,
    split_method: str = "spatial",
    refit_full: bool = False,
    # --- parallel processing ---
    n_jobs: int = -1,
    backend_sample: str = "loky",
    tile_size: int = 512,
    # --- outputs ---
    output_as_elevation: bool = False,
    water_mask_tif: Optional[str] = None,
    clip_prediction: bool = True,
    plot: bool = True,
    plot_path: Optional[str] = None,
    save_model_path: Optional[str] = None,
    stats_prefix: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    End-to-end pipeline: train -> external validation -> DEM GeoTIFF.

    Parameters
    ----------
    train_txt : ASCII training file (x, y, depth), same CRS as the image.
    image_tif : UAV RGB GeoTIFF (>= 3 bands, 8/16-bit or float).
    val_txt   : ASCII external validation file (e.g. USV SBES). Optional.
    model     : "svr", "rf" or "xgboost".
    output_dem: output GeoTIFF path. None = skip DEM export.
    train_cols, val_cols : (x, y, z) column indices or names.
    z_is_depth : True if z is depth (positive down); False if z is elevation (negative down).
    z_offset  : tide / datum correction added to depth of training and validation data.
    min_depth, max_depth : depth range used for training, validation and DEM clipping.
    pixel_window : 1 = single pixel; 3/5 = mean of a window around each point (pixels).
    feature_scale_m : RECOMMENDED for high-resolution UAV imagery. Spatial scale (metres)
                at which colour is averaged, e.g. 0.5. Training points use the mean RGB of a
                window of that footprint (overrides pixel_window) and the DEM is predicted on
                a grid of that cell size, so training and prediction see the same kind of
                input and pixel-scale noise is suppressed. None = pixel scale.
    aggregate : "mean" or "median" used to average the pixels inside the window / DEM cell.
                Median is more robust to sun glint, foam and shadow pixels.
    thin_cell : cell size (m) to average dense points (e.g. 1.0). None = no thinning.
    model_params : dict overriding default hyper-parameters of the chosen model.
    tune      : True = GridSearchCV with spatial GroupKFold.
    param_grid: custom grid for tuning (keys as in the estimator; for SVR use
                'model__regressor__C', etc.). None = built-in grid.
    cv_folds, block_size : spatial cross-validation settings (block_size in map units).
    test_size : fraction of the training file kept aside as a hold-out test set,
                e.g. 0.25 -> 75 % training / 25 % testing. None = use all points for
                training (no hold-out). Tuning and CV only use the training part.
    split_method : "spatial" = whole blocks (block_size) go to train or test, which avoids
                the optimistic scores caused by neighbouring points (recommended);
                "random" = random points (leaky for dense SBES lines).
    refit_full : after scoring on the hold-out test set, refit the model on ALL training
                points (train + test) before external validation and DEM export.
    output_as_elevation : DEM stores -depth instead of depth.
    water_mask_tif : optional water mask raster (1 = water) on the same grid.
    plot, plot_path : draw validation figure / save it to a PNG.
    save_model_path : joblib file for the fitted model + metadata (see load_model / apply_model).
    stats_prefix : if given (e.g. "out/rf_siteA"), statistics are saved to CSV / JSON / Excel
                with save_results(); the validation plots use plot_path.
    n_jobs    : parallel workers (-1 = all cores, 1 = serial). Used for point sampling,
                SVR grid search / CV (RF and XGBoost use the cores inside the model),
                and block-parallel DEM prediction.
    backend_sample : joblib backend for point sampling ("loky" or "threading").
    tile_size : DEM tile size in pixels.

    Returns
    -------
    dict with keys: model, model_name, features, train_df (has a 'split' column),
    train_metrics, cv_metrics, test_df, test_metrics, val_df, val_metrics, dem_path,
    best_params, figure, figure_test.
    """
    name = _canonical(model)
    if aggregate not in ("mean", "median"):
        raise ValueError("aggregate must be 'mean' or 'median'.")

    def log(*a):
        if verbose:
            print(*a)

    # ---- training data ----
    train = read_ascii_points(train_txt, train_cols, sep, z_is_depth, z_offset, min_depth, max_depth)
    log(f"[1] Training points read      : {len(train)}")
    if thin_cell:
        train = _thin_points(train, thin_cell)
        log(f"    After thinning ({thin_cell} m)  : {len(train)}")

    if feature_scale_m:
        import rasterio

        with rasterio.open(image_tif) as _s:
            _pix = abs(_s.res[0])
        pixel_window = max(1, int(round(feature_scale_m / _pix)))
        if pixel_window % 2 == 0:
            pixel_window += 1  # centred window needs an odd size
        log(f"    Feature scale {feature_scale_m} m -> {pixel_window} px window (pixel size {_pix:.3f} m)")

    rgb = sample_raster_rgb(image_tif, train["x"].values, train["y"].values, pixel_window, n_jobs, backend_sample, aggregate)
    feats = build_features(rgb["R"], rgb["G"], rgb["B"])
    train_df = pd.concat([train, feats], axis=1)
    train_df = train_df.dropna(subset=FEATURES + ["depth"]).reset_index(drop=True)
    log(f"[2] Valid points after sampling: {len(train_df)}")
    if len(train_df) < 20:
        raise ValueError(
            "Too few valid training samples. Check that the points and the image share the same CRS and overlap."
        )

    X, y = train_df[FEATURES], train_df["depth"].values

    # ---- spatial groups for CV ----
    groups = (
        np.floor(train_df["x"] / block_size).astype(int).astype(str)
        + "_"
        + np.floor(train_df["y"] / block_size).astype(int).astype(str)
    ).values

    # ---- optional hold-out split (e.g. test_size=0.25 -> 75 % train / 25 % test) ----
    train_df["split"] = "train"
    test_df, test_metrics, fig_test = None, None, None
    if test_size is not None:
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1 (e.g. 0.25).")
        from sklearn.model_selection import GroupShuffleSplit, train_test_split

        n_blocks = len(np.unique(groups))
        method = split_method.lower()
        if method not in ("spatial", "random"):
            raise ValueError("split_method must be 'spatial' or 'random'.")
        if method == "spatial" and n_blocks < 4:
            warnings.warn(
                f"Only {n_blocks} spatial block(s) at block_size={block_size}; "
                "using a random split instead. Reduce block_size for a spatial split."
            )
            method = "random"

        if method == "spatial":
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            tr_idx, te_idx = next(gss.split(X, y, groups))
        else:
            tr_idx, te_idx = train_test_split(
                np.arange(len(y)), test_size=test_size, random_state=random_state
            )
        train_df.loc[te_idx, "split"] = "test"
        test_df = train_df.iloc[te_idx].reset_index(drop=True)
        X_test, y_test = X.iloc[te_idx], y[te_idx]
        X, y, groups = X.iloc[tr_idx].reset_index(drop=True), y[tr_idx], groups[tr_idx]
        log(
            f"    Split ({method}) : train {len(y)} ({100 * len(y) / len(train_df):.0f} %) | "
            f"test {len(y_test)} ({100 * len(y_test) / len(train_df):.0f} %)"
        )

    n_groups = len(np.unique(groups))
    folds = max(2, min(cv_folds, n_groups))

    # ---- fit ----
    from sklearn.model_selection import GroupKFold, GridSearchCV, KFold, cross_val_predict

    if n_groups >= 2:
        splitter = GroupKFold(n_splits=folds)
    else:
        warnings.warn(
            f"Only {n_groups} spatial block(s) at block_size={block_size}; "
            "falling back to random KFold (optimistic). Reduce block_size."
        )
        splitter = KFold(n_splits=max(2, cv_folds), shuffle=True, random_state=random_state)
        groups = None

    estimator = get_model(name, model_params, random_state, n_jobs)
    # SVR is single-threaded -> parallelise over CV folds / grid; RF & XGB already use all cores
    cv_jobs = n_jobs if name == "svr" else 1
    best_params = None
    if tune:
        grid = param_grid or _DEFAULT_GRIDS[name]
        gs = GridSearchCV(
            estimator,
            grid,
            cv=splitter,
            scoring="neg_root_mean_squared_error",
            n_jobs=cv_jobs,
            refit=True,
        )
        gs.fit(X, y, groups=groups) if groups is not None else gs.fit(X, y)
        estimator, best_params = gs.best_estimator_, gs.best_params_
        log(f"[3] Tuned {name.upper()} best params: {best_params}")
    else:
        estimator.fit(X, y)
        log(f"[3] Fitted {name.upper()}")

    train_metrics = compute_metrics(y, estimator.predict(X))

    # Spatial cross-validation (out-of-fold predictions on the training set)
    cv_metrics = None
    cv_pred = cross_val_predict(
        get_model(name, {**(model_params or {}), **_strip_prefix(best_params)}, random_state, n_jobs),
        X,
        y,
        cv=splitter,
        groups=groups,
        n_jobs=cv_jobs,
    )
    cv_metrics = compute_metrics(y, cv_pred)
    log(f"    Training   : {_fmt(train_metrics)}")
    if cv_metrics:
        log(f"    CV         : {_fmt(cv_metrics)}")

    # ---- hold-out test evaluation ----
    if test_df is not None:
        test_pred = estimator.predict(X_test)
        if clip_prediction and (min_depth is not None or max_depth is not None):
            test_pred = np.clip(test_pred, min_depth, max_depth)
        test_df["depth_pred"] = test_pred
        test_df["residual"] = test_pred - y_test
        test_metrics = compute_metrics(y_test, test_pred)
        log(f"    Hold-out test : {_fmt(test_metrics)}")
        if plot or plot_path:
            test_plot_path = None
            if plot_path:
                root, ext = os.path.splitext(plot_path)
                test_plot_path = f"{root}_test{ext or '.png'}"
            fig_test = plot_validation(
                y_test, test_pred, title=f"{name.upper()} - hold-out test",
                save_path=test_plot_path, show=plot,
            )

        if refit_full:
            # Final model uses every training point (train + test) with the chosen settings
            X_all, y_all = train_df[FEATURES], train_df["depth"].values
            estimator = get_model(
                name, {**(model_params or {}), **_strip_prefix(best_params)}, random_state, n_jobs
            ).fit(X_all, y_all)
            log("    Refitted final model on all training points (train + test)")

    # ---- external validation ----
    val_df, val_metrics, fig = None, None, None
    if val_txt:
        val = read_ascii_points(val_txt, val_cols, sep, z_is_depth, z_offset, min_depth, max_depth)
        vrgb = sample_raster_rgb(image_tif, val["x"].values, val["y"].values, pixel_window, n_jobs, backend_sample, aggregate)
        vfeat = build_features(vrgb["R"], vrgb["G"], vrgb["B"])
        val_df = pd.concat([val, vfeat], axis=1).dropna(subset=FEATURES + ["depth"]).reset_index(drop=True)
        log(f"[4] External validation points : {len(val_df)} of {len(val)} (inside image, valid pixels)")
        if len(val_df) < 3:
            warnings.warn("Fewer than 3 valid validation points; skipping validation statistics.")
        else:
            pred = estimator.predict(val_df[FEATURES])
            if clip_prediction and (min_depth is not None or max_depth is not None):
                pred = np.clip(pred, min_depth, max_depth)
            val_df["depth_pred"] = pred
            val_df["residual"] = pred - val_df["depth"]
            val_metrics = compute_metrics(val_df["depth"], pred)
            log(f"    Validation : {_fmt(val_metrics)}")
            if plot or plot_path:
                fig = plot_validation(
                    val_df["depth"],
                    pred,
                    title=f"{name.upper()} - external validation",
                    save_path=plot_path,
                    show=plot,
                )

    # ---- DEM ----
    dem_path = None
    if output_dem:
        dem_path = predict_dem(
            estimator,
            image_tif,
            output_dem,
            min_depth=min_depth if clip_prediction else None,
            max_depth=max_depth if clip_prediction else None,
            output_as_elevation=output_as_elevation,
            z_offset=z_offset,
            water_mask_tif=water_mask_tif,
            n_jobs=n_jobs,
            tile_size=tile_size,
            verbose=verbose,
            target_resolution=feature_scale_m,
            aggregate=aggregate,
        )
        log(f"[5] DEM written                : {dem_path}")

    if save_model_path:
        import rasterio

        with rasterio.open(image_tif) as _src:
            img_info = {"dtype": _src.dtypes[0], "bands": _src.count, "crs": str(_src.crs),
                        "pixel_size": _src.res}
        meta = {
            "depth_range": [float(train_df["depth"].min()), float(train_df["depth"].max())],
            "n_training_points": int(len(train_df)),
            "fitted_on": "train+test" if (test_size is not None and refit_full) else
                         ("train only" if test_size is not None else "all training points"),
            "best_params": best_params,
            "model_params": model_params,
            "settings": {"z_is_depth": z_is_depth, "z_offset": z_offset, "pixel_window": pixel_window,
                         "thin_cell": thin_cell, "feature_scale_m": feature_scale_m, "aggregate": aggregate, "test_size": test_size, "split_method": split_method},
            "metrics": {"train": train_metrics, "cv": cv_metrics, "test": test_metrics,
                        "external_validation": val_metrics},
            "image": img_info,
            "train_file": os.path.basename(train_txt),
        }
        save_model(save_model_path, estimator, name, meta)
        log(f"    Model saved                : {save_model_path}")

    import datetime

    settings = {
        "model": name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "train_file": train_txt,
        "image": image_tif,
        "validation_file": val_txt,
        "dem": dem_path,
        "n_training_points": int(len(train_df)),
        "depth_range_training": [float(train_df["depth"].min()), float(train_df["depth"].max())],
        "feature_scale_m": feature_scale_m,
        "aggregate": aggregate,
        "pixel_window": pixel_window,
        "thin_cell": thin_cell,
        "test_size": test_size,
        "split_method": split_method if test_size is not None else None,
        "refit_full": refit_full,
        "tuned": tune,
        "best_params": best_params,
        "model_params": model_params,
        "z_is_depth": z_is_depth,
        "z_offset": z_offset,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "versions": _library_versions(),
    }

    result = {
        "model": estimator,
        "model_name": name,
        "features": FEATURES,
        "settings": settings,
        "train_df": train_df,
        "train_metrics": train_metrics,
        "cv_metrics": cv_metrics,
        "test_df": test_df,
        "test_metrics": test_metrics,
        "val_df": val_df,
        "val_metrics": val_metrics,
        "dem_path": dem_path,
        "best_params": best_params,
        "figure": fig,
        "figure_test": fig_test,
    }
    if stats_prefix:
        result["stats_files"] = save_results(result, stats_prefix)
        log("    Statistics saved           : " + ", ".join(result["stats_files"].values()))
    return result


def _compare_one(train_txt, image_tif, val_txt, m, output_prefix, kwargs) -> dict:
    """Worker for compare_models: run one model and return only picklable results."""
    kw = dict(kwargs)
    kw["output_dem"] = f"{output_prefix}_{_canonical(m)}.tif" if output_prefix else None
    if output_prefix:
        kw["plot_path"] = f"{output_prefix}_{_canonical(m)}.png"
    res = run_bathymetry(train_txt, image_tif, val_txt, model=m, **kw)
    return {"model": _canonical(m), **(res["val_metrics"] or {})}


def compare_models(
    train_txt: str,
    image_tif: str,
    val_txt: str,
    models: Iterable[str] = ("svr", "rf", "xgboost"),
    output_prefix: Optional[str] = None,
    parallel_models: bool = False,
    n_jobs: int = -1,
    save_path: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run several models with identical settings and return a comparison table
    (external validation RMSE, MAE, R2, Bias, N).

    output_prefix   : if given, DEMs are written as '<prefix>_<model>.tif' and plots as
                      '<prefix>_<model>.png'. If None, no DEMs are exported.
    parallel_models : True = run the models at the same time in separate processes and
                      split the cores between them (no interactive plots, quiet log).
                      False = run one after another, each using all cores.
    n_jobs          : total cores available.
    save_path       : save the comparison table (".csv" or ".xlsx").
    **kwargs        : any other argument of run_bathymetry().
    """
    models = list(models)
    total = effective_n_jobs(n_jobs)

    if parallel_models and len(models) > 1:
        outer = min(len(models), total)
        inner = max(1, total // outer)  # cores per model -> no oversubscription
        kw = {**kwargs, "plot": False, "verbose": False, "n_jobs": inner}
        rows = Parallel(n_jobs=outer, backend="loky")(
            delayed(_compare_one)(train_txt, image_tif, val_txt, m, output_prefix, kw) for m in models
        )
    else:
        rows = [
            _compare_one(train_txt, image_tif, val_txt, m, output_prefix, {**kwargs, "n_jobs": n_jobs})
            for m in models
        ]
    table = pd.DataFrame(rows).set_index("model")
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        if save_path.lower().endswith((".xlsx", ".xls")):
            table.round(4).to_excel(save_path)
        else:
            table.round(4).to_csv(save_path)
    return table


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _fmt(m: Dict[str, float]) -> str:
    return f"RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}  R2={m['R2']:.3f}  Bias={m['Bias']:.3f}  N={m['N']}"


def _strip_prefix(params: Optional[dict]) -> dict:
    """Convert GridSearch parameter names ('model__regressor__C') to get_model() keys ('C')."""
    if not params:
        return {}
    return {k.split("__")[-1]: v for k, v in params.items()}
