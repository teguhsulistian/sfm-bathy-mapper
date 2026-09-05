import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import Parallel, delayed, effective_n_jobs
from concurrent.futures import ThreadPoolExecutor

"""
svm_depth_prediction.py
========================

Predict bathymetric depth for a Structure-from-Motion (SfM) point cloud
using a Support Vector Regression (SVR) model calibrated with reference
depth measurements collected by an Unmanned Surface Vehicle (e.g.
echo-sounder survey or bathymetric LiDAR).

Input formats
-------------
- sfm_points : NumPy array, shape (N, 3) = [X, Y, Z]
                          or shape (N, 6) = [X, Y, Z, R, G, B]
               (a plain array — NOT a DataFrame). A file path (str) to a
               CSV or whitespace-delimited ASCII point-cloud file with the
               same column layout, no header, is also accepted.
- usv_points : NumPy array, shape (M, 3) = [X, Y, Z]
               or a file path (str) to a CSV / general ASCII text file,
               NO HEADER, columns in the fixed order X, Y, Z. Both comma
               and whitespace delimiters are auto-detected.

Only SfM points below the water level (Z <= wl) are used for SVR
training and depth prediction. Points above the water level (land) are
kept in the output point cloud unchanged, with Z_pred = NaN.

Workflow
--------
1.  Split the SfM point cloud into underwater (Z <= wl) and land
    (Z > wl) subsets. Only the underwater subset is ever touched by
    the SVR - land points bypass the whole optical-correction step.
2.  For every USV point, find the nearest UNDERWATER SfM point (spatial
    match) to build a training set: features = (X, Y, Z_sfm) -> target
    = Z_usv (true depth).
3.  Train an SVR (RBF kernel by default) with feature scaling, and
    optional hyperparameter search (C, epsilon, gamma).
4.  Predict corrected depth for every underwater SfM point.
5.  Recombine corrected underwater points with the untouched land
    points into one point cloud (RGB carried through, if provided).

Dependencies: numpy, pandas, scikit-learn, scipy
    pip install numpy pandas scikit-learn scipy --break-system-packages
"""


# =====================================================================
# Generic ASCII / CSV point loader (no DataFrame required as input)
# Uses pandas' C engine (much faster than np.genfromtxt on large files).
# =====================================================================
def _load_ascii_points(path):
    """
    Load a point cloud / point list from a CSV or whitespace-delimited
    ASCII text file. Auto-detects comma vs. whitespace delimiter and
    auto-detects (and skips) a single header line if present. Uses
    pandas' C parser, which is substantially faster than
    np.genfromtxt for large (multi-million row) point clouds.

    Returns
    -------
    numpy.ndarray, shape (N, ncols)
    """
    with open(path, "r") as f:
        first_line = f.readline().strip()

    sep = "," if "," in first_line else r"\s+"

    tokens = first_line.replace(",", " ").split()
    has_header = any(_not_a_number(tok) for tok in tokens)

    df = pd.read_csv(
        path,
        sep=sep,
        header=0 if has_header else None,
        engine="c",
    )
    return df.to_numpy(dtype=float)


def _not_a_number(tok):
    try:
        float(tok)
        return False
    except ValueError:
        return True


def _as_array(points, name):
    """Accept either a file path (str) or an array-like; always return a plain ndarray."""
    if isinstance(points, str):
        return _load_ascii_points(points)
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array-like of shape (N, ncols); got shape {arr.shape}")
    return arr


def _load_inputs_parallel(sfm_points, usv_points):
    """
    Load sfm_points and usv_points concurrently when both are file paths
    (I/O-bound work, so a thread pool is the right tool - no GIL/pickling
    overhead like a process pool would add). If an input is already an
    array, it passes through unchanged with no thread spawned for it.
    """
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_sfm = ex.submit(_as_array, sfm_points, "sfm_points")
        fut_usv = ex.submit(_as_array, usv_points, "usv_points")
        return fut_sfm.result(), fut_usv.result()


# =====================================================================
# Parallel chunked SVR prediction
# =====================================================================
def _predict_parallel(svm_model, X_scaled, n_jobs=-1, chunk_size=None):
    """
    Run svm_model.predict() over a large array by splitting it into
    chunks and predicting each chunk in a separate worker process
    (joblib, loky backend). For SVR with an RBF kernel, prediction cost
    scales with n_support_vectors * n_query_points, so this is the step
    that benefits most from parallelism on large SfM point clouds
    (hundreds of thousands to millions of points).

    On a single-core machine this still works correctly (joblib falls
    back to sequential execution) - it just won't be faster there.

    Parameters
    ----------
    svm_model : fitted sklearn.svm.SVR
    X_scaled : ndarray, already scaler_X-transformed features
    n_jobs : int, default -1 (use all available cores)
    chunk_size : int, optional
        Rows per chunk. Default: split evenly across n_jobs workers,
        with a floor of 2000 rows/chunk so tiny inputs don't pay
        process-spawn overhead for no benefit.

    Returns
    -------
    ndarray of predictions, same order as X_scaled
    """
    n = X_scaled.shape[0]
    if n == 0:
        return np.array([])

    workers = effective_n_jobs(n_jobs)

    if chunk_size is None:
        chunk_size = max(2000, int(np.ceil(n / workers)))

    # Small inputs: skip parallel overhead entirely.
    if n <= chunk_size or workers <= 1:
        return svm_model.predict(X_scaled)

    chunks = [X_scaled[i:i + chunk_size] for i in range(0, n, chunk_size)]
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(svm_model.predict)(chunk) for chunk in chunks
    )
    return np.concatenate(results)


# =====================================================================
# Main calibration + full-cloud prediction
# =====================================================================
def svm_depth_prediction(
    sfm_points,
    usv_points,
    wl,
    max_match_dist=None,
    kernel="rbf",
    C=None,
    epsilon=None,
    gamma=None,
    grid_search=True,
    test_size=0.2,
    random_state=42,
    n_jobs=-1,
    verbose=True,
):
    """
    Train an SVR model on USV reference depths (underwater points only)
    and predict corrected depth for the full underwater portion of an
    SfM point cloud.

    Parameters
    ----------
    sfm_points : ndarray (N,3)=[X,Y,Z] or (N,6)=[X,Y,Z,R,G,B], or str path
        SfM point cloud. Z is the raw/uncorrected SfM elevation.
    usv_points : ndarray (M,3)=[X,Y,Z], or str path to CSV/ASCII (no header,
        column order X,Y,Z). These are the ground-truth depths.
    wl : float
        Water level at the time of the SfM/UAV survey. SfM points with
        Z <= wl are treated as underwater and used for SVR training/
        prediction. Points with Z > wl are treated as land and passed
        through unchanged (Z_pred = NaN).
    max_match_dist : float, optional
        Maximum XY distance allowed when matching a USV point to its
        nearest underwater SfM point. Matches farther than this are
        discarded. Default: no limit.
    kernel : str, default 'rbf'
    C, epsilon, gamma : float, optional
        SVR hyperparameters. If None and grid_search=True, tuned
        automatically via GridSearchCV.
    grid_search : bool, default True
    test_size : float, default 0.2
        Fraction of matched training pairs held out to report accuracy.
    random_state : int, default 42
    n_jobs : int, default -1
        Number of CPU cores to use for GridSearchCV, the KD-tree
        nearest-neighbor query, and the full-cloud SVR prediction step.
        -1 uses all available cores. Has no effect on correctness -
        only wall-clock time - and safely falls back to sequential
        execution on single-core machines.
    verbose : bool, default True

    Returns
    -------
    result : dict with keys:
        'point_cloud_predicted' : pandas.DataFrame
            Full point cloud (underwater + land) with columns
            [X, Y, Z_sfm, (R, G, B if provided), Z_pred].
            Land rows have Z_pred = NaN.
        'svm_model', 'scaler_X', 'scaler_y' : fitted sklearn objects
        'svm_constants' : dict of learned SVM constants
            (kernel, C, epsilon, gamma, intercept, n_support,
             support_vectors, dual_coef)
        'metrics' : dict with 'rmse', 'mae', 'r2' on the held-out test split
        'training_pairs' : pandas.DataFrame of matched
            [X, Y, Z_sfm, Z_usv, match_dist] used for training
        'n_underwater', 'n_land' : point counts in each class
    """

    # ---- 1. Load inputs as plain arrays (in parallel if both are files) --
    sfm_arr, usv_arr = _load_inputs_parallel(sfm_points, usv_points)

    if sfm_arr.shape[1] not in (3, 6):
        raise ValueError(
            f"sfm_points must have 3 columns [X,Y,Z] or 6 columns [X,Y,Z,R,G,B]; "
            f"got {sfm_arr.shape[1]} columns."
        )
    if usv_arr.shape[1] < 3:
        raise ValueError("usv_points must have at least 3 columns [X,Y,Z].")
    usv_arr = usv_arr[:, :3]  # keep only X, Y, Z even if extra columns exist

    has_rgb = sfm_arr.shape[1] == 6
    X_all, Y_all, Z_all = sfm_arr[:, 0], sfm_arr[:, 1], sfm_arr[:, 2]
    RGB_all = sfm_arr[:, 3:6] if has_rgb else None

    underwater = Z_all <= wl
    land = ~underwater

    X_uw, Y_uw, Z_uw = X_all[underwater], Y_all[underwater], Z_all[underwater]
    RGB_uw = RGB_all[underwater] if has_rgb else None

    X_ld, Y_ld, Z_ld = X_all[land], Y_all[land], Z_all[land]
    RGB_ld = RGB_all[land] if has_rgb else None

    if underwater.sum() == 0:
        raise ValueError(f"No SfM points found below water level (wl={wl}). Check units/datum.")

    if verbose:
        print(f"[svm_depth_prediction] {underwater.sum()} underwater points (Z <= {wl}) "
              f"used for SVR; {land.sum()} land points (Z > {wl}) passed through unchanged.")

    # ---- 2. Match each USV point to its nearest UNDERWATER SfM point -----
    # workers=n_jobs parallelizes the nearest-neighbor query across cores
    # (scipy >= 1.6). Falls back to sequential automatically if n_jobs=1.
    tree = cKDTree(np.column_stack([X_uw, Y_uw]))
    dist, idx = tree.query(usv_arr[:, :2], k=1, workers=n_jobs)

    matched = pd.DataFrame({
        "X": X_uw[idx],
        "Y": Y_uw[idx],
        "Z_sfm": Z_uw[idx],
        "Z_usv": usv_arr[:, 2],
        "match_dist": dist,
    })

    if max_match_dist is not None:
        matched = matched[matched["match_dist"] <= max_match_dist].reset_index(drop=True)

    if len(matched) < 10:
        raise ValueError(
            f"Only {len(matched)} USV-SfM matches found. Need more overlapping "
            "points (check coordinate systems / max_match_dist / water level)."
        )

    if verbose:
        print(f"[svm_depth_prediction] Matched {len(matched)} USV points to underwater SfM points "
              f"(mean match distance = {matched['match_dist'].mean():.3f}).")

    # ---- 3. Build feature matrix / target, train/test split --------------
    X_feat = matched[["X", "Y", "Z_sfm"]].to_numpy()
    y_target = matched["Z_usv"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y_target, test_size=test_size, random_state=random_state
    )

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    X_train_s = scaler_X.transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).ravel()

    # ---- 4. Train SVR (with optional hyperparameter search) --------------
    if grid_search and (C is None or epsilon is None or gamma is None):
        param_grid = {
            "C": [0.1, 1, 10, 100] if C is None else [C],
            "epsilon": [0.01, 0.05, 0.1, 0.2] if epsilon is None else [epsilon],
            "gamma": ["scale", "auto", 0.01, 0.1, 1] if gamma is None else [gamma],
        }
        base_svr = SVR(kernel=kernel)
        search = GridSearchCV(
            base_svr, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=n_jobs
        )
        search.fit(X_train_s, y_train_s)
        svm_model = search.best_estimator_
        if verbose:
            print(f"[svm_depth_prediction] Best params from grid search: {search.best_params_}")
    else:
        svm_model = SVR(
            kernel=kernel,
            C=1.0 if C is None else C,
            epsilon=0.1 if epsilon is None else epsilon,
            gamma="scale" if gamma is None else gamma,
        )
        svm_model.fit(X_train_s, y_train_s)

    # ---- 5. Evaluate on held-out test split -------------------------------
    y_pred_test_s = svm_model.predict(X_test_s)
    y_pred_test = scaler_y.inverse_transform(y_pred_test_s.reshape(-1, 1)).ravel()

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "mae": float(mean_absolute_error(y_test, y_pred_test)),
        "r2": float(r2_score(y_test, y_pred_test)),
    }
    if verbose:
        print(f"[svm_depth_prediction] Test metrics -> "
              f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")

    # ---- 6. Predict corrected depth for ALL underwater SfM points --------
    # This is usually the slowest step for large point clouds (RBF-kernel
    # SVR prediction cost scales with n_support_vectors * n_query_points),
    # so it runs chunked across processes via _predict_parallel().
    X_full = np.column_stack([X_uw, Y_uw, Z_uw])
    X_full_s = scaler_X.transform(X_full)
    y_full_pred_s = _predict_parallel(svm_model, X_full_s, n_jobs=n_jobs)
    y_full_pred = scaler_y.inverse_transform(y_full_pred_s.reshape(-1, 1)).ravel()

    # ---- 7. Recombine underwater (corrected) + land (unchanged) ----------
    if has_rgb:
        uw_block = np.column_stack([X_uw, Y_uw, y_full_pred, RGB_uw])
        ld_block = np.column_stack([X_ld, Y_ld, Z_ld, RGB_ld])
        columns = ["X", "Y", "Z_corrected", "R", "G", "B"]
    else:
        uw_block = np.column_stack([X_uw, Y_uw, y_full_pred])
        ld_block = np.column_stack([X_ld, Y_ld, Z_ld])
        columns = ["X", "Y", "Z_corrected"]

    point_cloud_predicted = pd.DataFrame(np.vstack([uw_block, ld_block]), columns=columns)

    # ---- 8. Collect SVM constants -----------------------------------------
    svm_constants = {
        "kernel": svm_model.kernel,
        "C": svm_model.C,
        "epsilon": svm_model.epsilon,
        "gamma": svm_model.gamma if isinstance(svm_model.gamma, str) else float(svm_model.gamma),
        "intercept": svm_model.intercept_.tolist(),
        "n_support": int(svm_model.support_vectors_.shape[0]),
        "support_vectors": svm_model.support_vectors_,   # in scaled feature space
        "dual_coef": svm_model.dual_coef_,                # alpha_i * y_i
    }

    return {
        "point_cloud_predicted": point_cloud_predicted,
        "svm_model": svm_model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "svm_constants": svm_constants,
        "metrics": metrics,
        "training_pairs": matched,
        "n_underwater": int(underwater.sum()),
        "n_land": int(land.sum()),
    }


# =====================================================================
# Reuse a trained model on a NEW SfM point cloud (no USV points needed)
# =====================================================================
def predict_new_sfm_cloud(sfm_points, wl, svm_model, scaler_X, scaler_y, n_jobs=-1, verbose=True):
    """
    Apply an ALREADY-TRAINED SVR model to a new SfM point cloud that has
    no USV reference points of its own (pure inference, no training).
    Only points below the water level are predicted; land points are
    passed through with Z_pred = NaN.

    Parameters
    ----------
    sfm_points : ndarray (N,3)=[X,Y,Z] or (N,6)=[X,Y,Z,R,G,B], or str path
    wl : float
        Water level for this new point cloud.
    svm_model, scaler_X, scaler_y : fitted objects from a previous
        svm_depth_prediction() call (or load_model()). Do NOT re-fit
        the scalers on the new data.

    Returns
    -------
    pandas.DataFrame [X, Y, Z_sfm, (R, G, B if provided), Z_pred]
    """
    sfm_arr = _as_array(sfm_points, "sfm_points")
    if sfm_arr.shape[1] not in (3, 6):
        raise ValueError("sfm_points must have 3 or 6 columns [X,Y,Z] or [X,Y,Z,R,G,B].")

    has_rgb = sfm_arr.shape[1] == 6
    X_all, Y_all, Z_all = sfm_arr[:, 0], sfm_arr[:, 1], sfm_arr[:, 2]
    RGB_all = sfm_arr[:, 3:6] if has_rgb else None

    underwater = Z_all <= wl
    land = ~underwater

    X_uw, Y_uw, Z_uw = X_all[underwater], Y_all[underwater], Z_all[underwater]
    RGB_uw = RGB_all[underwater] if has_rgb else None
    X_ld, Y_ld, Z_ld = X_all[land], Y_all[land], Z_all[land]
    RGB_ld = RGB_all[land] if has_rgb else None

    if verbose:
        print(f"[predict_new_sfm_cloud] {underwater.sum()} underwater points (Z <= {wl}) predicted; "
              f"{land.sum()} land points passed through unchanged.")

    X_full_s = scaler_X.transform(np.column_stack([X_uw, Y_uw, Z_uw]))
    y_pred_s = _predict_parallel(svm_model, X_full_s, n_jobs=n_jobs)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

    # ----Recombine underwater (corrected) + land (unchanged) ----------
    if has_rgb:
        uw_block = np.column_stack([X_uw, Y_uw, y_full_pred, RGB_uw])
        ld_block = np.column_stack([X_ld, Y_ld, Z_ld, RGB_ld])
        columns = ["X", "Y", "Z_corrected", "R", "G", "B"]
    else:
        uw_block = np.column_stack([X_uw, Y_uw, y_full_pred])
        ld_block = np.column_stack([X_ld, Y_ld, Z_ld])
        columns = ["X", "Y", "Z_corrected"]

    point_cloud_predicted = pd.DataFrame(np.vstack([uw_block, ld_block]), columns=columns)

    return point_cloud_predicted


# =====================================================================
# Save / load trained model
# =====================================================================
def save_model(result, path_prefix="svm_depth_model"):
    """Persist the trained model + scalers to disk for later reuse."""
    import joblib
    joblib.dump(result["svm_model"], f"{path_prefix}_svr.joblib")
    joblib.dump(result["scaler_X"], f"{path_prefix}_scalerX.joblib")
    joblib.dump(result["scaler_y"], f"{path_prefix}_scalerY.joblib")


def load_model(path_prefix="svm_depth_model"):
    """Load a previously saved model + scalers."""
    import joblib
    svm_model = joblib.load(f"{path_prefix}_svr.joblib")
    scaler_X = joblib.load(f"{path_prefix}_scalerX.joblib")
    scaler_y = joblib.load(f"{path_prefix}_scalerY.joblib")
    return svm_model, scaler_X, scaler_y


# =====================================================================
# Save output point cloud back to ASCII / CSV
# =====================================================================
def save_point_cloud_csv(point_cloud_predicted, path, include_header=True):
    """Write the predicted point cloud DataFrame to a plain CSV/ASCII file."""
    point_cloud_predicted.to_csv(path, index=False, header=include_header)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Minimal synthetic example (replace with your real SfM / USV data)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    WL = 0.0  # water level

    # Fake SfM cloud: (N,6) = X,Y,Z,R,G,B, mixing land + underwater points
    xs, ys = np.meshgrid(np.linspace(0, 100, 60), np.linspace(0, 100, 60))
    true_bed = 2.0 - 0.08 * xs - 0.02 * ys + 0.5 * np.sin(xs / 10)  # crosses WL=0 partway
    sfm_z = np.where(true_bed <= WL, true_bed * 1.3 + 0.5, true_bed) + rng.normal(0, 0.2, xs.shape)
    r = np.clip(180 + sfm_z * 8 + rng.normal(0, 5, xs.shape), 0, 255)
    g = np.clip(160 + sfm_z * 5 + rng.normal(0, 5, xs.shape), 0, 255)
    b = np.clip(140 - sfm_z * 10 + rng.normal(0, 5, xs.shape), 0, 255)
    sfm_cloud = np.column_stack([xs.ravel(), ys.ravel(), sfm_z.ravel(),
                                  r.ravel(), g.ravel(), b.ravel()])

    # Fake USV reference points (underwater only), saved as headerless CSV
    underwater_mask_true = true_bed.ravel() <= WL
    uw_idx_pool = np.flatnonzero(underwater_mask_true)
    usv_idx = rng.choice(uw_idx_pool, size=min(300, len(uw_idx_pool)), replace=False)
    usv_xy = sfm_cloud[usv_idx, :2] + rng.normal(0, 0.2, (len(usv_idx), 2))
    usv_true_z = (
        2.0 - 0.08 * usv_xy[:, 0] - 0.02 * usv_xy[:, 1]
        + 0.5 * np.sin(usv_xy[:, 0] / 10)
        + rng.normal(0, 0.05, len(usv_idx))
    )
    usv_csv_path = "usv_points_example.csv"
    np.savetxt(usv_csv_path, np.column_stack([usv_xy, usv_true_z]), delimiter=",")
    print(f"Wrote example headerless USV CSV -> {usv_csv_path}")

    # sfm_points passed directly as an array (N,6); usv_points loaded from CSV file
    result = svm_depth_prediction(
        sfm_points=sfm_cloud,
        usv_points=usv_csv_path,
        wl=WL,
        max_match_dist=2.0,
        grid_search=True,
    )

    print("\nSVM constants:")
    for k, v in result["svm_constants"].items():
        if k in ("support_vectors", "dual_coef"):
            print(f"  {k}: array shape {np.asarray(v).shape}")
        else:
            print(f"  {k}: {v}")

    print(f"\nUnderwater points corrected: {result['n_underwater']}  |  "
          f"Land points passed through: {result['n_land']}")
    print("\nPredicted point cloud (head):")
    print(result["point_cloud_predicted"].head())

    # save_point_cloud_csv(result["point_cloud_predicted"], "sfm_predicted_depth.csv")

    # ------------------------------------------------------------------
    # Example: reuse the trained model on a NEW SfM cloud, no USV needed
    # ------------------------------------------------------------------
    new_xs, new_ys = np.meshgrid(np.linspace(0, 100, 20), np.linspace(0, 100, 20))
    new_true_bed = 2.0 - 0.08 * new_xs - 0.02 * new_ys
    new_sfm_z = np.where(new_true_bed <= WL, new_true_bed * 1.3 + 0.5, new_true_bed)
    new_r = np.clip(180 + new_sfm_z * 8, 0, 255)
    new_g = np.clip(160 + new_sfm_z * 5, 0, 255)
    new_b = np.clip(140 - new_sfm_z * 10, 0, 255)
    new_sfm_cloud = np.column_stack([new_xs.ravel(), new_ys.ravel(), new_sfm_z.ravel(),
                                      new_r.ravel(), new_g.ravel(), new_b.ravel()])

    new_predicted = predict_new_sfm_cloud(
        new_sfm_cloud, WL, result["svm_model"], result["scaler_X"], result["scaler_y"]
    )
    print("\nPrediction on a NEW SfM cloud (no USV training data needed here):")
    print(new_predicted.head())
