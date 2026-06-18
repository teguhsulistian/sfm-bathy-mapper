import sys
import warnings
import laspy
import numpy as np
import pandas as pd
import matplotlib.path as mplPath
from shapely.geometry import Polygon
from shapely import STRtree
from shapely import points as shp_points   # vectorized C-level constructor
from multiprocessing import Pool, cpu_count
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import csr_matrix, issparse


def _ray_plane_intersect_batch(ray_origins, ray_dirs, plane_z):
    """
    Calculating the intersection of rays with a horizontal plane z = plane_z.

    Parameters
    ----------
    ray_origins : ndarray (N, 3)  — rays origin (camera position)
    ray_dirs    : ndarray (N, 3)  — rays direction (from sensor angle to camera, then extended)
    plane_z     : float           — height of the ground plane

    Returns
    -------
    xy : ndarray (N, 2) — Ground plane coordinates (X,Y) of the intersection points.
         Rows are set to NaN if the rays are parallel to the plane (no intersection).
    """
    # t = (plane_z - origin_z) / dir_z
    dz = ray_dirs[:, 2]
    #Avoid division by zero (rays parallel to plane)
    valid = np.abs(dz) > 1e-12
    t = np.where(valid, (plane_z - ray_origins[:, 2]) / np.where(valid, dz, 1.0), np.nan)

    # Intersection points
    xy = ray_origins[:, :2] + t[:, np.newaxis] * ray_dirs[:, :2]
    xy[~valid] = np.nan
    return xy

def _build_rotation_matrices(pitches, yaws, rolls):
    """
    Build the combined rotation matrix R = Rx @ Ry @ Rz for each camera
    in a vectorized manner.

    Parameters
    ----------
    pitches, yaws, rolls : ndarray (N,) in radians

    Returns
    -------
    R : ndarray (N, 3, 3)
    """
    N = len(pitches)

    cp, sp = np.cos(pitches), np.sin(pitches)
    cy, sy = np.cos(yaws),   np.sin(yaws)
    cr, sr = np.cos(rolls),  np.sin(rolls)

    # Rx (pitch)
    Rx = np.zeros((N, 3, 3))
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] =  cp;  Rx[:, 1, 2] = -sp
    Rx[:, 2, 1] =  sp;  Rx[:, 2, 2] =  cp

    # Ry (roll)
    Ry = np.zeros((N, 3, 3))
    Ry[:, 0, 0] =  cr;  Ry[:, 0, 2] = sr
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -sr;  Ry[:, 2, 2] = cr

    # Rz (yaw)
    Rz = np.zeros((N, 3, 3))
    Rz[:, 0, 0] =  cy;  Rz[:, 0, 1] = -sy
    Rz[:, 1, 0] =  sy;  Rz[:, 1, 1] =  cy
    Rz[:, 2, 2] = 1.0

    # R = Rx @ Ry @ Rz  (batch matmul)
    R = np.einsum('nij,njk->nik', np.einsum('nij,njk->nik', Rx, Ry), Rz)
    return R


def ifov_calculation(eo, sensor, mean_elev, chunk_size=1000, n_jobs=1, verbose=True):
    """
    Calculate the Instantaneous Field of View (IFOV) for each camera.

    Fully vectorized using NumPy — no Python loops per camera or per sensor angle.

    Parameters
    ----------
    eo         : pd.DataFrame (N × ≥6) — columns: x, y, z, yaw, pitch, roll (exterior orientation)
    sensor     : pd.DataFrame (1 × 3)  — columns: focal (mm), sensor_x (mm), sensor_y (mm)
    mean_elev  : float  — average ground elevation (meters)
    chunk_size : int    — batch processing size (default 1000; reduce if RAM is limited)
    n_jobs     : int    — number of parallel processes (default 1; use -1 for all CPU cores)
    verbose    : bool   — display progress

    Returns
    -------
    pd.DataFrame (N x 1) with a single column 'fov' containing matplotlib.path.Path objects.
    Cameras that exceed the critical pitch will have Path objects containing NaN.

    """
    eo = pd.read_csv(eo)
    sensor = pd.read_csv(sensor)
    
    N = eo.shape[0]
   
    # Convert sensor dimensions to meters
    f  = sensor.focal[0]    * 1e-3
    sx = sensor.sensor_x[0] * 5e-4   # /2 * 0.001
    sy = sensor.sensor_y[0] * 5e-4

    # Critical pitch
    crit_pitch = 90.0 - np.rad2deg(np.arctan(sy / f))

    if verbose:
        print(f"Processing {N:,} cameras (chunk={chunk_size}, jobs={n_jobs})...")
        sys.stdout.flush()

    # Extract NumPy arrays from DataFrame
    xs     = eo['x'].to_numpy(dtype=np.float64)
    ys     = eo['y'].to_numpy(dtype=np.float64)
    zs     = eo['z'].to_numpy(dtype=np.float64)
    yaws   = np.deg2rad(eo['yaw'].to_numpy(dtype=np.float64))
    pitches_raw = eo['pitch'].to_numpy(dtype=np.float64)
    rolls  = np.deg2rad(eo['roll'].to_numpy(dtype=np.float64))

    # Pitch for rotation: 90 - pitch_raw (in radians) for correct orientation
    pitches = np.deg2rad(90.0 - pitches_raw)

    # Mark cameras that exceed the critical pitch
    valid_mask = pitches_raw < crit_pitch   # (N,) bool

    # Template sensor corners relative to the camera center before rotation
    # Order: Upper-Right, Lower-Right, Lower-Left, Upper-Left → shape (4, 3)

    corner_offsets = np.array([
        [ sx, -f,  sy],
        [ sx, -f, -sy],
        [-sx, -f, -sy],
        [-sx, -f,  sy],
    ], dtype=np.float64)  # (4, 3)

    # Prepare output
    # all_corners_world: (N, 4, 2) , Ground plane coordinates (X,Y) of the footprint.
    all_inter = np.full((N, 4, 2), np.nan)

    # Process inside chunk
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    indices = np.where(valid_mask)[0]   # Only process valid cameras

    for start in range(0, len(indices), chunk_size):
        chunk_idx = indices[start: start + chunk_size]
        n_chunk   = len(chunk_idx)

        # Camera positions batch → (n_chunk, 3) 
        cam_pts = np.stack([xs[chunk_idx], ys[chunk_idx], zs[chunk_idx]], axis=1)

        # Combined rotation matrix R = Rx @ Ry @ Rz
        R = _build_rotation_matrices(pitches[chunk_idx], yaws[chunk_idx], rolls[chunk_idx])

        # Broadcast corner_offsets to (n_chunk, 4, 3)
        offsets = np.broadcast_to(corner_offsets, (n_chunk, 4, 3)).copy()

        # Rotation: (n_chunk, 4, 3) @ (n_chunk, 3, 3)^T → (n_chunk, 4, 3)
        # einsum: for each camera i, each corner j: out[i,j] = offsets[i,j] @ R[i]
        rotated = np.einsum('ncv,nuv->ncu', offsets, R)   # (n_chunk, 4, 3)

        # Add camera position → world coordinates of sensor corners
        corners_world = rotated + cam_pts[:, np.newaxis, :]   # (n_chunk, 4, 3)

        # Ray–Plane intersection for all (camera × corner) simultaneously
        # Ray: from corners_world through cam_pts, direction = cam_pts - corners_world
        # Flatten to (n_chunk*4, 3)
                    
        cam_pts_flat = np.repeat(cam_pts, 4, axis=0)                         # (n_chunk*4, 3)
        origins_flat = corners_world.reshape(-1, 3)                          # (n_chunk*4, 3) 
        dirs_flat = cam_pts_flat - origins_flat                              # (n_chunk*4, 3) 

        # Normalization of ray directions
        # Using plane_z = mean_elev
        xy_flat = _ray_plane_intersect_batch(origins_flat, dirs_flat, mean_elev)  # (n_chunk*4, 2)

        # Reshape → (n_chunk, 4, 2)
        xy_chunk = xy_flat.reshape(n_chunk, 4, 2)

        all_inter[chunk_idx] = xy_chunk

        if verbose and (start + n_chunk) % max(chunk_size * 5, 1000) < chunk_size:
            print(f"  {start + n_chunk:,} / {len(indices):,} Valid cameras processed...")
            sys.stdout.flush()

    # Output DataFrame
    fov_list = [mplPath.Path(all_inter[i]) for i in range(N)]
    result = pd.DataFrame({'fov': fov_list})

    if verbose:
        n_invalid = int((~valid_mask).sum())
        print(f"Finished {N - n_invalid:,} footprint valid, {n_invalid:,} NaN (critical pitch).")

    return result

def _contains_single(args):
    """
    Check whether pc_xy is inside a single Path object.
    Required as a top-level function so it can be pickled by multiprocessing.

    Parameters
    ----------
    args : tuple (path, pc_xy)
        path  : matplotlib.path.Path
        pc_xy : ndarray (N_pt, 2)

    Returns
    -------
    ndarray (N_pt,) bool
    """
    path, pc_xy = args
    return path.contains_points(pc_xy)


def _path_to_polygon(mpl_path):
    """Convert matplotlib Path → Shapely Polygon. NaN path → None."""
    verts = mpl_path.vertices
    if np.any(np.isnan(verts)):
        return None
    return Polygon(verts)
 
 
# ─────────────────────────────────────────────────────────────────
# MAIN FUNCTION: Visible points and inclination angle r (sparse output)
# ─────────────────────────────────────────────────────────────────
 
def visible_points(eo, ifov, pc, n_jobs=1, chunk_size=50, verbose=False):
    """
    Determine the visibility of point cloud points relative to each camera
    and compute the inclination angle r. Output in sparse matrix format
    for maximum memory efficiency.
 
    Parameters
    ----------
    eo         : pd.DataFrame (N_cam × ≥3) — columns: x, y, z
    ifov       : pd.DataFrame (N_cam × 1)  — columns: fov (matplotlib Path)
    pc         : ndarray (N_pt × ≥3)       — columns: x, y, z
    n_jobs     : int  — number of threads (1=serial, -1=all CPUs)
    chunk_size : int  — number of cameras per chunk (reduce if RAM is still limited)
    verbose    : bool — show progress
 
    Returns
    -------
    r_sparse : scipy.sparse.csr_matrix, shape (N_pt, N_cam), dtype float32
        Inclination angle in degrees for visible pairs.
        A value of 0 in the sparse matrix means not visible (not r=0°).
        Use r_sparse.nnz for the number of visible pairs.
 
    How to use the output:
        # Retrieve all values as dense (only if RAM is sufficient)
        r_dense = r_sparse.toarray()
        r_dense[r_dense == 0] = np.nan
 
        # Iterate per camera without dense conversion
        for ci in range(r_sparse.shape[1]):
            col = r_sparse.getcol(ci)
            pt_idxs = col.nonzero()[0]      # indices of visible points
            r_vals  = col.data               # r values for those points
 
        # Convert to DataFrame of visible pairs
        cx, cy = r_sparse.nonzero()
        r_vals = np.array(r_sparse[cx, cy]).flatten()
        df = pd.DataFrame({'pt_idx': cx, 'cam_idx': cy, 'r': r_vals})
    """
    n_cam = eo.shape[0]
    n_pt  = pc.shape[0]
 
    if verbose:
        mem_dense_gb = n_pt * n_cam * 8 / 1e9
        print(f"N_pt={n_pt:,}  N_cam={n_cam:,}  "
              f"(dense seria {mem_dense_gb:.1f} GB — menggunakan sparse)")
 
    # ── 1. Convert Path → Polygon ─────────────────────────────────
    polys = [_path_to_polygon(p) for p in ifov['fov']]
 
    # ── 2. Build STRtree from all points pc (once) ─────────────
    pc_xy   = pc[:, :2].astype(np.float64)
    shp_pts = shp_points(pc_xy)          # vectorized, without Python loop
    pt_tree = STRtree(shp_pts)
 
    # ── 3. Extract camera and pc coordinates as 1D arrays ────────
    eo_x = eo['x'].to_numpy(np.float64)  # (N_cam,)
    eo_y = eo['y'].to_numpy(np.float64)
    eo_z = eo['z'].to_numpy(np.float64)
    pc_x = pc[:, 0].astype(np.float64)  # (N_pt,)
    pc_y = pc[:, 1].astype(np.float64)
    pc_z = pc[:, 2].astype(np.float64)
 
    # ── 4. Accumulate COO data for sparse matrix ──────────────────
    # COO (Coordinate format): store (row, col, value) only for non-zero entries
    rows_list = []
    cols_list = []
    vals_list = []
 
    n_workers = cpu_count() if n_jobs == -1 else max(1, n_jobs)
 
    def _process_cam(ci):
        """Process a single camera: query visible points, calculate r, return COO."""
        poly = polys[ci]
        if poly is None:
            return None
        pt_idxs = pt_tree.query(poly, predicate='contains')  # indices of points inside the polygon
        if len(pt_idxs) == 0:
            return None
 
        # Compute r only for visible points — small array (k,) not (N_pt,)
        dx = eo_x[ci] - pc_x[pt_idxs]
        dy = eo_y[ci] - pc_y[pt_idxs]
        dz = eo_z[ci] - pc_z[pt_idxs]
        d  = np.hypot(dx, dy)
        r  = np.rad2deg(np.arctan2(d, dz)).astype(np.float32)
 
        return pt_idxs, r
 
    # ── 5. Run per chunk Camera ───────────────────────────────
    cam_indices = range(n_cam)
 
    if n_workers == 1:
        results = [_process_cam(ci) for ci in cam_indices]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_process_cam, cam_indices))
 
    # ── 6. Accumulate COO data ──────────────────────────────────────
    for ci, res in enumerate(results):
        if res is None:
            continue
        pt_idxs, r_vals = res
        rows_list.append(pt_idxs.astype(np.int32))
        cols_list.append(np.full(len(pt_idxs), ci, dtype=np.int32))
        vals_list.append(r_vals)
 
    if not rows_list:
        # No visible pairs at all
        return csr_matrix((n_pt, n_cam), dtype=np.float32)
 
    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_vals = np.concatenate(vals_list)
 
    # ── 7. Build sparse matrix ────────────────────────────────────
    r_sparse = csr_matrix(
        (all_vals, (all_rows, all_cols)),
        shape=(n_pt, n_cam),
        dtype=np.float32
    )
 
    if verbose:
        nnz = r_sparse.nnz
        mem_sparse_mb = (all_vals.nbytes + all_rows.nbytes + all_cols.nbytes) / 1e6
        print(f"Visible: {nnz:,} / {n_pt * n_cam:,} pasangan "
              f"({100 * nnz / (n_pt * n_cam):.4f}%)")
        print(f"Memory sparse: {mem_sparse_mb:.1f} MB  "
              f"(vs {n_pt * n_cam * 8 / 1e9:.1f} GB dense)")
 
    return r_sparse, polys
 
 
# ─────────────────────────────────────────────────────────────────
# UTILITIES: sparse conversion → dense per camera (stream, RAM efficient)
# ─────────────────────────────────────────────────────────────────
 
def iter_camera_results(r_sparse):
    """
    Generator: iterate through results per camera without creating a full dense array.
 
    Yields
    ------
    ci       : int   — camera index
    pt_idxs  : ndarray (k,) int   — indices of visible points
    r_vals   : ndarray (k,) float — angle r for those points
    """
    r_csr = r_sparse.tocsr()
    r_csc = r_csr.tocsc()   # column-slicing efisien
    for ci in range(r_csc.shape[1]):
        start = r_csc.indptr[ci]
        end   = r_csc.indptr[ci + 1]
        if end > start:
            idxs   = r_csc.indices[start:end]
            r_vals = r_csc.data[start:end].astype(np.float32)
            yield ci, idxs, r_vals
 
 
def to_dataframe(r_sparse):
    """
    Convert sparse matrix to DataFrame of visible pairs.
    Only use if the number of visible pairs is not too large.
 
    Returns
    -------
    pd.DataFrame with columns: pt_idx, cam_idx, r
    """
    cx, cy = r_sparse.nonzero()
    r_vals = np.asarray(r_sparse[cx, cy]).flatten()
    return pd.DataFrame({
        'pt_idx':  cx.astype(np.int32),
        'cam_idx': cy.astype(np.int32),
        'r':       r_vals.astype(np.float32),
    })


# ─────────────────────────────────────────────────────────────────
# CORE: Refraction Correction per element (fully vectorized)
# ─────────────────────────────────────────────────────────────────
 
def _refract_depth_per_element(r_deg, z_apparent, wl, n_water):
    """
    Calculate the refraction-corrected depth for each (point, camera) pair.
    All operations are vectorized — no Python loops.

    Parameters
    ----------
    r_deg      : ndarray (k,) — inclination angle in degrees
    z_apparent : ndarray (k,) — SfM-derived z value (apparent depth) of the corresponding point
    wl         : float        — water level
    n_water    : float        — refractive index of water

    Returns
    -------
    depth_corr : ndarray (k,) float64
    Refraction-corrected depth (z value).
    Returns NaN if the point is above the water level (wl) or if tan(i) ≈ 0.
    """
    rad_r = np.deg2rad(r_deg)
 
    # Snell's law: sin(i) = sin(r) / n_water
    sin_i = np.clip((1.0 / n_water) * np.sin(rad_r), -1.0, 1.0)
    tan_i = np.tan(np.arcsin(sin_i))
 
    # Process points below water level
    below     = z_apparent < wl
    depth_app = np.where(below, wl - z_apparent, np.nan)   # apparent depth (positif)
    xd        = depth_app * np.tan(rad_r)                  # jarak horizontal
 
    # True depth
    safe_tan  = np.where(np.abs(tan_i) > 1e-10, tan_i, np.nan)
    return xd / safe_tan   # z_corrected = wl - depth_true
 
 
def _mean_depth_bincount(row_idx, depth_corr, n_pt):
    """
    Calculate the mean depth_corr per point using np.bincount.
    Much faster than np.add.at because bincount is an O(n) C-level operation.
 
    Returns ndarray (n_pt,) — NaN for points without valid values.
    """
    valid = ~np.isnan(depth_corr)
    if not valid.any():
        return np.full(n_pt, np.nan)
    s = np.bincount(row_idx[valid], weights=depth_corr[valid], minlength=n_pt)
    c = np.bincount(row_idx[valid], minlength=n_pt).astype(np.float64)
    c[c == 0] = np.nan
    return s / c
 
 
# ─────────────────────────────────────────────────────────────────
# Refraction correction for all points observed by multiple cameras:
# ─────────────────────────────────────────────────────────────────
 
def process_refraction(r, pc, wl, n_water="default", n_jobs=1, verbose=True):
    """
    Correct point cloud depths for the effects of light refraction in water.

    For each point observed by multiple cameras:
    1. Compute the refraction-corrected depth using the inclination angle (r) from EACH camera.
    2. Average the corrected depths obtained from all observing cameras.

    Parameters
    ----------
    r        : scipy.sparse.csr_matrix (N_pt, N_cam) or ndarray (N_pt, N_cam)
           Inclination angles in degrees returned by visible_points().
           Sparse: 0 indicates not visible.
           Dense: NaN indicates not visible.
    pc       : ndarray (N_pt, ≥3) — columns: x, y, z, ...
    wl       : float
           Water level at the time of data acquisition (meters).
    n_water  : float or "default"
           Refractive index of water (default: 1.33).
    n_jobs   : int
           1  = serial execution (default; best for n_vis < 500k)
           -1 = use all CPU cores
           >1 = explicit number of worker threads
    verbose  : bool
           Display summary information.

    Returns
    -------
    pc_corrected : ndarray (N_pt, ≥3)
    Point cloud with refraction-corrected z values.
    Points above the water level are left unchanged.
    Points not visible from any camera retain their original z values.

    depth_mean : ndarray (N_pt,)
    Mean refraction-corrected z value for each point,
    averaged across all cameras that observe the point.
    NaN if the point is not visible from any camera.
    """
 
    # ── 1. Setup ───────────────────────────────────────────────────
    n_water  = 1.33 if n_water == "default" else float(n_water)
    n_pt     = pc.shape[0]
    n_workers = cpu_count() if n_jobs == -1 else max(1, n_jobs)
 
    if not issparse(r):
        # Dense conversion → sparse (NaN → 0)
        from scipy.sparse import csr_matrix as csr
        r_arr = np.asarray(r, dtype=np.float32)
        r_arr[np.isnan(r_arr)] = 0
        r = csr(r_arr)
 
    r_csr = r.tocsr()
    n_vis = r_csr.nnz
 
    if verbose:
        n_cam = r_csr.shape[1]
        mem_dense_gb = n_pt * n_cam * 8 / 1e9
        print(f"N_pt={n_pt:,}  N_cam={n_cam}  n_visible={n_vis:,}  "
              f"(dense seria {mem_dense_gb:.1f} GB)")
 
    # ── 2. Compute corrected depth for each sparse element ───────────────
    # Each element represents a single (point, camera) pair.
    # row_idx[k] = point index corresponding to the k-th element.
    row_idx = np.repeat(np.arange(n_pt), np.diff(r_csr.indptr))  # (n_vis,)
    r_data  = r_csr.data.astype(np.float64)                       # (n_vis,) degree
    z_elem  = pc[row_idx, 2]                                       # (n_vis,) z each point
 
    if n_workers == 1 or n_vis < 50_000:
        # ── Serial: Each operation vectorized for all elements ────
        depth_per_elem = _refract_depth_per_element(r_data, z_elem, wl, n_water)
        depth_mean = _mean_depth_bincount(row_idx, depth_per_elem, n_pt)
 
    else:
        # ── Parallel: divide points (rows) into n_workers chunks ────────
        # Each worker processes a subset of rows, so there is no overlap.
        # Therefore, local bincount operations are safe and do not require locking.
        pt_chunks = np.array_split(np.arange(n_pt), n_workers)
 
        def _worker(pt_idx):
            """Process subset of rows r_csr[pt_idx, :]."""
            sub      = r_csr[pt_idx, :]               # CSR row slicing
            r_sub    = sub.data.astype(np.float64)
            loc_rows = np.repeat(
                np.arange(len(pt_idx)), np.diff(sub.indptr)
            )                                          # local indices (0..len-1)
            z_sub    = pc[pt_idx[loc_rows], 2]
            dc       = _refract_depth_per_element(r_sub, z_sub, wl, n_water)
            return _mean_depth_bincount(loc_rows, dc, len(pt_idx))
 
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            parts = list(executor.map(_worker, pt_chunks))
        depth_mean = np.concatenate(parts)
 
    # ── 3. Apply correction to point cloud ─────────────────────────
    pc_corrected = pc.copy()
 
    below_mask = pc[:, 2] < wl
    valid_corr = below_mask & ~np.isnan(depth_mean)
 
    # z_corrected = wl - depth_mean
    pc_corrected[valid_corr, 2] = wl - depth_mean[valid_corr]
 
    # ── 4. Summary ───────────────────────────────────────────────
    if verbose:
        n_below   = int(below_mask.sum())
        n_above   = int((~below_mask).sum())
        n_corrected = int(valid_corr.sum())
        n_no_cam  = int((below_mask & np.isnan(depth_mean)).sum())
 
        print(f"Total points below WL : {n_below:,} ({100*n_below/n_pt:.2f}%)")
        print(f"Points above WL       : {n_above:,} ({100*n_above/n_pt:.2f}%)")
        print(f"Corrected points      : {n_corrected:,}")
        if n_no_cam > 0:
            print(f"Points without camera : {n_no_cam:,} — z not corrected")
        print(f"Original point cloud  : {n_pt:,} points")
        print(f"Corrected point cloud : {pc_corrected.shape[0]:,} points")
 
    return pc_corrected, depth_mean



def process_small_angle(pc, WL, n_water):
    """
    Process the point cloud to correct for refraction based on the water level and refractive index.

    Parameters:
    pc (numpy.ndarray): The input point cloud as a Nx6 array (x, y, z, red, green, blue).
    WL (float): The water level (tide height) at the time of data capture.
    n_water (float): The refractive index of water.

    Returns:
    numpy.ndarray: The corrected point cloud.
    """
    # Defining the refractive index of water, default is 1.33 for visible light in water
    if n_water == "default":
        n_water = 1.33
    else:
        n_water = float(n_water)

    # Processing the point cloud to correct for refraction (small angel approach)
    pc_filtered = pc[pc[:,2] < WL]
    pc_filtered[:,2] = ((pc_filtered[:,2]-WL) * n_water) + WL

    pc_land = pc[pc[:,2] >= WL]

    pc_corrected = np.vstack((pc_filtered, pc_land))

    print(f"Number of points below water level: {len(pc_filtered)}, percentage: {len(pc_filtered)/len(pc)*100:.2f}%")
    print(f"Number of points above water level: {len(pc_land)}, percentage: {len(pc_land)/len(pc)*100:.2f}%")   
    print(f"Original point cloud size: {len(pc)}, Corrected point cloud size: {len(pc_corrected)}")
    
    return pc_corrected


def export_pc(pc_corrected, las, output_path):
    """
    Save the corrected point cloud to a new LAS file.

    Parameters:
    pc_corrected (numpy.ndarray): The corrected point cloud as a Nx6 array (x, y, z, red, green, blue).
    las (laspy.LasData): The original LAS data object to copy header information from.
    output_path (str): The file path to save the corrected LAS file.
    """
    # Create a new LAS object with the same header as the original
    las_corrected = laspy.LasData(las.header)

    # Update the point data with the corrected point cloud
    las_corrected.x = pc_corrected[:, 0]
    las_corrected.y = pc_corrected[:, 1]
    las_corrected.z = pc_corrected[:, 2]
    las_corrected.red = pc_corrected[:, 3].astype(np.uint16)
    las_corrected.green = pc_corrected[:, 4].astype(np.uint16)
    las_corrected.blue = pc_corrected[:, 5].astype(np.uint16)

    # Save the corrected LAS file
    las_corrected.write(output_path)
    print(f"Corrected LAS file saved to: {output_path}")