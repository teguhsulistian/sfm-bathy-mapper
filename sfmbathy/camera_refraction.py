import sys
import numpy as np
import pandas as pd
import matplotlib.path as mplPath
from multiprocessing import Pool, cpu_count
from functools import partial


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

    This version is fully vectorized using NumPy — no Python loops per camera or per sensor angle, and it does not depend on SymPy.

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
    N = eo.shape[0]

    # ── Convert sensor dimensions to meters ──
    f  = sensor.focal[0]    * 1e-3
    sx = sensor.sensor_x[0] * 5e-4   # /2 * 0.001
    sy = sensor.sensor_y[0] * 5e-4

    # ── Critical pitch ──
    crit_pitch = 90.0 - np.rad2deg(np.arctan(sy / f))

    if verbose:
        print(f"Processing {N:,} cameras (chunk={chunk_size}, jobs={n_jobs})...")
        sys.stdout.flush()

    # ── Extract NumPy arrays from DataFrame ──
    xs     = eo['x'].to_numpy(dtype=np.float64)
    ys     = eo['y'].to_numpy(dtype=np.float64)
    zs     = eo['z'].to_numpy(dtype=np.float64)
    yaws   = np.deg2rad(eo['yaw'].to_numpy(dtype=np.float64))
    pitches_raw = eo['pitch'].to_numpy(dtype=np.float64)
    rolls  = np.deg2rad(eo['roll'].to_numpy(dtype=np.float64))

    # Pitch for rotation: 90 - pitch_raw (in radians) for correct orientation
    pitches = np.deg2rad(90.0 - pitches_raw)

    # ── Mark cameras that exceed the critical pitch ──
    valid_mask = pitches_raw < crit_pitch   # (N,) bool

    # Template sensor corners relative to the camera center before rotation
    # Order: Upper-Right, Lower-Right, Lower-Left, Upper-Left → shape (4, 3)

    corner_offsets = np.array([
        [ sx, -f,  sy],
        [ sx, -f, -sy],
        [-sx, -f, -sy],
        [-sx, -f,  sy],
    ], dtype=np.float64)  # (4, 3)

    # ── Prepare output ──
    # all_corners_world: (N, 4, 2) — Ground plane coordinates (X,Y) of the footprint.
    all_inter = np.full((N, 4, 2), np.nan)

    # ── Process inside chunk  ──
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

              # ── Ray–Plane intersection for all (camera × corner) simultaneously ──
        # Ray: from corners_world through cam_pts, direction = cam_pts - corners_world
        # Flatten to (n_chunk*4, 3)
            
        origins_flat = corners_world.reshape(-1, 3)                          # (n_chunk*4, 3)
        cam_pts_flat = np.repeat(cam_pts, 4, axis=0)                         # (n_chunk*4, 3)
        dirs_flat    = cam_pts_flat - origins_flat                            # (n_chunk*4, 3)

        # Normalization of ray directions
        # Using plane_z = mean_elev
        xy_flat = _ray_plane_intersect_batch(origins_flat, dirs_flat, mean_elev)  # (n_chunk*4, 2)

        # Reshape → (n_chunk, 4, 2)
        xy_chunk = xy_flat.reshape(n_chunk, 4, 2)

        all_inter[chunk_idx] = xy_chunk

        if verbose and (start + n_chunk) % max(chunk_size * 5, 1000) < chunk_size:
            print(f"  {start + n_chunk:,} / {len(indices):,} kamera valid diproses...")
            sys.stdout.flush()

    # ── Output DataFrame ──
    fov_list = [mplPath.Path(all_inter[i]) for i in range(N)]
    result = pd.DataFrame({'fov': fov_list})

    if verbose:
        n_invalid = int((~valid_mask).sum())
        print(f"Finished {N - n_invalid:,} footprint valid, {n_invalid:,} NaN (critical pitch).")

    return result

def _contains_single(args):
    """
    Cek apakah pc_xy berada di dalam satu Path object.
    Diperlukan sebagai fungsi top-level agar bisa di-pickle oleh multiprocessing.

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


def visible_points(eo, ifov, pc, n_jobs=1, verbose=False):
    """
    Cek visibilitas titik point cloud (pc) terhadap setiap kamera (eo),
    dan hitung sudut inklinasi r (untuk koreksi refraksi).

    Point cloud (pc) which is visible from a camera is determined by whether its (x,y) coordinates
    and calculated inclination angle r fall within the camera's instantaneous field of view (ifov) polygon.

    Parameters
    ----------
    eo     : pd.DataFrame (N_cam × ≥6)  — columns: x, y, z, ... (exterior orientation)
    ifov   : pd.DataFrame (N_cam × 1)   — columns: fov (matplotlib Path, instantaneous FOV)
    pc     : pd.DataFrame (N_pt × ≥3)   — columns: x, y, z (point cloud)
    n_jobs : int  — number of parallel processes (default 1; -1 = all CPU)
    verbose: bool — display progress and summary

    Returns
    -------
    r_filter : ndarray (N_pt, N_cam) float64
        inclination angle in degrees. NaN if the point is not visible
        from the camera, or if the ifov contains NaN.
    """
    n_cam = eo.shape[0]
    n_pt  = pc.shape[0]

    # 1. Prepare pc coordinates once for all cameras
    # Shape (N_pt, 2) — used repeatedly for all cameras
    pc_xy = np.column_stack([
        pc['x'].to_numpy(dtype=np.float64),
        pc['y'].to_numpy(dtype=np.float64),
    ])

    # 2. Visibility calculating for each constains_points
    # Opsi A (serial) — A stack without overhead Pool
    # Opsi B (parallel) — Distributed in many process

    paths = ifov['fov'].tolist()   # list N_cam Path objects

    if n_jobs == 1:
        # Serial: Faster than original due to no multiprocessing overhead, especially for small N_cam
        vis_cols = [p.contains_points(pc_xy) for p in paths]

    else:
        # Parallel: each process handing one camera's Path.contains_points → faster for large N_cam, but beware of overhead and RAM usage
        workers = cpu_count() if n_jobs == -1 else n_jobs
        args = [(p, pc_xy) for p in paths]
        with Pool(workers) as pool:
            vis_cols = pool.map(_contains_single, args)

    # Stack list of (N_pt,) → array (N_pt, N_cam), dtype bool
    vis = np.column_stack(vis_cols)   # shape (N_pt, N_cam)

    if verbose:
        n_visible = int(vis.sum())
        print(f"Visible points: {n_visible:,} pasangan pc-kamera terlihat "
              f"dari {n_pt * n_cam:,} total.")

    # 3. Calculate delta x, y, z from camera to point cloud for all pairs (N_pt, N_cam)
    # eo_x  : shape (1, N_cam)
    # pc_x  : shape (N_pt, 1)
    # result : shape (N_pt, N_cam)  — automatic broadcasting
    eo_x = eo['x'].to_numpy(dtype=np.float64)[np.newaxis, :]   # (1, N_cam)
    eo_y = eo['y'].to_numpy(dtype=np.float64)[np.newaxis, :]
    eo_z = eo['z'].to_numpy(dtype=np.float64)[np.newaxis, :]

    pc_x = pc['x'].to_numpy(dtype=np.float64)[:, np.newaxis]   # (N_pt, 1)
    pc_y = pc['y'].to_numpy(dtype=np.float64)[:, np.newaxis]
    pc_z = pc['z'].to_numpy(dtype=np.float64)[:, np.newaxis]

    dx = eo_x - pc_x   # (N_pt, N_cam)
    dy = eo_y - pc_y
    dz = eo_z - pc_z

    # 4. Calculate horizontal distance d and refraction angle r
    # np.hypot more stable numerically than sqrt(dx²+dy²)
    # d = Euclidean distance to the SfM point from the camera
    # r = angle of refraction (from nadir to the SfM point)
    d = np.hypot(dx, dy)                      # (N_pt, N_cam)
    r = np.rad2deg(np.arctan2(d, dz))         # arctan2 safe while dz=0

    # 5. Masking: NaN for invisible points or ifov with NaN
    r_filter = np.where(vis, r, np.nan)         # (N_pt, N_cam)

    return r_filter