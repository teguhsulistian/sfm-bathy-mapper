import sys
import numpy as np
import pandas as pd
import matplotlib.path as mplPath
from multiprocessing import Pool, cpu_count
from functools import partial


# ──────────────────────────────────────────────
# Ray–Plane intersection (pure NumPy)
# ──────────────────────────────────────────────

def _ray_plane_intersect_batch(ray_origins, ray_dirs, plane_z):
    """
    Hitung titik perpotongan antara sekumpulan sinar dengan bidang horizontal z = plane_z.

    Parameters
    ----------
    ray_origins : ndarray (N, 3)  — titik asal sinar (titik kamera)
    ray_dirs    : ndarray (N, 3)  — arah sinar (dari sudut sensor ke kamera, lalu dilanjut)
    plane_z     : float           — ketinggian bidang tanah

    Returns
    -------
    xy : ndarray (N, 2) — koordinat perpotongan X,Y di bidang tanah
         Baris di-set NaN jika sinar sejajar dengan bidang (tidak berpotongan).
    """
    # t = (plane_z - origin_z) / dir_z
    dz = ray_dirs[:, 2]
    # Hindari division by zero (sinar sejajar bidang)
    valid = np.abs(dz) > 1e-12
    t = np.where(valid, (plane_z - ray_origins[:, 2]) / np.where(valid, dz, 1.0), np.nan)

    # Titik perpotongan
    xy = ray_origins[:, :2] + t[:, np.newaxis] * ray_dirs[:, :2]
    xy[~valid] = np.nan
    return xy


# ──────────────────────────────────────────────
# HELPER: Rotation matrices (batch, vectorized)
# ──────────────────────────────────────────────

def _build_rotation_matrices(pitches, yaws, rolls):
    """
    Bangun matriks rotasi gabungan R = Rx @ Ry @ Rz untuk setiap kamera
    secara serentak (vectorized).

    Parameters
    ----------
    pitches, yaws, rolls : ndarray (N,) dalam radian

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


# ──────────────────────────────────────────────
# FUNGSI UTAMA
# ──────────────────────────────────────────────

def footprints(cam, sensor, base_elev, chunk_size=1000, n_jobs=1, verbose=True):
    """
    Hitung Instantaneous Field of View (IFOV) untuk setiap kamera.

    Versi ini sepenuhnya ter-vektorisasi menggunakan NumPy — tidak ada loop
    Python per kamera maupun per sudut sensor, dan tidak bergantung pada SymPy.

    Parameters
    ----------
    cam        : pd.DataFrame (N × ≥6) — kolom: x, y, z, yaw, pitch, roll
    sensor     : pd.DataFrame (1 × 3)  — kolom: focal (mm), sensor_x (mm), sensor_y (mm)
    base_elev  : float  — elevasi rata-rata permukaan tanah (meter)
    chunk_size : int    — ukuran batch pemrosesan (default 1000; turunkan jika RAM terbatas)
    n_jobs     : int    — jumlah proses paralel (default 1; gunakan -1 untuk semua CPU)
    verbose    : bool   — tampilkan progress

    Returns
    -------
    pd.DataFrame (N × 1) dengan kolom 'fov' berisi matplotlib.path.Path objects.
    Kamera yang melampaui critical pitch akan menghasilkan Path berisi NaN.
    """
    N = cam.shape[0]

    # ── Konversi dimensi sensor ke meter ──
    f  = sensor.focal[0]    * 1e-3
    sx = sensor.sensor_x[0] * 5e-4   # /2 * 0.001
    sy = sensor.sensor_y[0] * 5e-4

    # ── Critical pitch ──
    crit_pitch = 90.0 - np.rad2deg(np.arctan(sy / f))

    if verbose:
        print(f"Memproses {N:,} kamera (chunk={chunk_size}, jobs={n_jobs})...")
        sys.stdout.flush()

    # ── Ekstrak array NumPy dari DataFrame ──
    xs     = cam['x'].to_numpy(dtype=np.float64)
    ys     = cam['y'].to_numpy(dtype=np.float64)
    zs     = cam['z'].to_numpy(dtype=np.float64)
    yaws   = np.deg2rad(cam['yaw'].to_numpy(dtype=np.float64))
    pitches_raw = cam['pitch'].to_numpy(dtype=np.float64)
    rolls  = np.deg2rad(cam['roll'].to_numpy(dtype=np.float64))

    # pitch untuk rotasi: 90 - pitch_raw (dalam radian)
    pitches = np.deg2rad(90.0 - pitches_raw)

    # ── Tandai kamera yang melampaui critical pitch ──
    valid_mask = pitches < crit_pitch   # (N,) bool

    # ── Template sudut sensor relatif terhadap pusat kamera ──
    # Urutan: UR, LR, LL, UL  → shape (4, 3)
    corner_offsets = np.array([
        [ sx, -f,  sy],
        [ sx, -f, -sy],
        [-sx, -f, -sy],
        [-sx, -f,  sy],
    ], dtype=np.float64)  # (4, 3)

    # ── Siapkan output ──
    # all_corners_world: (N, 4, 2) — koordinat XY footprint di tanah
    all_inter = np.full((N, 4, 2), np.nan)

    # ── Proses dalam chunk ──
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    indices = np.where(valid_mask)[0]   # hanya kamera yang valid

    for start in range(0, len(indices), chunk_size):
        chunk_idx = indices[start: start + chunk_size]
        n_chunk   = len(chunk_idx)

        # Posisi kamera batch → (n_chunk, 3)
        cam_pts = np.stack([xs[chunk_idx], ys[chunk_idx], zs[chunk_idx]], axis=1)

        # Matriks rotasi gabungan → (n_chunk, 3, 3)
        R = _build_rotation_matrices(pitches[chunk_idx], yaws[chunk_idx], rolls[chunk_idx])

        # Broadcast corner_offsets ke (n_chunk, 4, 3)
        offsets = np.broadcast_to(corner_offsets, (n_chunk, 4, 3)).copy()

        # Rotasi: (n_chunk, 4, 3) @ (n_chunk, 3, 3)^T → (n_chunk, 4, 3)
        # einsum: for each camera i, each corner j: out[i,j] = offsets[i,j] @ R[i]
        rotated = np.einsum('ncv,nuv->ncu', offsets, R)   # (n_chunk, 4, 3)

        # Tambah posisi kamera → koordinat dunia sudut sensor
        corners_world = rotated + cam_pts[:, np.newaxis, :]   # (n_chunk, 4, 3)

        # ── Ray–Plane intersection untuk semua (kamera × corner) sekaligus ──
        # Ray: dari corners_world melewati cam_pts, arah = cam_pts - corners_world
        # Flatten ke (n_chunk*4, 3)
        origins_flat = corners_world.reshape(-1, 3)                          # (n_chunk*4, 3)
        cam_pts_flat = np.repeat(cam_pts, 4, axis=0)                         # (n_chunk*4, 3)
        dirs_flat    = cam_pts_flat - origins_flat                            # (n_chunk*4, 3)

        # Normalisasi arah (opsional, tidak wajib untuk interseksi)
        # Gunakan plane_z = base_elev
        xy_flat = _ray_plane_intersect_batch(origins_flat, dirs_flat, base_elev)  # (n_chunk*4, 2)

        # Reshape kembali → (n_chunk, 4, 2)
        xy_chunk = xy_flat.reshape(n_chunk, 4, 2)

        all_inter[chunk_idx] = xy_chunk

        if verbose and (start + n_chunk) % max(chunk_size * 5, 1000) < chunk_size:
            print(f"  {start + n_chunk:,} / {len(indices):,} kamera valid diproses...")
            sys.stdout.flush()

    # ── Bangun output DataFrame ──
    fov_list = [mplPath.Path(all_inter[i]) for i in range(N)]
    result = pd.DataFrame({'fov': fov_list})

    if verbose:
        n_invalid = int((~valid_mask).sum())
        print(f"Selesai. {N - n_invalid:,} footprint valid, {n_invalid:,} NaN (critical pitch).")

    return result


# ──────────────────────────────────────────────
# FUNGSI UTILITAS TAMBAHAN
# ──────────────────────────────────────────────

def footprint_bounds(fp_df):
    """
    Hitung bounding box (minx, miny, maxx, maxy) untuk setiap footprint.

    Parameters
    ----------
    fp_df : pd.DataFrame — output dari footprints()

    Returns
    -------
    pd.DataFrame dengan kolom: minx, miny, maxx, maxy
    """
    verts = np.array([p.vertices for p in fp_df['fov']])  # (N, 4, 2)
    return pd.DataFrame({
        'minx': np.nanmin(verts[:, :, 0], axis=1),
        'miny': np.nanmin(verts[:, :, 1], axis=1),
        'maxx': np.nanmax(verts[:, :, 0], axis=1),
        'maxy': np.nanmax(verts[:, :, 1], axis=1),
    })


def contains_point_batch(fp_df, points):
    """
    Cek apakah setiap titik dalam `points` berada di dalam footprint masing-masing.

    Parameters
    ----------
    fp_df  : pd.DataFrame — output dari footprints()
    points : ndarray (N, 2) — koordinat XY yang akan dicek

    Returns
    -------
    ndarray (N,) bool
    """
    return np.array([
        fp_df['fov'].iloc[i].contains_point(points[i])
        for i in range(len(fp_df))
    ])


# ──────────────────────────────────────────────
# CONTOH PENGGUNAAN & BENCHMARK
# ──────────────────────────────────────────────

if __name__ == '__main__':
    import time

    # ── Generate data kamera dummy ──
    N = 5000
    rng = np.random.default_rng(42)

    cam = pd.DataFrame({
        'x':     rng.uniform(0, 1000, N),
        'y':     rng.uniform(0, 1000, N),
        'z':     rng.uniform(50, 150, N),
        'yaw':   rng.uniform(0, 360, N),
        'pitch': rng.uniform(0, 45, N),     # di bawah critical pitch
        'roll':  rng.uniform(-5, 5, N),
    })

    sensor = pd.DataFrame({
        'focal':    [35.0],   # mm
        'sensor_x': [36.0],   # mm
        'sensor_y': [24.0],   # mm
    })

    base_elev = 0.0

    # ── Jalankan & ukur waktu ──
    t0 = time.perf_counter()
    fp = footprints(cam, sensor, base_elev, chunk_size=1000, verbose=True)
    t1 = time.perf_counter()

    print(f"\nWaktu pemrosesan : {t1 - t0:.3f} detik untuk {N:,} kamera")
    print(f"Throughput       : {N / (t1 - t0):,.0f} kamera/detik")
    print(f"\nContoh footprint pertama:\n{fp['fov'].iloc[0].vertices}")

    # ── Bounding boxes ──
    bounds = footprint_bounds(fp)
    print(f"\nBounding box 5 footprint pertama:\n{bounds.head()}")
