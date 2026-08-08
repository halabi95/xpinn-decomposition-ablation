"""
==============================================================================
SPHINX RBF Baseline — Traditional Reconstruction Comparison
==============================================================================
Reconstructs the flow fields (u, v, p) from sparse measurements using thin
plate spline radial basis function interpolation, on the SAME sparse points
that SPHINX uses (same RNG seeding). Outputs are written in the same format
as the SPHINX sweep so the aggregation script can read both.

Compares against SPHINX on:
  - 4 cases: cylinder Re=40, NACA Re=1k a=5, NACA Re=10k a=5, FSI1 Re=20
  - 6 interface fractions: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
  - 3 seeds: 42, 123, 456
Total: 72 RBF fits, each takes seconds on CPU.

Usage (run from any of the case directories):
  python sphinx_rbf_baseline.py --case cylinder    --output-root /path/to/rbf_sweep
  python sphinx_rbf_baseline.py --case naca_re1k   --output-root /path/to/rbf_sweep
  python sphinx_rbf_baseline.py --case naca_re10k  --output-root /path/to/rbf_sweep
  python sphinx_rbf_baseline.py --case fsi1        --output-root /path/to/rbf_sweep

Or run all four cases sequentially with the wrapper script run_rbf_all.sh.

Output: <output-root>/<case>/frac_X.X/seed_YYY/metrics.npz with the same field
names as the SPHINX sweep (u_err, v_err, p_err, etc.) plus method='rbf'.
==============================================================================
"""

import argparse
import os
import sys
import time
import numpy as np
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator


# ============================================================================
# CLI
# ============================================================================
parser = argparse.ArgumentParser(description='RBF baseline for SPHINX comparison')
parser.add_argument('--case', required=True,
                    choices=['cylinder', 'naca_re1k', 'naca_re10k', 'fsi1'])
parser.add_argument('--output-root', required=True,
                    help='Root directory for RBF sweep results')
parser.add_argument('--n-points', type=int, default=200)
parser.add_argument('--fracs', type=float, nargs='+',
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
parser.add_argument('--kernel', default='thin_plate_spline',
                    choices=['thin_plate_spline', 'multiquadric', 'gaussian',
                             'inverse_multiquadric', 'linear', 'cubic',
                             'quintic'])
parser.add_argument('--smoothing', type=float, default=0.0,
                    help='RBF regularization (0 = exact interpolation)')
args = parser.parse_args()


# ============================================================================
# Per-case setup: reference file, geometry, subdomain layout
# ============================================================================
def setup_cylinder():
    ref = np.load('cylinder_re40_reference.npz')
    R = float(ref['R']); cx = float(ref['cx']); cy = float(ref['cy'])

    def inside_solid(x, y, margin=0.02):
        return np.sqrt((x - cx)**2 + (y - cy)**2) < R + margin

    x_split = cx; y_split = cy; overlap = 0.5
    x_min = float(ref['x_extract_min']); x_max = float(ref['x_extract_max'])
    y_min = float(ref['y_extract_min']); y_max = float(ref['y_extract_max'])
    subdomains = {
        0: {'x_lo': x_min, 'x_hi': x_split + overlap, 'y_lo': y_min, 'y_hi': y_max},
        1: {'x_lo': x_split - overlap, 'x_hi': x_max,
            'y_lo': y_split - overlap, 'y_hi': y_max},
        2: {'x_lo': x_split - overlap, 'x_hi': x_max,
            'y_lo': y_min, 'y_hi': y_split + overlap},
        3: {'x_lo': cx - 2.0, 'x_hi': cx + 3.0, 'y_lo': cy - 2.0, 'y_hi': cy + 2.0},
    }
    bulk_pad = 0.5; mask_margin = 0.1; interface_margin = 0.1
    return ref, inside_solid, subdomains, bulk_pad, mask_margin, interface_margin


def setup_naca(re_value):
    ref_file = (f'naca0012_re{re_value}_aoa5_reference.npz' if re_value != 1000
                else 'naca0012_re1000_aoa5_reference.npz')
    if not os.path.exists(ref_file):
        print(f"  ERROR: {ref_file} not found in {os.getcwd()}")
        sys.exit(1)
    ref = np.load(ref_file)
    chord = float(ref['chord'])

    def naca_half_thickness(x, c=chord):
        t = 0.12; xc = np.clip(x/c, 0, 1)
        return 5.0*t*(0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*xc**2
                      + 0.2843*xc**3 - 0.1015*xc**4)*c

    def inside_solid(x, y, margin=0.01):
        xc = x / chord
        in_range = (xc >= 0) & (xc <= 1)
        yt = np.zeros_like(x)
        yt[in_range] = naca_half_thickness(x[in_range])
        return in_range & (np.abs(y) < yt + margin)

    x_split = 0.5 * chord; y_split = 0.0; overlap = 0.3
    x_min = float(ref['x_extract_min']); x_max = float(ref['x_extract_max'])
    y_min = float(ref['y_extract_min']); y_max = float(ref['y_extract_max'])
    subdomains = {
        0: {'x_lo': x_min, 'x_hi': x_split + overlap, 'y_lo': y_min, 'y_hi': y_max},
        1: {'x_lo': x_split - overlap, 'x_hi': x_max,
            'y_lo': y_split - overlap, 'y_hi': y_max},
        2: {'x_lo': x_split - overlap, 'x_hi': x_max,
            'y_lo': y_min, 'y_hi': y_split + overlap},
        3: {'x_lo': -0.3, 'x_hi': 1.8, 'y_lo': -0.8, 'y_hi': 0.8},
    }
    bulk_pad = 0.1; mask_margin = 0.02; interface_margin = 0.02
    return ref, inside_solid, subdomains, bulk_pad, mask_margin, interface_margin


def setup_fsi1():
    ref_file = 'turek_hron_fsi1_reference.npz'
    if not os.path.exists(ref_file):
        print(f"  ERROR: {ref_file} not found in {os.getcwd()}")
        sys.exit(1)
    ref = np.load(ref_file)
    # Cylinder + beam geometry from Turek-Hron
    cx_c, cy_c = 0.2, 0.2
    R_c = 0.05  # cylinder radius
    # Beam: from cylinder downstream side to x = 0.6 (length 0.35), thickness 0.02
    beam_xlo = 0.24899  # x where beam meets cylinder
    beam_xhi = 0.6
    beam_yhi = 0.21; beam_ylo = 0.19  # +/- 0.01 around cy

    def inside_solid(x, y, margin=0.005):
        in_cyl = np.sqrt((x - cx_c)**2 + (y - cy_c)**2) < (R_c + margin)
        in_beam = ((x >= beam_xlo - margin) & (x <= beam_xhi + margin) &
                   (y >= beam_ylo - margin) & (y <= beam_yhi + margin))
        return in_cyl | in_beam

    # Channel domain: [0, 2.5] x [0, 0.41]
    x_min, x_max = 0.0, 2.5
    y_min, y_max = 0.0, 0.41
    x_split = 0.5; y_split = 0.2; overlap = 0.1
    subdomains = {
        0: {'x_lo': x_min, 'x_hi': x_split + overlap, 'y_lo': y_min, 'y_hi': y_max},
        1: {'x_lo': x_split - overlap, 'x_hi': x_max,
            'y_lo': y_split - overlap, 'y_hi': y_max},
        2: {'x_lo': x_split - overlap, 'x_hi': x_max,
            'y_lo': y_min, 'y_hi': y_split + overlap},
        3: {'x_lo': 0.1, 'x_hi': 0.8, 'y_lo': 0.1, 'y_hi': 0.3},
    }
    bulk_pad = 0.02; mask_margin = 0.01; interface_margin = 0.01
    return ref, inside_solid, subdomains, bulk_pad, mask_margin, interface_margin


CASE_SETUP = {
    'cylinder':   lambda: setup_cylinder(),
    'naca_re1k':  lambda: setup_naca(1000),
    'naca_re10k': lambda: setup_naca(10000),
    'fsi1':       lambda: setup_fsi1(),
}


# ============================================================================
# Sparse point sampling: matches SPHINX scripts exactly
# ============================================================================
INTERFACE_PAIRS = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]


def sample_sparse_random(n_pts, rng, x_min, x_max, y_min, y_max,
                          bulk_pad, inside_solid, mask_margin):
    if n_pts <= 0:
        return np.zeros((0, 2))
    pts = []
    while len(pts) < n_pts:
        x = rng.uniform(x_min + bulk_pad, x_max - bulk_pad, n_pts * 3)
        y = rng.uniform(y_min + bulk_pad, y_max - bulk_pad, n_pts * 3)
        mask = ~inside_solid(x, y, margin=mask_margin)
        gx, gy = x[mask], y[mask]
        for i in range(min(len(gx), n_pts - len(pts))):
            pts.append([gx[i], gy[i]])
    return np.array(pts[:n_pts])


def sample_interface_data(n_intf, rng, subdomains, inside_solid, interface_margin):
    if n_intf <= 0:
        return np.zeros((0, 2))
    pts = []
    n_per = max(2, (n_intf + len(INTERFACE_PAIRS) - 1) // len(INTERFACE_PAIRS))
    for (i, j) in INTERFACE_PAIRS:
        sd_i, sd_j = subdomains[i], subdomains[j]
        x_lo = max(sd_i['x_lo'], sd_j['x_lo']); x_hi = min(sd_i['x_hi'], sd_j['x_hi'])
        y_lo = max(sd_i['y_lo'], sd_j['y_lo']); y_hi = min(sd_i['y_hi'], sd_j['y_hi'])
        if x_lo >= x_hi or y_lo >= y_hi:
            continue
        x = rng.uniform(x_lo, x_hi, n_per * 3)
        y = rng.uniform(y_lo, y_hi, n_per * 3)
        mask = ~inside_solid(x, y, margin=interface_margin)
        gx, gy = x[mask], y[mask]
        for k in range(min(len(gx), n_per)):
            pts.append([gx[k], gy[k]])
    return np.array(pts) if pts else np.zeros((0, 2))


def generate_sparse_points(n_points, interface_frac, seed_for_layout,
                            subdomains, inside_solid, x_min, x_max, y_min, y_max,
                            bulk_pad, mask_margin, interface_margin):
    """Generate sparse points using the same RNG seeding as the SPHINX scripts."""
    data_rng = np.random.RandomState(
        999 + n_points + int(round(interface_frac * 1000)))
    n_intf = int(round(interface_frac * n_points))
    n_random = n_points - n_intf

    xy_rand = sample_sparse_random(n_random, data_rng, x_min, x_max, y_min, y_max,
                                    bulk_pad, inside_solid, mask_margin)
    xy_intf_all = sample_interface_data(n_intf, data_rng, subdomains,
                                         inside_solid, interface_margin)
    xy_intf = xy_intf_all[:n_intf] if len(xy_intf_all) >= n_intf else xy_intf_all

    if len(xy_rand) > 0 and len(xy_intf) > 0:
        xy = np.vstack([xy_rand, xy_intf])
    elif len(xy_rand) > 0:
        xy = xy_rand
    elif len(xy_intf) > 0:
        xy = xy_intf
    else:
        xy = np.zeros((0, 2))
    return xy, n_random, n_intf


# ============================================================================
# RBF fit and evaluate
# ============================================================================
def rbf_reconstruct(sparse_xy, sparse_u, sparse_v, sparse_p,
                    X_grid, Y_grid, U_ref, V_ref, P_ref,
                    inside_solid, mask_margin, kernel, smoothing):
    """Fit RBF on sparse data and evaluate on the reference grid."""
    x_flat = X_grid.flatten(); y_flat = Y_grid.flatten()
    xy_eval = np.stack([x_flat, y_flat], axis=1)

    # Mask: don't evaluate inside the solid
    mask = ~inside_solid(x_flat, y_flat, margin=mask_margin)

    U_p = np.full_like(x_flat, np.nan)
    V_p = np.full_like(x_flat, np.nan)
    P_p = np.full_like(x_flat, np.nan)

    if len(sparse_xy) < 3:
        return None  # RBF needs at least 3 points

    try:
        rbf_u = RBFInterpolator(sparse_xy, sparse_u, kernel=kernel,
                                 smoothing=smoothing)
        rbf_v = RBFInterpolator(sparse_xy, sparse_v, kernel=kernel,
                                 smoothing=smoothing)
        rbf_p = RBFInterpolator(sparse_xy, sparse_p, kernel=kernel,
                                 smoothing=smoothing)
        # Evaluate in chunks to avoid memory issues on large grids
        chunk = 10000
        u_vals = np.empty(mask.sum()); v_vals = np.empty(mask.sum())
        p_vals = np.empty(mask.sum())
        xy_valid = xy_eval[mask]
        for s in range(0, len(xy_valid), chunk):
            e = min(s + chunk, len(xy_valid))
            u_vals[s:e] = rbf_u(xy_valid[s:e])
            v_vals[s:e] = rbf_v(xy_valid[s:e])
            p_vals[s:e] = rbf_p(xy_valid[s:e])
        U_p[mask] = u_vals; V_p[mask] = v_vals; P_p[mask] = p_vals
    except Exception as e:
        print(f"    RBF fit failed: {e}")
        return None

    U_p = U_p.reshape(X_grid.shape)
    V_p = V_p.reshape(X_grid.shape)
    P_p = P_p.reshape(X_grid.shape)

    valid = ~np.isnan(U_ref) & ~np.isnan(U_p)
    if valid.sum() == 0:
        return None
    ue = (np.sqrt(np.nanmean((U_p[valid] - U_ref[valid])**2)) /
          np.sqrt(np.nanmean(U_ref[valid]**2)) * 100)
    ve = (np.sqrt(np.nanmean((V_p[valid] - V_ref[valid])**2)) /
          np.sqrt(np.nanmean(V_ref[valid]**2)) * 100)
    pe = (np.sqrt(np.nanmean((P_p[valid] - P_ref[valid])**2)) /
          np.sqrt(np.nanmean(P_ref[valid]**2)) * 100)
    return ue, ve, pe, U_p, V_p, P_p


# ============================================================================
# Main: loop over fractions and seeds for this case
# ============================================================================
print("=" * 70)
print(f"  SPHINX RBF Baseline — case: {args.case}")
print(f"  kernel: {args.kernel}, smoothing: {args.smoothing}")
print(f"  N: {args.n_points}, fracs: {args.fracs}, seeds: {args.seeds}")
print(f"  output root: {args.output_root}")
print("=" * 70)

# Set up the case
(ref, inside_solid, subdomains,
 bulk_pad, mask_margin, interface_margin) = CASE_SETUP[args.case]()
X_ref, Y_ref = ref['X'], ref['Y']
U_ref, V_ref, P_ref = ref['U'], ref['V'], ref['P']
x_grid, y_grid = ref['x_grid'], ref['y_grid']

# Domain bounds
if args.case == 'fsi1':
    x_min, x_max = 0.0, 2.5
    y_min, y_max = 0.0, 0.41
else:
    x_min = float(ref['x_extract_min']); x_max = float(ref['x_extract_max'])
    y_min = float(ref['y_extract_min']); y_max = float(ref['y_extract_max'])

# Interpolators for sparse-point reference values
interp_u = RegularGridInterpolator((y_grid, x_grid), U_ref, method='linear',
                                    bounds_error=False, fill_value=np.nan)
interp_v = RegularGridInterpolator((y_grid, x_grid), V_ref, method='linear',
                                    bounds_error=False, fill_value=np.nan)
interp_p = RegularGridInterpolator((y_grid, x_grid), P_ref, method='linear',
                                    bounds_error=False, fill_value=np.nan)

# Output root for this case
case_root = os.path.join(args.output_root, args.case)
os.makedirs(case_root, exist_ok=True)

t_start = time.time()
n_total = len(args.fracs) * len(args.seeds)
n_done = 0
for frac in args.fracs:
    for seed in args.seeds:
        n_done += 1
        t0 = time.time()
        print(f"\n[{n_done}/{n_total}] case={args.case}, frac={frac}, seed={seed}")

        # Generate sparse points (same as SPHINX would, given the same frac)
        # Note: SPHINX scripts ignore `seed` for sparse-layout RNG (they use a
        # seed derived from N and frac only). The trial seed `seed` controls
        # network weights and collocation sampling in SPHINX, which RBF doesn't
        # use. For consistency, we still loop over seeds, but the RBF result
        # for a given (frac) is deterministic across seeds — we record it per
        # seed so the output structure mirrors SPHINX exactly.
        xy, n_random, n_intf = generate_sparse_points(
            args.n_points, frac, seed,
            subdomains, inside_solid, x_min, x_max, y_min, y_max,
            bulk_pad, mask_margin, interface_margin)

        # Sample reference values at sparse points
        if len(xy) == 0:
            print(f"  No sparse points; skipping")
            continue
        yx = xy[:, ::-1]
        u_sp = interp_u(yx); v_sp = interp_v(yx); p_sp = interp_p(yx)
        valid = ~np.isnan(u_sp) & ~np.isnan(v_sp) & ~np.isnan(p_sp)
        sparse_xy = xy[valid]
        sparse_u = u_sp[valid]; sparse_v = v_sp[valid]; sparse_p = p_sp[valid]
        print(f"  Sparse points: {len(sparse_xy)} valid "
              f"({n_random} random + {n_intf} interface target)")

        # Fit RBF and evaluate
        result = rbf_reconstruct(
            sparse_xy, sparse_u, sparse_v, sparse_p,
            X_ref, Y_ref, U_ref, V_ref, P_ref,
            inside_solid, mask_margin, args.kernel, args.smoothing)

        if result is None:
            print(f"  RBF failed for this trial")
            continue
        ue, ve, pe, Up, Vp, Pp = result
        t_trial = time.time() - t0
        print(f"  u_err={ue:.4f}%, v_err={ve:.4f}%, p_err={pe:.4f}%, "
              f"t={t_trial:.1f}s")

        # Save in SPHINX sweep format
        out_dir = os.path.join(case_root, f"frac_{frac}", f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'metrics.npz')
        np.savez(out_path,
            method='rbf',
            kernel=args.kernel,
            smoothing=args.smoothing,
            u_err=ue, v_err=ve, p_err=pe,
            case_tag=args.case,
            interface_frac=frac,
            seed=seed,
            n_points=args.n_points,
            n_intf_target=n_intf,
            n_random_target=n_random,
            n_valid=int(valid.sum()),
            train_time=t_trial,
            U_pred=Up, V_pred=Vp, P_pred=Pp,
            sparse_xy=sparse_xy,
        )

elapsed = time.time() - t_start
print(f"\n{'='*70}")
print(f"  Done: {n_done} RBF fits for case {args.case} in {elapsed:.1f}s "
      f"({elapsed/60:.1f} min)")
print(f"{'='*70}")
