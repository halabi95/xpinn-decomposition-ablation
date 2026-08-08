"""
==============================================================================
SPHINX-FSI: Fluid-Only XPINN with Prescribed Structural Motion
==============================================================================
Option B: The beam deformation is taken from turtleFSI reference solutions.
SPHINX reconstructs the fluid field (u, v, p) around the deforming beam.

Key contribution: Interface-aware sparse data placement (30% at FSI interface
+ 70% random) improves fluid predictions near the FSI boundary — the region
where Farea et al. (2025) showed errors are highest (12.9% → 2.39%).

Architecture:
  - 4 fluid subdomains (XPINN), 8×128 per subdomain
  - FSI1: Steady, input (x,y), output (u,v,p)
  - FSI2/3: Unsteady, input (x,y,t) with Fourier time encoding, output (u,v,p)
  - Beam surface: no-slip BC with velocity = ∂d/∂t from reference
  - Sparse data from turtleFSI fluid solution

Usage:
  python sphinx_fsi_fluid.py --case fsi1 --n_sparse 200
  python sphinx_fsi_fluid.py --case fsi2 --n_sparse 200
  python sphinx_fsi_fluid.py --case fsi1 --n_sparse 0   # XPINN baseline

Reference: Turek & Hron (2006), validated via turtleFSI
==============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# 0. CONFIGURATION
# ============================================================================
case = 'fsi1'
n_sparse = 200
placement_mode = 'interface'
interface_frac = 0.3      # NEW: configurable for sensitivity sweep (was hard-coded 0.3)
output_dir = None         # NEW: if set, save metrics.npz here
N_epochs = 100000
N_trials = 3
seeds = [42, 123, 456]

for i, arg in enumerate(sys.argv):
    if arg == '--case' and i + 1 < len(sys.argv):
        case = sys.argv[i + 1]
    if arg == '--n_sparse' and i + 1 < len(sys.argv):
        n_sparse = int(sys.argv[i + 1])
    if arg == '--mode' and i + 1 < len(sys.argv):
        placement_mode = sys.argv[i + 1]
    if arg == '--interface-frac' and i + 1 < len(sys.argv):     # NEW
        interface_frac = float(sys.argv[i + 1])
        placement_mode = 'interface' if interface_frac > 0 else 'random'
    if arg == '--output-dir' and i + 1 < len(sys.argv):         # NEW
        output_dir = sys.argv[i + 1]
    if arg == '--epochs' and i + 1 < len(sys.argv):
        N_epochs = int(sys.argv[i + 1])
    if arg == '--trials' and i + 1 < len(sys.argv):
        N_trials = int(sys.argv[i + 1])
    if arg == '--seed' and i + 1 < len(sys.argv):
        seeds = [int(sys.argv[i + 1])]
        N_trials = 1

if not (0.0 <= interface_frac <= 1.0):
    raise ValueError(f"interface_frac must be in [0, 1], got {interface_frac}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
is_steady = (case == 'fsi1')

print(f"{'='*70}")
print(f"  SPHINX-FSI (Option B): Fluid-Only — {case.upper()}")
print(f"  {'Steady' if is_steady else 'Unsteady'} | N_sparse={n_sparse} | "
      f"{placement_mode} | {N_epochs} epochs | {N_trials} trials")
print(f"{'='*70}")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# 1. LOAD REFERENCE DATA
# ============================================================================
ref_file = f'turek_hron_{case}_reference.npz'
if not os.path.exists(ref_file):
    print(f"  ERROR: {ref_file} not found!"); sys.exit(1)

ref = np.load(ref_file)
coords = ref['coordinates']           # (N_nodes, 2) — reference config
topology = ref['topology']            # (N_elem, 3)
domain_markers = ref['domain_markers']  # 1=fluid, 2=solid
times_ref = ref['times']              # (N_snap,)
velocity_ref = ref['velocity']        # (N_snap, N_nodes, 2)
pressure_ref = ref['pressure']        # (N_snap, N_nodes)
displacement_ref = ref['displacement']  # (N_snap, N_nodes, 2)

N_nodes = coords.shape[0]
N_snap = len(times_ref)

print(f"  Reference: {N_snap} snapshots, {N_nodes} nodes, "
      f"{topology.shape[0]} elements")


# ============================================================================
# 2. IDENTIFY REGIONS
# ============================================================================
node_is_solid = np.zeros(N_nodes, dtype=bool)
node_is_fluid = np.zeros(N_nodes, dtype=bool)
for ei in range(topology.shape[0]):
    nodes = topology[ei]
    if domain_markers[ei] == 2:
        node_is_solid[nodes] = True
    else:
        node_is_fluid[nodes] = True

node_is_fsi = node_is_solid & node_is_fluid
node_fluid_only = node_is_fluid & ~node_is_solid

# Channel/geometry parameters
x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
H = y_max - y_min  # 0.41m
cx, cy, R = 0.2, 0.2, 0.05

# Beam geometry (reference config)
solid_coords = coords[node_is_solid]
beam_x_min = solid_coords[:, 0].min()
beam_x_max = solid_coords[:, 0].max()
beam_y_min = solid_coords[:, 1].min()
beam_y_max = solid_coords[:, 1].max()

print(f"  Channel: [{x_min},{x_max}] × [{y_min},{y_max}]")
print(f"  Beam: [{beam_x_min:.4f},{beam_x_max:.4f}] × "
      f"[{beam_y_min:.4f},{beam_y_max:.4f}]")
print(f"  Fluid nodes: {node_fluid_only.sum()}, "
      f"FSI interface: {node_is_fsi.sum()}, "
      f"Solid nodes: {node_is_solid.sum()}")


# ============================================================================
# 3. PHYSICAL PARAMETERS & NORMALIZATION
# ============================================================================
rho_f = 1000.0
nu_f = 0.001

if case == 'fsi1':
    U_mean = 0.2
elif case == 'fsi2':
    U_mean = 1.0
elif case == 'fsi3':
    U_mean = 2.0

Re = U_mean * 2 * R / nu_f

# Normalization
L_ref = H       # length scale
U_ref = U_mean  # velocity scale
P_ref_scale = rho_f * U_mean**2  # pressure scale
T_ref = L_ref / U_ref  # time scale

nu_n = nu_f / (U_ref * L_ref)  # normalized viscosity

print(f"  Re = {Re:.0f}, nu_n = {nu_n:.6f}")
print(f"  Scales: L={L_ref}, U={U_ref}, P={P_ref_scale:.1f}, T={T_ref:.4f}")

# Normalize coordinates
coords_n = coords / L_ref
x_min_n, x_max_n = x_min / L_ref, x_max / L_ref
y_min_n, y_max_n = y_min / L_ref, y_max / L_ref

# Normalize reference fields
vel_n = velocity_ref / U_ref         # (N_snap, N_nodes, 2)
pres_n = pressure_ref / P_ref_scale  # (N_snap, N_nodes)
disp_n = displacement_ref / L_ref    # (N_snap, N_nodes, 2)
times_n = times_ref / T_ref          # (N_snap,)


# ============================================================================
# 4. SELECT TRAINING SNAPSHOTS
# ============================================================================
if is_steady:
    # Use final snapshot
    train_snaps = [N_snap - 1]
    T_min_n, T_max_n = 0.0, 1.0  # dummy
    print(f"  Steady: using final snapshot (t={times_ref[-1]:.2f}s)")
else:
    # Use snapshots from the periodic regime (last 50% of simulation)
    t_cutoff = 0.5 * times_ref[-1]
    train_snaps = np.where(times_ref >= t_cutoff)[0].tolist()
    T_min_n = times_n[train_snaps[0]]
    T_max_n = times_n[train_snaps[-1]]
    print(f"  Unsteady: {len(train_snaps)} snapshots, "
          f"t=[{times_ref[train_snaps[0]]:.2f}, {times_ref[train_snaps[-1]]:.2f}]s")
    print(f"  Normalized t=[{T_min_n:.2f}, {T_max_n:.2f}]")


# ============================================================================
# 5. DEFORMED BEAM GEOMETRY (prescribed from turtleFSI)
# ============================================================================
def get_beam_boundary(snap_idx):
    """Get deformed beam surface coordinates at a given snapshot.

    Returns arrays of (x, y) for top, bottom, tip surfaces and
    the beam velocity (∂d/∂t) at those points.
    """
    # Deformed coordinates
    d = displacement_ref[snap_idx]  # (N_nodes, 2)
    deformed = coords + d  # (N_nodes, 2)

    # Beam surface nodes (not including cylinder attachment)
    solid_idx = np.where(node_is_solid)[0]
    beam_deformed = deformed[solid_idx]

    # Identify surface nodes by extremal y or x values
    # Top surface: nodes near max y for each x
    # Bottom surface: nodes near min y for each x
    # Tip: nodes near max x

    # For the thin beam, we can use all solid boundary nodes
    # The FSI interface nodes are the best approximation
    fsi_idx = np.where(node_is_fsi)[0]
    fsi_deformed = deformed[fsi_idx]

    # Beam velocity (finite difference for unsteady)
    if not is_steady and snap_idx > 0:
        dt = times_ref[snap_idx] - times_ref[snap_idx - 1]
        d_prev = displacement_ref[snap_idx - 1]
        vel_beam = (d - d_prev) / dt  # (N_nodes, 2)
    else:
        vel_beam = np.zeros_like(d)

    return {
        'fsi_coords': fsi_deformed,
        'fsi_coords_n': fsi_deformed / L_ref,
        'fsi_vel': vel_beam[fsi_idx] / U_ref,  # normalized
        'all_solid_deformed': beam_deformed,
        'all_solid_deformed_n': beam_deformed / L_ref,
    }


def inside_cylinder_n(x, y, margin=0.0):
    """Check if normalized points are inside the cylinder."""
    cx_n, cy_n, R_n = cx / L_ref, cy / L_ref, R / L_ref
    r = np.sqrt((x - cx_n)**2 + (y - cy_n)**2)
    return r < R_n + margin


def inside_beam_deformed_n(x, y, snap_idx, margin=0.002/0.41):
    """Check if normalized points are inside the deformed beam.

    Uses a simple bounding box of the deformed beam + margin.
    For more accuracy, we could use point-in-polygon, but the beam
    is thin enough that a bounding box works well.
    """
    d = displacement_ref[snap_idx]
    deformed = coords + d
    solid_def = deformed[node_is_solid]
    bx_lo = solid_def[:, 0].min() / L_ref - margin
    bx_hi = solid_def[:, 0].max() / L_ref + margin
    by_lo = solid_def[:, 1].min() / L_ref - margin
    by_hi = solid_def[:, 1].max() / L_ref + margin
    return (x >= bx_lo) & (x <= bx_hi) & (y >= by_lo) & (y <= by_hi)


def inside_beam_deformed_fast(x, y, bbox, margin=0.002/0.41):
    """Fast version using precomputed bounding box tuple (xlo, xhi, ylo, yhi)."""
    return ((x >= bbox[0] - margin) & (x <= bbox[1] + margin) &
            (y >= bbox[2] - margin) & (y <= bbox[3] + margin))


def inside_obstacle_n(x, y, snap_idx, margin=0.003/0.41):
    """Check if points are inside cylinder OR deformed beam."""
    return inside_cylinder_n(x, y, margin) | inside_beam_deformed_n(x, y, snap_idx, margin)


def inside_obstacle_fast(x, y, bbox, margin=0.003/0.41):
    """Fast version using precomputed beam bounding box."""
    return inside_cylinder_n(x, y, margin) | inside_beam_deformed_fast(x, y, bbox, margin)


# ============================================================================
# 6. DOMAIN DECOMPOSITION (4 fluid subdomains)
# ============================================================================
ovlp = 0.15 / L_ref
beam_mid_x_n = (beam_x_min + beam_x_max) / 2.0 / L_ref
cy_n = cy / L_ref

subdomains = {
    0: {'name': 'Upstream',
        'x_lo': x_min_n, 'x_hi': beam_mid_x_n + ovlp,
        'y_lo': y_min_n, 'y_hi': y_max_n},
    1: {'name': 'Top-downstream',
        'x_lo': beam_mid_x_n - ovlp, 'x_hi': x_max_n,
        'y_lo': cy_n - ovlp, 'y_hi': y_max_n},
    2: {'name': 'Bot-downstream',
        'x_lo': beam_mid_x_n - ovlp, 'x_hi': x_max_n,
        'y_lo': y_min_n, 'y_hi': cy_n + ovlp},
    3: {'name': 'Near-body',
        'x_lo': (cx - 3*R) / L_ref, 'x_hi': (beam_x_max + 5*R) / L_ref,
        'y_lo': (cy - 5*R) / L_ref, 'y_hi': (cy + 5*R) / L_ref},
}

interfaces = [
    (0, 1, 'vertical'), (0, 2, 'vertical'), (1, 2, 'horizontal'),
    (0, 3, 'overlap'), (1, 3, 'overlap'), (2, 3, 'overlap'),
]

print(f"\n  Domain decomposition: {len(subdomains)} subdomains")
for k, sd in subdomains.items():
    print(f"    Ω{k} ({sd['name']}): [{sd['x_lo']:.3f},{sd['x_hi']:.3f}] "
          f"× [{sd['y_lo']:.3f},{sd['y_hi']:.3f}]")


# ============================================================================
# 7. NETWORK ARCHITECTURE
# ============================================================================
class FluidPINN(nn.Module):
    """Fluid subdomain network.

    Steady:   (x, y) → (u, v, p)
    Unsteady: (x, y, t_fourier_features...) → (u, v, p)
    """
    def __init__(self, n_input=2, n_output=3, n_layers=8, n_neurons=128):
        super().__init__()
        layers = [nn.Linear(n_input, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, n_output))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# Fourier time encoding for unsteady cases
N_fourier = 4  # number of Fourier features for time

def fourier_time_features(t, n_features=N_fourier):
    """Encode time with Fourier features: [sin(πt), cos(πt), sin(2πt), ...]"""
    features = []
    for k in range(1, n_features + 1):
        features.append(torch.sin(k * np.pi * t))
        features.append(torch.cos(k * np.pi * t))
    return torch.cat(features, dim=-1)


if is_steady:
    n_input = 2  # (x, y)
else:
    n_input = 2 + 2 * N_fourier  # (x, y, sin(πt), cos(πt), ...)

_tmp = FluidPINN(n_input=n_input)
_n_params = sum(p.numel() for p in _tmp.parameters())
print(f"  Network: 8×128, input_dim={n_input}, {_n_params:,} params/subdomain, "
      f"{_n_params*4:,} total")
del _tmp


# ============================================================================
# 8. SAMPLING FUNCTIONS
# ============================================================================
def sample_subdomain(sd, n, rng, snap_idx=-1, bbox=None):
    """Sample collocation points in fluid subdomain, excluding obstacles."""
    pts = []
    margin = 0.005 / L_ref
    while len(pts) < n:
        x = rng.uniform(sd['x_lo'], sd['x_hi'], n * 3)
        y = rng.uniform(sd['y_lo'], sd['y_hi'], n * 3)
        if bbox is not None:
            mask = ~inside_obstacle_fast(x, y, bbox, margin)
        else:
            mask = ~inside_obstacle_n(x, y, snap_idx, margin)
        gx, gy = x[mask], y[mask]
        for i in range(min(len(gx), n - len(pts))):
            pts.append([gx[i], gy[i]])
    return np.array(pts[:n])


def sample_interface(sd_i, sd_j, n, rng, snap_idx=-1, bbox=None):
    """Sample points in overlap region between two subdomains."""
    x_lo = max(sd_i['x_lo'], sd_j['x_lo'])
    x_hi = min(sd_i['x_hi'], sd_j['x_hi'])
    y_lo = max(sd_i['y_lo'], sd_j['y_lo'])
    y_hi = min(sd_i['y_hi'], sd_j['y_hi'])
    if x_lo >= x_hi or y_lo >= y_hi:
        return np.zeros((0, 2))
    pts = []
    margin = 0.005 / L_ref
    while len(pts) < n:
        x = rng.uniform(x_lo, x_hi, n * 3)
        y = rng.uniform(y_lo, y_hi, n * 3)
        if bbox is not None:
            mask = ~inside_obstacle_fast(x, y, bbox, margin)
        else:
            mask = ~inside_obstacle_n(x, y, snap_idx, margin)
        gx, gy = x[mask], y[mask]
        for i in range(min(len(gx), n - len(pts))):
            pts.append([gx[i], gy[i]])
    return np.array(pts[:n])


def point_in_subdomain(x, y, sd):
    return ((x >= sd['x_lo']) & (x <= sd['x_hi']) &
            (y >= sd['y_lo']) & (y <= sd['y_hi']))


def parabolic_inlet(y_n):
    """Parabolic inlet profile: u = 4 U_mean y(H-y)/H², normalized."""
    y_dim = y_n * L_ref
    return 4.0 * U_mean * y_dim * (H - y_dim) / H**2 / U_ref


def get_bc_points(sd, snap_idx=-1):
    """Boundary conditions for a fluid subdomain.

    Includes: cylinder no-slip, deformed beam no-slip, parabolic inlet,
    channel walls (no-slip).
    """
    xy, u_vals, v_vals = [], [], []

    # --- Cylinder wall (no-slip) ---
    theta = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    cx_n, cy_n_loc, R_n = cx / L_ref, cy / L_ref, R / L_ref
    offset = 0.002 / L_ref
    for t in theta:
        px = cx_n + (R_n + offset) * np.cos(t)
        py = cy_n_loc + (R_n + offset) * np.sin(t)
        if sd['x_lo'] <= px <= sd['x_hi'] and sd['y_lo'] <= py <= sd['y_hi']:
            xy.append([px, py]); u_vals.append(0.0); v_vals.append(0.0)

    # --- Deformed beam surface (no-slip or moving wall) ---
    beam_info = get_beam_boundary(snap_idx)
    fsi_pts = beam_info['fsi_coords_n']
    fsi_vel = beam_info['fsi_vel']  # (N_fsi, 2) normalized

    for j in range(len(fsi_pts)):
        px, py = fsi_pts[j, 0], fsi_pts[j, 1]
        if sd['x_lo'] <= px <= sd['x_hi'] and sd['y_lo'] <= py <= sd['y_hi']:
            xy.append([px, py])
            u_vals.append(float(fsi_vel[j, 0]))
            v_vals.append(float(fsi_vel[j, 1]))

    # --- Parabolic inlet ---
    if sd['x_lo'] <= x_min_n + 0.001:
        for yi in np.linspace(max(sd['y_lo'], y_min_n),
                               min(sd['y_hi'], y_max_n), 150):
            xy.append([x_min_n, yi])
            u_vals.append(parabolic_inlet(yi))
            v_vals.append(0.0)

    # --- Top wall (no-slip) ---
    if sd['y_hi'] >= y_max_n - 0.001:
        for xi in np.linspace(max(sd['x_lo'], x_min_n),
                               min(sd['x_hi'], x_max_n), 100):
            xy.append([xi, y_max_n]); u_vals.append(0.0); v_vals.append(0.0)

    # --- Bottom wall (no-slip) ---
    if sd['y_lo'] <= y_min_n + 0.001:
        for xi in np.linspace(max(sd['x_lo'], x_min_n),
                               min(sd['x_hi'], x_max_n), 100):
            xy.append([xi, y_min_n]); u_vals.append(0.0); v_vals.append(0.0)

    if len(xy) == 0:
        return None, None, None
    return (torch.tensor(np.array(xy), dtype=torch.float32).to(device),
            torch.tensor(np.array(u_vals), dtype=torch.float32).to(device).unsqueeze(1),
            torch.tensor(np.array(v_vals), dtype=torch.float32).to(device).unsqueeze(1))


# ============================================================================
# 9. PREPARE MODEL INPUT (handles steady vs unsteady)
# ============================================================================
def make_input(xy, t_n=None):
    """Create network input tensor.

    Steady:   just (x, y)
    Unsteady: (x, y, fourier_features(t))
    """
    if is_steady:
        return xy
    else:
        t_col = torch.full((xy.shape[0], 1), t_n, dtype=torch.float32,
                           device=xy.device)
        ft = fourier_time_features(t_col, N_fourier)
        return torch.cat([xy, ft], dim=1)


# ============================================================================
# 10. PDE RESIDUAL
# ============================================================================
def compute_pde_loss(model, xy_c, t_n=None):
    """Incompressible Navier-Stokes residual.

    Steady:   u·∇u + ∇p - ν∇²u = 0,  ∇·u = 0
    Unsteady: ∂u/∂t + u·∇u + ∇p - ν∇²u = 0,  ∇·u = 0

    For unsteady, we differentiate through the Fourier features w.r.t. t.
    """
    xy_c = xy_c.detach().requires_grad_(True)

    if is_steady:
        inp = xy_c
    else:
        # Need t as a differentiable variable
        t_col = torch.full((xy_c.shape[0], 1), t_n, dtype=torch.float32,
                           device=device, requires_grad=True)
        ft = fourier_time_features(t_col, N_fourier)
        inp = torch.cat([xy_c, ft], dim=1)

    out = model(inp)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    # Spatial gradients (through xy_c)
    gu = torch.autograd.grad(u, xy_c, torch.ones_like(u), create_graph=True)[0]
    gv = torch.autograd.grad(v, xy_c, torch.ones_like(v), create_graph=True)[0]
    gp = torch.autograd.grad(p, xy_c, torch.ones_like(p), create_graph=True)[0]

    du_dx, du_dy = gu[:, 0:1], gu[:, 1:2]
    dv_dx, dv_dy = gv[:, 0:1], gv[:, 1:2]
    dp_dx, dp_dy = gp[:, 0:1], gp[:, 1:2]

    # Laplacian
    d2u = (torch.autograd.grad(du_dx, xy_c, torch.ones_like(du_dx),
                                create_graph=True)[0][:, 0:1] +
           torch.autograd.grad(du_dy, xy_c, torch.ones_like(du_dy),
                                create_graph=True)[0][:, 1:2])
    d2v = (torch.autograd.grad(dv_dx, xy_c, torch.ones_like(dv_dx),
                                create_graph=True)[0][:, 0:1] +
           torch.autograd.grad(dv_dy, xy_c, torch.ones_like(dv_dy),
                                create_graph=True)[0][:, 1:2])

    # Continuity
    rc = du_dx + dv_dy

    if is_steady:
        # Steady N-S
        rx = u * du_dx + v * du_dy + dp_dx - nu_n * d2u
        ry = u * dv_dx + v * dv_dy + dp_dy - nu_n * d2v
    else:
        # Time derivatives (through Fourier features via t_col)
        du_dt = torch.autograd.grad(u, t_col, torch.ones_like(u),
                                     create_graph=True)[0]
        dv_dt = torch.autograd.grad(v, t_col, torch.ones_like(v),
                                     create_graph=True)[0]
        # Unsteady N-S
        rx = du_dt + u * du_dx + v * du_dy + dp_dx - nu_n * d2u
        ry = dv_dt + u * dv_dx + v * dv_dy + dp_dy - nu_n * d2v

    return torch.mean(rc**2) + torch.mean(rx**2) + torch.mean(ry**2)


# ============================================================================
# 11. SPARSE DATA GENERATION
# ============================================================================
def generate_sparse_data(n_total, rng, mode='interface', frac=0.3):
    """Generate sparse measurement data from turtleFSI reference.

    For steady (FSI1): sample from final snapshot fluid nodes.
    For unsteady (FSI2/3): sample from multiple periodic snapshots.

    Interface-aware: 30% at FSI interface nodes, 70% random fluid nodes.
    """
    if n_total == 0:
        return None

    if mode == 'interface':
        n_fsi = int(round(frac * n_total))   # Was: int(0.3 * n_total)
        n_random = n_total - n_fsi
    else:
        n_fsi = 0
        n_random = n_total

    all_xy, all_u, all_v, all_p = [], [], [], []

    if is_steady:
        snap = train_snaps[0]
        u_snap = vel_n[snap, :, 0]
        v_snap = vel_n[snap, :, 1]
        p_snap = pres_n[snap]

        # Random fluid nodes
        fluid_idx = np.where(node_fluid_only)[0]
        if n_random > 0:
            chosen = rng.choice(fluid_idx, size=min(n_random, len(fluid_idx)),
                                replace=False)
            for idx in chosen:
                all_xy.append(coords_n[idx])
                all_u.append(u_snap[idx])
                all_v.append(v_snap[idx])
                all_p.append(p_snap[idx])

        # FSI interface nodes
        if n_fsi > 0:
            fsi_idx = np.where(node_is_fsi)[0]
            chosen_fsi = rng.choice(fsi_idx,
                                     size=min(n_fsi, len(fsi_idx)),
                                     replace=True)
            for idx in chosen_fsi:
                all_xy.append(coords_n[idx])
                all_u.append(u_snap[idx])
                all_v.append(v_snap[idx])
                all_p.append(p_snap[idx])

    else:
        # Unsteady: sample across time snapshots
        n_per_snap = max(1, n_total // len(train_snaps))
        n_fsi_per = max(1, n_fsi // len(train_snaps)) if n_fsi > 0 else 0
        n_rand_per = n_per_snap - n_fsi_per

        for snap in train_snaps:
            u_snap = vel_n[snap, :, 0]
            v_snap = vel_n[snap, :, 1]
            p_snap = pres_n[snap]
            t_snap = times_n[snap]

            # Random fluid nodes at this snapshot
            fluid_idx = np.where(node_fluid_only)[0]
            if n_rand_per > 0:
                chosen = rng.choice(fluid_idx,
                                     size=min(n_rand_per, len(fluid_idx)),
                                     replace=False)
                for idx in chosen:
                    all_xy.append(np.append(coords_n[idx], t_snap))
                    all_u.append(u_snap[idx])
                    all_v.append(v_snap[idx])
                    all_p.append(p_snap[idx])

            # FSI interface nodes at this snapshot
            if n_fsi_per > 0:
                fsi_idx = np.where(node_is_fsi)[0]
                chosen_fsi = rng.choice(fsi_idx,
                                         size=min(n_fsi_per, len(fsi_idx)),
                                         replace=True)
                for idx in chosen_fsi:
                    all_xy.append(np.append(coords_n[idx], t_snap))
                    all_u.append(u_snap[idx])
                    all_v.append(v_snap[idx])
                    all_p.append(p_snap[idx])

    return {
        'xy': np.array(all_xy),
        'u': np.array(all_u),
        'v': np.array(all_v),
        'p': np.array(all_p),
    }


# ============================================================================
# 12. XPINN TRAINING
# ============================================================================
def run_trial(seed, sparse_data=None, N_epochs=100000,
              lambda_data=10.0, lambda_intf=10.0):
    """Run one SPHINX-FSI training trial (fluid only)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    rng = np.random.RandomState(seed)

    # Create models
    models = {k: FluidPINN(n_input=n_input).to(device) for k in subdomains}

    # Precompute BC for all training snapshots (critical for unsteady perf)
    print(f"    Precomputing BCs for {len(train_snaps)} snapshots...")
    t_precomp = time.time()
    bc_cache = {}  # {snap_idx: {subdomain_key: (xy, u, v)}}
    for si, snap in enumerate(train_snaps):
        bc_cache[snap] = {k: get_bc_points(sd, snap) for k, sd in subdomains.items()}
        if (si + 1) % 20 == 0 or si == len(train_snaps) - 1:
            print(f"      [{si+1}/{len(train_snaps)}]")
    print(f"    BC precomputation: {time.time()-t_precomp:.1f}s")

    # Also precompute deformed beam bounding boxes for obstacle exclusion
    beam_bbox_cache = {}
    for snap in train_snaps:
        d = displacement_ref[snap]
        deformed = coords + d
        solid_def = deformed[node_is_solid]
        beam_bbox_cache[snap] = (
            solid_def[:, 0].min() / L_ref,
            solid_def[:, 0].max() / L_ref,
            solid_def[:, 1].min() / L_ref,
            solid_def[:, 1].max() / L_ref,
        )

    if is_steady:
        snap_idx = train_snaps[0]
        bc_data = bc_cache[snap_idx]

    # Distribute sparse data to subdomains
    has_data = sparse_data is not None
    sparse_per_sd = {k: None for k in subdomains}

    if has_data:
        xy_sp = sparse_data['xy']
        # For steady: xy_sp is (N, 2); for unsteady: (N, 3) with time
        xy_spatial = xy_sp[:, :2] if xy_sp.shape[1] > 2 else xy_sp

        for k, sd in subdomains.items():
            mask = (point_in_subdomain(xy_spatial[:, 0], xy_spatial[:, 1], sd) &
                    ~inside_obstacle_n(xy_spatial[:, 0], xy_spatial[:, 1],
                                        train_snaps[0], 0.0))
            if mask.any():
                if is_steady:
                    inp_sp = torch.tensor(xy_sp[mask], dtype=torch.float32).to(device)
                else:
                    # Build full input with Fourier features
                    xy_m = torch.tensor(xy_sp[mask, :2], dtype=torch.float32).to(device)
                    t_m = torch.tensor(xy_sp[mask, 2:3], dtype=torch.float32).to(device)
                    ft_m = fourier_time_features(t_m, N_fourier)
                    inp_sp = torch.cat([xy_m, ft_m], dim=1)

                sparse_per_sd[k] = {
                    'inp': inp_sp,
                    'u': torch.tensor(sparse_data['u'][mask], dtype=torch.float32).to(device).unsqueeze(1),
                    'v': torch.tensor(sparse_data['v'][mask], dtype=torch.float32).to(device).unsqueeze(1),
                    'p': torch.tensor(sparse_data['p'][mask], dtype=torch.float32).to(device).unsqueeze(1),
                }

    # Optimizer
    all_params = []
    for m in models.values():
        all_params += list(m.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_epochs, eta_min=1e-5)

    N_colloc = 2500
    N_intf = 300
    lambda_bc = 10.0

    t0 = time.time()

    for ep in range(1, N_epochs + 1):
        for m in models.values():
            m.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        if is_steady:
            snap_idx = train_snaps[0]
            t_n = None
            bbox = beam_bbox_cache[snap_idx]
        else:
            # Random snapshot for this epoch
            snap_idx = rng.choice(train_snaps)
            t_n = float(times_n[snap_idx])
            bbox = beam_bbox_cache[snap_idx]

        # --- PDE + BC + Data per subdomain ---
        for k, sd in subdomains.items():
            # Collocation points
            xy_c = torch.tensor(sample_subdomain(sd, N_colloc, rng, snap_idx, bbox),
                                dtype=torch.float32).to(device)
            loss_pde = compute_pde_loss(models[k], xy_c, t_n)
            total_loss = total_loss + loss_pde

            # Boundary conditions (from precomputed cache)
            if is_steady:
                xy_bc, u_bc, v_bc = bc_data[k]
            else:
                xy_bc, u_bc, v_bc = bc_cache[snap_idx][k]

            if xy_bc is not None and len(xy_bc) > 0:
                inp_bc = make_input(xy_bc, t_n)
                out_bc = models[k](inp_bc)
                loss_bc = (torch.mean((out_bc[:, 0:1] - u_bc)**2) +
                           torch.mean((out_bc[:, 1:2] - v_bc)**2))
                total_loss = total_loss + lambda_bc * loss_bc

            # Sparse data
            if has_data and sparse_per_sd[k] is not None:
                sd_d = sparse_per_sd[k]
                out_d = models[k](sd_d['inp'])
                loss_d = (torch.mean((out_d[:, 0:1] - sd_d['u'])**2) +
                          torch.mean((out_d[:, 1:2] - sd_d['v'])**2) +
                          torch.mean((out_d[:, 2:3] - sd_d['p'])**2))
                total_loss = total_loss + lambda_data * loss_d

        # --- Fluid-Fluid Interface Losses ---
        for (i, j, itype) in interfaces:
            sd_i, sd_j = subdomains[i], subdomains[j]
            xy_if = sample_interface(sd_i, sd_j, N_intf, rng, snap_idx, bbox)
            if len(xy_if) == 0:
                continue
            xy_if_t = torch.tensor(xy_if, dtype=torch.float32).to(device)
            xy_if_t.requires_grad_(True)

            inp_if = make_input(xy_if_t, t_n)
            out_i = models[i](inp_if)
            out_j = models[j](inp_if)

            # Value matching
            loss_val = (torch.mean((out_i[:, 0] - out_j[:, 0])**2) +
                        torch.mean((out_i[:, 1] - out_j[:, 1])**2) +
                        torch.mean((out_i[:, 2] - out_j[:, 2])**2))

            # Normal direction
            x_lo_ov = max(sd_i['x_lo'], sd_j['x_lo'])
            x_hi_ov = min(sd_i['x_hi'], sd_j['x_hi'])
            y_lo_ov = max(sd_i['y_lo'], sd_j['y_lo'])
            y_hi_ov = min(sd_i['y_hi'], sd_j['y_hi'])
            normal = (torch.tensor([[1.0, 0.0]], device=device)
                      if (x_hi_ov - x_lo_ov) < (y_hi_ov - y_lo_ov)
                      else torch.tensor([[0.0, 1.0]], device=device))

            # Gradient matching (all 3 fields)
            u_i, v_i, p_i = out_i[:, 0:1], out_i[:, 1:2], out_i[:, 2:3]
            grad_u_i = torch.autograd.grad(u_i, xy_if_t, torch.ones_like(u_i),
                                            create_graph=True)[0]
            grad_v_i = torch.autograd.grad(v_i, xy_if_t, torch.ones_like(v_i),
                                            create_graph=True)[0]
            grad_p_i = torch.autograd.grad(p_i, xy_if_t, torch.ones_like(p_i),
                                            create_graph=True)[0]
            dudn_i = torch.sum(grad_u_i * normal, dim=1, keepdim=True)
            dvdn_i = torch.sum(grad_v_i * normal, dim=1, keepdim=True)
            dpdn_i = torch.sum(grad_p_i * normal, dim=1, keepdim=True)

            xy_if_t2 = xy_if_t.detach().clone().requires_grad_(True)
            inp_if2 = make_input(xy_if_t2, t_n)
            out_j2 = models[j](inp_if2)
            u_j, v_j, p_j = out_j2[:, 0:1], out_j2[:, 1:2], out_j2[:, 2:3]
            grad_u_j = torch.autograd.grad(u_j, xy_if_t2, torch.ones_like(u_j),
                                            create_graph=True)[0]
            grad_v_j = torch.autograd.grad(v_j, xy_if_t2, torch.ones_like(v_j),
                                            create_graph=True)[0]
            grad_p_j = torch.autograd.grad(p_j, xy_if_t2, torch.ones_like(p_j),
                                            create_graph=True)[0]
            dudn_j = torch.sum(grad_u_j * normal, dim=1, keepdim=True)
            dvdn_j = torch.sum(grad_v_j * normal, dim=1, keepdim=True)
            dpdn_j = torch.sum(grad_p_j * normal, dim=1, keepdim=True)

            loss_grad = (torch.mean((dudn_i - dudn_j)**2) +
                         torch.mean((dvdn_i - dvdn_j)**2) +
                         torch.mean((dpdn_i - dpdn_j)**2))

            total_loss = total_loss + lambda_intf * (loss_val + loss_grad)

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if ep % 10000 == 0 or ep == 1:
            elapsed = time.time() - t0
            print(f"    Ep {ep:6d}/{N_epochs} | L:{total_loss.item():.4e} | "
                  f"{elapsed:.0f}s")

    t_train = time.time() - t0

    # ----- EVALUATION -----
    for m in models.values():
        m.eval()

    if is_steady:
        snap = train_snaps[0]
        u_true = vel_n[snap, :, 0]
        v_true = vel_n[snap, :, 1]
        p_true = pres_n[snap]

        with torch.no_grad():
            u_pred = np.zeros(N_nodes)
            v_pred = np.zeros(N_nodes)
            p_pred = np.zeros(N_nodes)
            count = np.zeros(N_nodes)

            for k, sd in subdomains.items():
                mask = (point_in_subdomain(coords_n[:, 0], coords_n[:, 1], sd) &
                        ~inside_obstacle_n(coords_n[:, 0], coords_n[:, 1], snap, 0.0) &
                        node_fluid_only)
                if not mask.any():
                    continue
                idx = np.where(mask)[0]
                xy_t = torch.tensor(coords_n[idx], dtype=torch.float32).to(device)
                inp = make_input(xy_t)
                out = models[k](inp).cpu().numpy()
                u_pred[idx] += out[:, 0]
                v_pred[idx] += out[:, 1]
                p_pred[idx] += out[:, 2]
                count[idx] += 1

            valid = (count > 0) & node_fluid_only
            u_pred[valid] /= count[valid]
            v_pred[valid] /= count[valid]
            p_pred[valid] /= count[valid]

        # Errors on fluid-only nodes
        u_err = (np.sqrt(np.mean((u_pred[valid] - u_true[valid])**2)) /
                 max(np.sqrt(np.mean(u_true[valid]**2)), 1e-10) * 100)
        v_err = (np.sqrt(np.mean((v_pred[valid] - v_true[valid])**2)) /
                 max(np.sqrt(np.mean(v_true[valid]**2)), 1e-10) * 100)
        p_err = (np.sqrt(np.mean((p_pred[valid] - p_true[valid])**2)) /
                 max(np.sqrt(np.mean(p_true[valid]**2)), 1e-10) * 100)

        # Error specifically at FSI interface nodes
        fsi_mask = valid & node_is_fsi
        if fsi_mask.any():
            u_fsi_err = (np.sqrt(np.mean((u_pred[fsi_mask] - u_true[fsi_mask])**2)) /
                         max(np.sqrt(np.mean(u_true[fsi_mask]**2)), 1e-10) * 100)
            v_fsi_err = (np.sqrt(np.mean((v_pred[fsi_mask] - v_true[fsi_mask])**2)) /
                         max(np.sqrt(np.mean(v_true[fsi_mask]**2)), 1e-10) * 100)
            p_fsi_err = (np.sqrt(np.mean((p_pred[fsi_mask] - p_true[fsi_mask])**2)) /
                         max(np.sqrt(np.mean(p_true[fsi_mask]**2)), 1e-10) * 100)
        else:
            u_fsi_err = v_fsi_err = p_fsi_err = float('nan')

        # Drag and lift approximation (from pressure on cylinder+beam)
        # TODO: implement force integration

        return {
            'u_err': u_err, 'v_err': v_err, 'p_err': p_err,
            'u_fsi_err': u_fsi_err, 'v_fsi_err': v_fsi_err,
            'p_fsi_err': p_fsi_err,
            't_train': t_train, 'models': models,
        }

    else:
        # Unsteady: evaluate on a subset of snapshots
        eval_snaps = train_snaps[::max(1, len(train_snaps)//10)]  # ~10 snapshots
        u_errs_t, v_errs_t, p_errs_t = [], [], []
        u_fsi_errs_t, v_fsi_errs_t, p_fsi_errs_t = [], [], []

        with torch.no_grad():
            for snap in eval_snaps:
                t_n_eval = float(times_n[snap])
                u_true = vel_n[snap, :, 0]
                v_true = vel_n[snap, :, 1]
                p_true = pres_n[snap]

                u_pred = np.zeros(N_nodes)
                v_pred = np.zeros(N_nodes)
                p_pred = np.zeros(N_nodes)
                count = np.zeros(N_nodes)

                for k, sd in subdomains.items():
                    mask = (point_in_subdomain(coords_n[:, 0], coords_n[:, 1], sd) &
                            ~inside_obstacle_n(coords_n[:, 0], coords_n[:, 1], snap, 0.0) &
                            node_fluid_only)
                    if not mask.any():
                        continue
                    idx = np.where(mask)[0]
                    xy_t = torch.tensor(coords_n[idx], dtype=torch.float32).to(device)
                    inp = make_input(xy_t, t_n_eval)
                    out = models[k](inp).cpu().numpy()
                    u_pred[idx] += out[:, 0]
                    v_pred[idx] += out[:, 1]
                    p_pred[idx] += out[:, 2]
                    count[idx] += 1

                valid = (count > 0) & node_fluid_only
                u_pred[valid] /= count[valid]
                v_pred[valid] /= count[valid]
                p_pred[valid] /= count[valid]

                ue = (np.sqrt(np.mean((u_pred[valid] - u_true[valid])**2)) /
                      max(np.sqrt(np.mean(u_true[valid]**2)), 1e-10) * 100)
                ve = (np.sqrt(np.mean((v_pred[valid] - v_true[valid])**2)) /
                      max(np.sqrt(np.mean(v_true[valid]**2)), 1e-10) * 100)
                pe = (np.sqrt(np.mean((p_pred[valid] - p_true[valid])**2)) /
                      max(np.sqrt(np.mean(p_true[valid]**2)), 1e-10) * 100)
                u_errs_t.append(ue); v_errs_t.append(ve); p_errs_t.append(pe)

                # FSI interface error
                fsi_mask = valid & node_is_fsi
                if fsi_mask.any():
                    u_fsi_errs_t.append(
                        np.sqrt(np.mean((u_pred[fsi_mask] - u_true[fsi_mask])**2)) /
                        max(np.sqrt(np.mean(u_true[fsi_mask]**2)), 1e-10) * 100)

        return {
            'u_err': np.mean(u_errs_t), 'v_err': np.mean(v_errs_t),
            'p_err': np.mean(p_errs_t),
            'u_fsi_err': np.mean(u_fsi_errs_t) if u_fsi_errs_t else float('nan'),
            'v_fsi_err': float('nan'), 'p_fsi_err': float('nan'),
            't_train': t_train, 'models': models,
        }


# ============================================================================
# 13. GENERATE SPARSE DATA
# ============================================================================
print(f"\nGenerating sparse data (N={n_sparse}, mode={placement_mode}, frac={interface_frac})...")
data_rng = np.random.RandomState(999)
if n_sparse > 0:
    sparse_data = generate_sparse_data(n_sparse, data_rng, mode=placement_mode, frac=interface_frac)
    n_fsi_pts = int(round(interface_frac * n_sparse)) if placement_mode == 'interface' else 0
    n_rand_pts = n_sparse - n_fsi_pts
    print(f"  Generated {len(sparse_data['xy'])} points "
          f"({n_rand_pts} random + {n_fsi_pts} FSI interface)")
else:
    sparse_data = None
    print("  No sparse data (XPINN baseline)")


# ============================================================================
# 14. RUN EXPERIMENTS
# ============================================================================
all_results = []
t_global = time.time()

for trial, seed in enumerate(seeds):
    print(f"\n{'='*60}")
    print(f"  Trial {trial+1}/{N_trials} (seed={seed})")
    print(f"{'='*60}")

    result = run_trial(seed, sparse_data=sparse_data, N_epochs=N_epochs)
    all_results.append(result)

    print(f"\n  Global errors:  u={result['u_err']:.2f}%, "
          f"v={result['v_err']:.2f}%, p={result['p_err']:.2f}%")
    print(f"  FSI interface:  u={result['u_fsi_err']:.2f}%")
    print(f"  Training time:  {result['t_train']:.0f}s")

    elapsed = time.time() - t_global
    remaining = elapsed / (trial + 1) * (N_trials - trial - 1)
    print(f"  [{elapsed/60:.0f} min elapsed, ~{remaining/60:.0f} min remaining]")

t_total = time.time() - t_global


# ============================================================================
# 15. SUMMARY
# ============================================================================
u_errs = [r['u_err'] for r in all_results]
v_errs = [r['v_err'] for r in all_results]
p_errs = [r['p_err'] for r in all_results]
u_fsi_errs = [r['u_fsi_err'] for r in all_results]

config_name = f"SPHINX-FSI N={n_sparse}" if n_sparse > 0 else "XPINN-FSI Baseline"

print(f"\n{'='*80}")
print(f"  {config_name} — {case.upper()} ({N_trials} trials)")
print(f"{'='*80}")
print(f"  Global errors:")
print(f"    u: {np.mean(u_errs):.2f} ± {np.std(u_errs):.2f}%")
print(f"    v: {np.mean(v_errs):.2f} ± {np.std(v_errs):.2f}%")
print(f"    p: {np.mean(p_errs):.2f} ± {np.std(p_errs):.2f}%")
print(f"  FSI interface errors:")
print(f"    u: {np.mean(u_fsi_errs):.2f} ± {np.std(u_fsi_errs):.2f}%")
print(f"\n  Total time: {t_total:.0f}s ({t_total/60:.1f} min, "
      f"{t_total/3600:.1f} hrs)")
print(f"{'='*80}")


# ============================================================================
# 16. SAVE RESULTS
# ============================================================================
save_dict = {
    'case': case,
    'n_sparse': n_sparse,
    'placement': placement_mode,
    'interface_frac': interface_frac,
    'u_errs': u_errs, 'v_errs': v_errs, 'p_errs': p_errs,
    'u_fsi_errs': u_fsi_errs,
    'u_mean': np.mean(u_errs), 'u_std': np.std(u_errs),
    'v_mean': np.mean(v_errs), 'v_std': np.std(v_errs),
    'p_mean': np.mean(p_errs), 'p_std': np.std(p_errs),
    'u_fsi_mean': np.mean(u_fsi_errs), 'u_fsi_std': np.std(u_fsi_errs),
}

# Sweep mode: save metrics.npz to output_dir (consistent with cylinder/NACA)
if output_dir is not None:
    os.makedirs(output_dir, exist_ok=True)
    sweep_dict = {
        'u_err': float(np.mean(u_errs)),
        'v_err': float(np.mean(v_errs)),
        'p_err': float(np.mean(p_errs)),
        'u_fsi_err': float(np.mean(u_fsi_errs)),
        'case_tag': case,
        'interface_frac': interface_frac,
        'seed': seeds[0],
        'n_points': n_sparse,
        'n_epochs': N_epochs,
        'train_time': float(np.mean([r['t_train'] for r in all_results])),
    }
    # Include predicted fields from the (only) trial for post-hoc analysis
    if len(all_results) > 0 and 'U_pred' in all_results[0]:
        sweep_dict['U_pred'] = all_results[0]['U_pred']
        sweep_dict['V_pred'] = all_results[0]['V_pred']
        sweep_dict['P_pred'] = all_results[0]['P_pred']
    sweep_path = os.path.join(output_dir, 'metrics.npz')
    np.savez(sweep_path, **sweep_dict)
    print(f"\n  Saved sweep metrics: {sweep_path}")

# Original save (kept for legacy / non-sweep usage)
if N_trials == 1:
    outfile = f'sphinx_fsi_{case}_N{n_sparse}_{placement_mode}_s{seeds[0]}_results.npz'
else:
    outfile = f'sphinx_fsi_{case}_N{n_sparse}_{placement_mode}_results.npz'
np.savez(outfile, **save_dict)
print(f"  Saved: {outfile}")
