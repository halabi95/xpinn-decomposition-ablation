"""
==============================================================================
SPHINX Cylinder Re=40 — SENSITIVITY SWEEP VERSION
==============================================================================
Modified from sphinx_cylinder_xpinn.py to run ONE training trial per
invocation, with interface fraction and seed passed via CLI. Designed for
the sensitivity study requested by the Computers & Fluids editor.

Usage:
  python sphinx_sweep_cylinder.py \
      --interface-frac 0.3 \
      --seed 42 \
      --n-points 200 \
      --output-dir results/frac_0.3/seed_42

What changed vs sphinx_cylinder_xpinn.py:
  - Replaced --mode CLI with: --interface-frac, --seed, --n-points,
    --output-dir, --n-epochs
  - Removed multi-N / multi-config experiment loop (one trial per call now)
  - Sparse data split uses --interface-frac (was hard-coded 0.7 random / 0.3 intf)
  - Edge case: frac=0.0 → pure random; frac=1.0 → pure interface
  - Results saved to <output-dir>/metrics.npz (model .pth files NOT saved —
    18 model sets × 4 subdomains would be a lot of disk; we save predicted
    fields for post-hoc analysis instead)
  - Removed final results table / model saving
==============================================================================
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib
matplotlib.use('Agg')
from scipy.interpolate import RegularGridInterpolator

# ============================================================================
# CLI ARGS
# ============================================================================
parser = argparse.ArgumentParser(description='SPHINX cylinder sensitivity sweep — single trial')
parser.add_argument('--interface-frac', type=float, required=True,
                    help='Fraction of sparse points in interface overlap regions (0.0-1.0)')
parser.add_argument('--seed', type=int, required=True,
                    help='Random seed for this trial')
parser.add_argument('--n-points', type=int, default=200,
                    help='Total sparse data points (default: 200)')
parser.add_argument('--output-dir', type=str, required=True,
                    help='Directory to save results')
parser.add_argument('--n-epochs', type=int, default=100000,
                    help='Training epochs (default: 100000 — matches production)')
args = parser.parse_args()

if not (0.0 <= args.interface_frac <= 1.0):
    raise ValueError(f"interface_frac must be in [0, 1], got {args.interface_frac}")

print("="*70)
print(f"  SPHINX Cylinder Sensitivity Sweep — Single Trial")
print(f"  interface_frac = {args.interface_frac}")
print(f"  seed           = {args.seed}")
print(f"  n_points       = {args.n_points}")
print(f"  output_dir     = {args.output_dir}")
print(f"  n_epochs       = {args.n_epochs}")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Load reference data
ref_file = 'cylinder_re40_reference.npz'
if not os.path.exists(ref_file):
    print(f"  ERROR: {ref_file} not found!"); sys.exit(1)

data = np.load(ref_file)
Re, nu, U_inf = float(data['Re']), float(data['nu']), float(data['U_inf'])
R, cx, cy = float(data['R']), float(data['cx']), float(data['cy'])
x_min_d, x_max_d = float(data['x_extract_min']), float(data['x_extract_max'])
y_min_d, y_max_d = float(data['y_extract_min']), float(data['y_extract_max'])
X_ref, Y_ref = data['X'], data['Y']
U_ref, V_ref, P_ref = data['U'], data['V'], data['P']
x_grid, y_grid = data['x_grid'], data['y_grid']

print(f"  Re={Re}, U_inf={U_inf}, R={R}, center=({cx},{cy})")
print(f"  Domain: [{x_min_d},{x_max_d}] x [{y_min_d},{y_max_d}]")

interp_u = RegularGridInterpolator((y_grid, x_grid), U_ref, method='linear',
                                    bounds_error=False, fill_value=np.nan)
interp_v = RegularGridInterpolator((y_grid, x_grid), V_ref, method='linear',
                                    bounds_error=False, fill_value=np.nan)
interp_p = RegularGridInterpolator((y_grid, x_grid), P_ref, method='linear',
                                    bounds_error=False, fill_value=np.nan)


# ============================================================================
# 1. GEOMETRY
# ============================================================================
def inside_cylinder(x, y, margin=0.02):
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    return r < R + margin


# ============================================================================
# 2. DOMAIN DECOMPOSITION (4 subdomains)
# ============================================================================
x_split = cx
y_split = cy
overlap = 0.5

subdomains = {
    0: {'name': 'Upstream',  'x_lo': x_min_d, 'x_hi': x_split + overlap,
        'y_lo': y_min_d, 'y_hi': y_max_d},
    1: {'name': 'Top-Wake',  'x_lo': x_split - overlap, 'x_hi': x_max_d,
        'y_lo': y_split - overlap, 'y_hi': y_max_d},
    2: {'name': 'Bot-Wake',  'x_lo': x_split - overlap, 'x_hi': x_max_d,
        'y_lo': y_min_d, 'y_hi': y_split + overlap},
    3: {'name': 'Near-Body', 'x_lo': cx - 2.0, 'x_hi': cx + 3.0,
        'y_lo': cy - 2.0, 'y_hi': cy + 2.0},
}
interfaces = [
    (0, 1, 'vertical'), (0, 2, 'vertical'), (1, 2, 'horizontal'),
    (0, 3, 'overlap'), (1, 3, 'overlap'), (2, 3, 'overlap'),
]

print(f"\n  Domain decomposition: {len(subdomains)} subdomains")
for k, sd in subdomains.items():
    print(f"    Ω{k} ({sd['name']}): [{sd['x_lo']:.1f},{sd['x_hi']:.1f}] "
          f"× [{sd['y_lo']:.1f},{sd['y_hi']:.1f}]")


# ============================================================================
# 3. NETWORK (8×128 — matching original production)
# ============================================================================
class SubPINN(nn.Module):
    def __init__(self, n_layers=8, n_neurons=128):
        super().__init__()
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 3))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, xy):
        return self.net(xy)

_tmp = SubPINN()
_n_params = sum(p.numel() for p in _tmp.parameters())
print(f"  Network: 8×128, {_n_params:,} params/subdomain, {_n_params*4:,} total")
del _tmp


# ============================================================================
# 4. SAMPLING
# ============================================================================
def sample_subdomain(sd, n, rng):
    pts = []
    while len(pts) < n:
        x = rng.uniform(sd['x_lo'], sd['x_hi'], n * 3)
        y = rng.uniform(sd['y_lo'], sd['y_hi'], n * 3)
        mask = ~inside_cylinder(x, y, margin=0.02)
        gx, gy = x[mask], y[mask]
        for i in range(min(len(gx), n - len(pts))):
            pts.append([gx[i], gy[i]])
    return np.array(pts[:n])


def sample_interface(sd_i, sd_j, n, rng):
    x_lo = max(sd_i['x_lo'], sd_j['x_lo'])
    x_hi = min(sd_i['x_hi'], sd_j['x_hi'])
    y_lo = max(sd_i['y_lo'], sd_j['y_lo'])
    y_hi = min(sd_i['y_hi'], sd_j['y_hi'])
    if x_lo >= x_hi or y_lo >= y_hi:
        return np.zeros((0, 2))
    pts = []
    while len(pts) < n:
        x = rng.uniform(x_lo, x_hi, n * 3)
        y = rng.uniform(y_lo, y_hi, n * 3)
        mask = ~inside_cylinder(x, y, margin=0.02)
        gx, gy = x[mask], y[mask]
        for i in range(min(len(gx), n - len(pts))):
            pts.append([gx[i], gy[i]])
    return np.array(pts[:n])


def point_in_subdomain(x, y, sd):
    return ((x >= sd['x_lo']) & (x <= sd['x_hi']) &
            (y >= sd['y_lo']) & (y <= sd['y_hi']))


def get_bc_points_subdomain(sd):
    xy, u_vals, v_vals = [], [], []
    theta = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    offset = 0.003
    for t in theta:
        px = cx + (R + offset) * np.cos(t)
        py = cy + (R + offset) * np.sin(t)
        if sd['x_lo'] <= px <= sd['x_hi'] and sd['y_lo'] <= py <= sd['y_hi']:
            xy.append([px, py]); u_vals.append(0.0); v_vals.append(0.0)
    if sd['x_lo'] <= x_min_d + 0.01:
        for yi in np.linspace(max(sd['y_lo'], y_min_d),
                               min(sd['y_hi'], y_max_d), 150):
            xy.append([x_min_d, yi]); u_vals.append(U_inf); v_vals.append(0.0)
    if sd['y_hi'] >= y_max_d - 0.01:
        for xi in np.linspace(max(sd['x_lo'], x_min_d),
                               min(sd['x_hi'], x_max_d), 100):
            xy.append([xi, y_max_d]); u_vals.append(U_inf); v_vals.append(0.0)
    if sd['y_lo'] <= y_min_d + 0.01:
        for xi in np.linspace(max(sd['x_lo'], x_min_d),
                               min(sd['x_hi'], x_max_d), 100):
            xy.append([xi, y_min_d]); u_vals.append(U_inf); v_vals.append(0.0)
    if len(xy) == 0:
        return None, None, None
    return (torch.tensor(np.array(xy), dtype=torch.float32).to(device),
            torch.tensor(np.array(u_vals), dtype=torch.float32).to(device).unsqueeze(1),
            torch.tensor(np.array(v_vals), dtype=torch.float32).to(device).unsqueeze(1))


def sample_sparse_random(n_pts, rng):
    if n_pts <= 0:
        return np.zeros((0, 2))
    pts = []
    while len(pts) < n_pts:
        x = rng.uniform(x_min_d + 0.5, x_max_d - 0.5, n_pts * 3)
        y = rng.uniform(y_min_d + 0.5, y_max_d - 0.5, n_pts * 3)
        mask = ~inside_cylinder(x, y, margin=0.1)
        gx, gy = x[mask], y[mask]
        for i in range(min(len(gx), n_pts - len(pts))):
            pts.append([gx[i], gy[i]])
    return np.array(pts[:n_pts])


def sample_interface_data(n_intf, rng):
    """Sample sparse data at subdomain interface regions.
    Will gather at least n_intf points across the 6 interfaces; caller slices."""
    if n_intf <= 0:
        return np.zeros((0, 2))
    pts = []
    n_per = max(2, (n_intf + len(interfaces) - 1) // len(interfaces))  # ceil div
    for (i, j, itype) in interfaces:
        sd_i, sd_j = subdomains[i], subdomains[j]
        x_lo = max(sd_i['x_lo'], sd_j['x_lo'])
        x_hi = min(sd_i['x_hi'], sd_j['x_hi'])
        y_lo = max(sd_i['y_lo'], sd_j['y_lo'])
        y_hi = min(sd_i['y_hi'], sd_j['y_hi'])
        if x_lo >= x_hi or y_lo >= y_hi:
            continue
        x = rng.uniform(x_lo, x_hi, n_per * 3)
        y = rng.uniform(y_lo, y_hi, n_per * 3)
        mask = ~inside_cylinder(x, y, margin=0.1)
        gx, gy = x[mask], y[mask]
        for k in range(min(len(gx), n_per)):
            pts.append([gx[k], gy[k]])
    return np.array(pts) if pts else np.zeros((0, 2))


# ============================================================================
# 5. PDE RESIDUAL (Incompressible Navier-Stokes, laminar)
# ============================================================================
def compute_pde_loss(model, xy_c):
    xy_c.requires_grad_(True)
    out = model(xy_c)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    gu = torch.autograd.grad(u, xy_c, torch.ones_like(u), create_graph=True)[0]
    gv = torch.autograd.grad(v, xy_c, torch.ones_like(v), create_graph=True)[0]
    gp = torch.autograd.grad(p, xy_c, torch.ones_like(p), create_graph=True)[0]
    du_dx, du_dy = gu[:, 0:1], gu[:, 1:2]
    dv_dx, dv_dy = gv[:, 0:1], gv[:, 1:2]
    dp_dx, dp_dy = gp[:, 0:1], gp[:, 1:2]
    d2u = torch.autograd.grad(du_dx, xy_c, torch.ones_like(du_dx), create_graph=True)[0][:, 0:1] + \
          torch.autograd.grad(du_dy, xy_c, torch.ones_like(du_dy), create_graph=True)[0][:, 1:2]
    d2v = torch.autograd.grad(dv_dx, xy_c, torch.ones_like(dv_dx), create_graph=True)[0][:, 0:1] + \
          torch.autograd.grad(dv_dy, xy_c, torch.ones_like(dv_dy), create_graph=True)[0][:, 1:2]
    rc = du_dx + dv_dy
    rx = u * du_dx + v * du_dy + dp_dx - nu * d2u
    ry = u * dv_dx + v * dv_dy + dp_dy - nu * d2v
    return torch.mean(rc**2) + torch.mean(rx**2) + torch.mean(ry**2)


# ============================================================================
# 6. XPINN TRAINING
# ============================================================================
def run_xpinn_trial(seed, sparse_xy=None, sparse_u=None, sparse_v=None, sparse_p=None,
                    lambda_data=10.0, lambda_intf=10.0, N_epochs=100000):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    rng = np.random.RandomState(seed)

    models = {k: SubPINN().to(device) for k in subdomains}
    bc_data = {k: get_bc_points_subdomain(sd) for k, sd in subdomains.items()}

    sparse_per_sd = {k: {'xy': None} for k in subdomains}
    has_data = sparse_xy is not None and len(sparse_xy) > 0
    if has_data:
        for k, sd in subdomains.items():
            mask = point_in_subdomain(sparse_xy[:, 0], sparse_xy[:, 1], sd)
            if mask.any():
                sparse_per_sd[k] = {
                    'xy': torch.tensor(sparse_xy[mask], dtype=torch.float32).to(device),
                    'u': torch.tensor(sparse_u[mask], dtype=torch.float32).to(device).unsqueeze(1),
                    'v': torch.tensor(sparse_v[mask], dtype=torch.float32).to(device).unsqueeze(1),
                    'p': torch.tensor(sparse_p[mask], dtype=torch.float32).to(device).unsqueeze(1),
                }

    all_params = []
    for m in models.values():
        all_params += list(m.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_epochs, eta_min=1e-5)

    N_colloc = 2500
    N_intf = 500
    lambda_bc = 10.0

    t0 = time.time()
    final_loss = 0.0
    for ep in range(1, N_epochs + 1):
        for m in models.values():
            m.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        for k, sd in subdomains.items():
            xy_c = torch.tensor(sample_subdomain(sd, N_colloc, rng),
                                dtype=torch.float32).to(device)
            loss_pde = compute_pde_loss(models[k], xy_c)
            total_loss = total_loss + loss_pde

            xy_bc, u_bc, v_bc = bc_data[k]
            if xy_bc is not None and len(xy_bc) > 0:
                out_bc = models[k](xy_bc)
                loss_bc = (torch.mean((out_bc[:, 0:1] - u_bc)**2) +
                           torch.mean((out_bc[:, 1:2] - v_bc)**2))
                total_loss = total_loss + lambda_bc * loss_bc

            if has_data and sparse_per_sd[k]['xy'] is not None:
                sd_d = sparse_per_sd[k]
                out_d = models[k](sd_d['xy'])
                loss_data = (torch.mean((out_d[:, 0:1] - sd_d['u'])**2) +
                             torch.mean((out_d[:, 1:2] - sd_d['v'])**2) +
                             torch.mean((out_d[:, 2:3] - sd_d['p'])**2))
                total_loss = total_loss + lambda_data * loss_data

        for (i, j, itype) in interfaces:
            sd_i, sd_j = subdomains[i], subdomains[j]
            xy_if = sample_interface(sd_i, sd_j, N_intf, rng)
            if len(xy_if) == 0:
                continue
            xy_if_t = torch.tensor(xy_if, dtype=torch.float32).to(device)
            xy_if_t.requires_grad_(True)
            out_i = models[i](xy_if_t)
            out_j = models[j](xy_if_t)

            loss_val = (torch.mean((out_i[:, 0] - out_j[:, 0])**2) +
                        torch.mean((out_i[:, 1] - out_j[:, 1])**2) +
                        torch.mean((out_i[:, 2] - out_j[:, 2])**2))

            x_lo_ov = max(sd_i['x_lo'], sd_j['x_lo'])
            x_hi_ov = min(sd_i['x_hi'], sd_j['x_hi'])
            y_lo_ov = max(sd_i['y_lo'], sd_j['y_lo'])
            y_hi_ov = min(sd_i['y_hi'], sd_j['y_hi'])
            normal = (torch.tensor([[1.0, 0.0]], device=device)
                      if (x_hi_ov - x_lo_ov) < (y_hi_ov - y_lo_ov)
                      else torch.tensor([[0.0, 1.0]], device=device))

            u_i, v_i, p_i = out_i[:, 0:1], out_i[:, 1:2], out_i[:, 2:3]
            grad_u_i = torch.autograd.grad(u_i, xy_if_t, torch.ones_like(u_i), create_graph=True)[0]
            grad_v_i = torch.autograd.grad(v_i, xy_if_t, torch.ones_like(v_i), create_graph=True)[0]
            grad_p_i = torch.autograd.grad(p_i, xy_if_t, torch.ones_like(p_i), create_graph=True)[0]
            dudn_i = torch.sum(grad_u_i * normal, dim=1, keepdim=True)
            dvdn_i = torch.sum(grad_v_i * normal, dim=1, keepdim=True)
            dpdn_i = torch.sum(grad_p_i * normal, dim=1, keepdim=True)

            xy_if_t2 = xy_if_t.detach().clone().requires_grad_(True)
            out_j2 = models[j](xy_if_t2)
            u_j, v_j, p_j = out_j2[:, 0:1], out_j2[:, 1:2], out_j2[:, 2:3]
            grad_u_j = torch.autograd.grad(u_j, xy_if_t2, torch.ones_like(u_j), create_graph=True)[0]
            grad_v_j = torch.autograd.grad(v_j, xy_if_t2, torch.ones_like(v_j), create_graph=True)[0]
            grad_p_j = torch.autograd.grad(p_j, xy_if_t2, torch.ones_like(p_j), create_graph=True)[0]
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
        final_loss = total_loss.item()

        if ep % 20000 == 0 or ep == 1:
            elapsed = time.time() - t0
            print(f"    Ep {ep:6d}/{N_epochs} | L:{final_loss:.4e} | {elapsed:.0f}s")

    t_train = time.time() - t0

    # Evaluate
    for m in models.values():
        m.eval()
    x_flat, y_flat = X_ref.flatten(), Y_ref.flatten()
    with torch.no_grad():
        u_accum = np.zeros_like(x_flat)
        v_accum = np.zeros_like(x_flat)
        p_accum = np.zeros_like(x_flat)
        count = np.zeros_like(x_flat)
        for k, sd in subdomains.items():
            mask = (point_in_subdomain(x_flat, y_flat, sd) &
                    ~inside_cylinder(x_flat, y_flat, margin=0.005))
            if not mask.any():
                continue
            for s in range(0, mask.sum(), 10000):
                e = min(s + 10000, mask.sum())
                idx = np.where(mask)[0][s:e]
                xy = np.stack([x_flat[idx], y_flat[idx]], axis=1)
                o = models[k](torch.tensor(xy, dtype=torch.float32).to(device)).cpu().numpy()
                u_accum[idx] += o[:, 0]
                v_accum[idx] += o[:, 1]
                p_accum[idx] += o[:, 2]
                count[idx] += 1

    vc = count > 0
    u_pred = np.full_like(x_flat, np.nan)
    v_pred = np.full_like(x_flat, np.nan)
    p_pred = np.full_like(x_flat, np.nan)
    u_pred[vc] = u_accum[vc] / count[vc]
    v_pred[vc] = v_accum[vc] / count[vc]
    p_pred[vc] = p_accum[vc] / count[vc]
    U_p = u_pred.reshape(X_ref.shape)
    V_p = v_pred.reshape(X_ref.shape)
    P_p = p_pred.reshape(X_ref.shape)

    valid = ~np.isnan(U_ref) & ~np.isnan(U_p)
    ue = np.sqrt(np.nanmean((U_p[valid] - U_ref[valid])**2)) / np.sqrt(np.nanmean(U_ref[valid]**2)) * 100
    ve = np.sqrt(np.nanmean((V_p[valid] - V_ref[valid])**2)) / np.sqrt(np.nanmean(V_ref[valid]**2)) * 100
    pe = np.sqrt(np.nanmean((P_p[valid] - P_ref[valid])**2)) / np.sqrt(np.nanmean(P_ref[valid]**2)) * 100
    return ue, ve, pe, t_train, final_loss, U_p, V_p, P_p


# ============================================================================
# 7. GENERATE SPARSE DATA — single config from CLI
# ============================================================================
print(f"\nGenerating sparse data: N={args.n_points}, interface_frac={args.interface_frac}")
data_rng = np.random.RandomState(999 + args.n_points + int(round(args.interface_frac*1000)))
n_intf = int(round(args.interface_frac * args.n_points))
n_random = args.n_points - n_intf
print(f"  Split: {n_random} random + {n_intf} interface")

xy_rand = sample_sparse_random(n_random, data_rng)
xy_intf_all = sample_interface_data(n_intf, data_rng)
xy_intf = xy_intf_all[:n_intf] if len(xy_intf_all) >= n_intf else xy_intf_all

if len(xy_rand) > 0 and len(xy_intf) > 0:
    xy = np.vstack([xy_rand, xy_intf])
elif len(xy_rand) > 0:
    xy = xy_rand
elif len(xy_intf) > 0:
    xy = xy_intf
else:
    xy = np.zeros((0, 2))

yx = xy[:, ::-1]
u_sp = interp_u(yx); v_sp = interp_v(yx); p_sp = interp_p(yx)
valid = ~np.isnan(u_sp) & ~np.isnan(v_sp) & ~np.isnan(p_sp)
sparse_xy = xy[valid]
sparse_u = u_sp[valid]; sparse_v = v_sp[valid]; sparse_p = p_sp[valid]
print(f"  → {valid.sum()} valid points after cylinder masking "
      f"(target was {args.n_points})")


# ============================================================================
# 8. RUN SINGLE TRIAL
# ============================================================================
print(f"\n{'='*70}")
print(f"  Training: frac={args.interface_frac}, seed={args.seed}, "
      f"N_actual={len(sparse_xy)}")
print(f"{'='*70}")

t0 = time.time()
if len(sparse_xy) > 0:
    ue, ve, pe, tt, loss, Up, Vp, Pp = run_xpinn_trial(
        args.seed, sparse_xy, sparse_u, sparse_v, sparse_p,
        lambda_data=10.0, N_epochs=args.n_epochs)
else:
    ue, ve, pe, tt, loss, Up, Vp, Pp = run_xpinn_trial(
        args.seed, N_epochs=args.n_epochs)
elapsed = time.time() - t0

print(f"\n  Results:")
print(f"    u_error    = {ue:.4f}%")
print(f"    v_error    = {ve:.4f}%")
print(f"    p_error    = {pe:.4f}%")
print(f"    train_time = {tt:.0f}s ({tt/60:.1f} min)")
print(f"    total_time = {elapsed:.0f}s ({elapsed/60:.1f} min)")


# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, 'metrics.npz')
np.savez(output_path,
    u_err=ue, v_err=ve, p_err=pe,
    interface_frac=args.interface_frac,
    seed=args.seed,
    n_points=args.n_points,
    n_intf_target=n_intf,
    n_random_target=n_random,
    n_valid=int(valid.sum()),
    train_time=tt,
    final_loss=loss,
    n_epochs=args.n_epochs,
    U_pred=Up, V_pred=Vp, P_pred=Pp,
    sparse_xy=sparse_xy,
)
print(f"\n  Saved: {output_path}")
print("="*70)
