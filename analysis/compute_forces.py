import numpy as np, os
from scipy.interpolate import RegularGridInterpolator, griddata

SEED=42

def fill_mask(X, Y, F):
    """Fill NaN cells by nearest-valid extrapolation so the field is defined at the wall."""
    m = np.isfinite(F)
    if m.all(): return F
    pts = np.column_stack([X[m], Y[m]])
    out = F.copy()
    out[~m] = griddata(pts, F[m], np.column_stack([X[~m], Y[~m]]), method='nearest')
    return out

def make_interp(X, Y, F):
    return RegularGridInterpolator((Y[:,0], X[0,:]), F, bounds_error=False, fill_value=None)

def forces(X, Y, U, V, P, xs, ys, nx, ny, ds, nu, Lref):
    U,V,P = fill_mask(X,Y,U), fill_mask(X,Y,V), fill_mask(X,Y,P)
    fp, fu, fv = make_interp(X,Y,P), make_interp(X,Y,U), make_interp(X,Y,V)
    pts = np.column_stack([ys, xs])
    p_s = fp(pts)
    h = 0.02
    po  = np.column_stack([ys + h*ny,   xs + h*nx])
    po2 = np.column_stack([ys + 2*h*ny, xs + 2*h*nx])
    tx, ty = -ny, nx
    ut0 = fu(pts)*tx + fv(pts)*ty
    ut1 = fu(po)*tx  + fv(po)*ty
    ut2 = fu(po2)*tx + fv(po2)*ty
    dut = (-3*ut0 + 4*ut1 - ut2)/(2*h)      # 2nd-order one-sided
    tau = nu*dut
    Fx = np.sum((-p_s*nx + tau*tx)*ds)
    Fy = np.sum((-p_s*ny + tau*ty)*ds)
    q  = 0.5*Lref
    return Fx/q, Fy/q, p_s

def cyl_surf(ref, n=400):
    R=float(ref['R']); cx=float(ref['cx']); cy=float(ref['cy'])
    th=np.linspace(0,2*np.pi,n,endpoint=False)
    return (cx+R*np.cos(th), cy+R*np.sin(th), np.cos(th), np.sin(th),
            np.full(n,2*np.pi*R/n), 2*R, th)

def foil_surf(ref):
    xa,ya = np.asarray(ref['x_airfoil']), np.asarray(ref['y_airfoil'])
    if xa[0]!=xa[-1] or ya[0]!=ya[-1]:
        xa=np.append(xa,xa[0]); ya=np.append(ya,ya[0])
    xm=0.5*(xa[:-1]+xa[1:]); ym=0.5*(ya[:-1]+ya[1:])
    dx=xa[1:]-xa[:-1]; dy=ya[1:]-ya[:-1]
    ds=np.hypot(dx,dy); nx,ny = dy/ds, -dx/ds
    cx,cy = xm.mean(), ym.mean()
    s = np.sign((xm-cx)*nx + (ym-cy)*ny)
    return xm,ym,nx*s,ny*s,ds,float(ref['chord']),xm

CASES = {
 'cylinder': ('cases/cylinder_re40/cylinder_re40_reference.npz', cyl_surf,
   {'overlap near-body': f'results/sensitivity_sweep/cylinder/frac_0.1/seed_{SEED}/metrics.npz',
    'overlap quadrant':  f'cases/cylinder_re40/results/A_quadov/frac_0.1/seed_{SEED}/metrics.npz',
    'non-overlap near-body': f'cases/cylinder_re40/results/D_partition/seed_{SEED}/metrics.npz'}),
 'naca_re1k': ('cases/naca0012_re1k/naca0012_re1000_aoa5_reference.npz', foil_surf,
   {'overlap near-body': f'results/sensitivity_sweep/naca_re1k/frac_0.1/seed_{SEED}/metrics.npz',
    'overlap quadrant':  f'cases/naca0012_re1k/results/A_quadov/frac_0.1/seed_{SEED}/metrics.npz',
    'non-overlap near-body': f'cases/naca0012_re1k/results/D_partition/seed_{SEED}/metrics.npz',
    'non-overlap quadrant': f'cases/naca0012_re1k/results/nonoverlap/seed_{SEED}/metrics_nonoverlap.npz'}),
 'naca_re10k': ('cases/naca0012_re10k/naca0012_re10k_aoa5_reference.npz', foil_surf,
   {'overlap near-body': f'results/sensitivity_sweep/naca_re10k/frac_0.1/seed_{SEED}/metrics.npz',
    'overlap quadrant':  f'cases/naca0012_re10k/results/A_quadov/frac_0.1/seed_{SEED}/metrics.npz',
    'non-overlap near-body': f'cases/naca0012_re10k/results/D_partition/seed_{SEED}/metrics.npz',
    'non-overlap quadrant': f'cases/naca0012_re10k/results/nonoverlap/seed_{SEED}/metrics_nonoverlap.npz'}),
}

store={}
for key,(refp,surf,runs) in CASES.items():
    ref=np.load(refp,allow_pickle=True)
    X,Y,nu = ref['X'],ref['Y'],float(ref['nu'])
    xs,ys,nx,ny,ds,Lref,param = surf(ref)
    cd0,cl0,cp0 = forces(X,Y,ref['U'],ref['V'],ref['P'],xs,ys,nx,ny,ds,nu,Lref)
    Cd_s,Cl_s = float(ref['Cd']), float(ref['Cl'])
    ed = abs(cd0-Cd_s)/max(abs(Cd_s),1e-9)*100
    el = abs(cl0-Cl_s)
    print(f"\n=== {key} ===")
    print(f"  stored      Cd={Cd_s:8.4f}  Cl={Cl_s:8.4f}")
    print(f"  integrated  Cd={cd0:8.4f}  Cl={cl0:8.4f}   -> Cd err {ed:.1f}%, Cl abs err {el:.4f}")
    ok = ed < 10
    print(f"  VALIDATION: {'PASS' if ok else 'FAIL - do not use these numbers'}")
    store[key]=dict(param=param, cp_ref=cp0, preds={})
    if key=='cylinder':
        lp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'xpinn_nonoverlap_cylinder_random_results.npz')
        if os.path.exists(lp):
            L=np.load(lp,allow_pickle=True)
            if all(k in L.files for k in ('best_U','best_V','best_P')):
                cd,cl,cp=forces(X,Y,L['best_U'],L['best_V'],L['best_P'],xs,ys,nx,ny,ds,nu,Lref)
                store[key]['preds']['non-overlap quadrant']=cp
                print(f"    {'non-overlap quadrant':24} Cd={cd:8.4f} ({cd-Cd_s:+.4f})   Cl={cl:8.4f} ({cl-Cl_s:+.4f})   [legacy, best seed]")

    for name,path in runs.items():
        if not os.path.exists(path): print(f"    MISSING {name}"); continue
        d=np.load(path,allow_pickle=True)
        if 'U_pred' not in d.files: print(f"    no fields {name}"); continue
        cd,cl,cp = forces(X,Y,d['U_pred'],d['V_pred'],d['P_pred'],xs,ys,nx,ny,ds,nu,Lref)
        store[key]['preds'][name]=cp
        print(f"    {name:24} Cd={cd:8.4f} ({cd-Cd_s:+.4f})   Cl={cl:8.4f} ({cl-Cl_s:+.4f})")

np.savez('surface_cp.npz', **{f'{k}|{n}':v for k,d in store.items() for n,v in
         [('param',d['param']),('ref',d['cp_ref'])]+list(d['preds'].items())})
print('\nwrote surface_cp.npz for the Cp figure')
