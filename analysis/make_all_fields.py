import numpy as np, os, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

SEED = 42
OUT = 'figs_fields'; os.makedirs(OUT, exist_ok=True)

CASES = {
 'cylinder': dict(
    ref='cases/cylinder_re40/cylinder_re40_reference.npz',
    title='Cylinder, Re = 40', xlim=(-2,6), ylim=(-2.5,2.5), xlab='x/D',
    body='circle',
    runs={'overlap near-body':    f'results/sensitivity_sweep/cylinder/frac_0.1/seed_{SEED}/metrics.npz',
          'overlap quadrant':     f'cases/cylinder_re40/results/A_quadov/frac_0.1/seed_{SEED}/metrics.npz',
          'non-overlap near-body':f'cases/cylinder_re40/results/D_partition/seed_{SEED}/metrics.npz',
          'non-overlap quadrant': f'cases/cylinder_re40/results/nonoverlap/seed_{SEED}/metrics.npz'}),
 'naca_re1k': dict(
    ref='cases/naca0012_re1k/naca0012_re1000_aoa5_reference.npz',
    title='NACA 0012, Re = 1,000', xlim=(-0.5,2.0), ylim=(-0.8,0.8), xlab='x/c',
    body='airfoil',
    runs={'overlap near-body':    f'results/sensitivity_sweep/naca_re1k/frac_0.1/seed_{SEED}/metrics.npz',
          'overlap quadrant':     f'cases/naca0012_re1k/results/A_quadov/frac_0.1/seed_{SEED}/metrics.npz',
          'non-overlap near-body':f'cases/naca0012_re1k/results/D_partition/seed_{SEED}/metrics.npz',
          'non-overlap quadrant': f'cases/naca0012_re1k/results/nonoverlap/seed_{SEED}/metrics.npz'}),
 'naca_re10k': dict(
    ref='cases/naca0012_re10k/naca0012_re10k_aoa5_reference.npz',
    title='NACA 0012, Re = 10,000', xlim=(-0.5,2.0), ylim=(-0.8,0.8), xlab='x/c',
    body='airfoil',
    runs={'overlap near-body':    f'results/sensitivity_sweep/naca_re10k/frac_0.1/seed_{SEED}/metrics.npz',
          'overlap quadrant':     f'cases/naca0012_re10k/results/A_quadov/frac_0.1/seed_{SEED}/metrics.npz',
          'non-overlap near-body':f'cases/naca0012_re10k/results/D_partition/seed_{SEED}/metrics.npz',
          'non-overlap quadrant': f'cases/naca0012_re10k/results/nonoverlap/seed_{SEED}/metrics.npz'}),
}

def zc(a): return a - np.nanmean(a)

def draw_body(ax, cfg, ref):
    if cfg['body']=='airfoil':
        ax.fill(ref['x_airfoil'], ref['y_airfoil'], color='0.35', zorder=6)
    else:
        from matplotlib.patches import Circle
        ax.add_patch(Circle((float(ref['cx']),float(ref['cy'])), float(ref['R']),
                            facecolor='0.35', ec='k', lw=0.8, zorder=6))

def build(key, cfg, var):
    ref = np.load(cfg['ref'], allow_pickle=True)
    X, Y = ref['X'], ref['Y']
    F_ref = ref[var.upper()]
    if var=='p': F_ref = zc(F_ref)

    preds={}
    for name, path in cfg['runs'].items():
        if not os.path.exists(path):
            alt = path.replace('metrics.npz', 'metrics_nonoverlap.npz')
            if os.path.exists(alt):
                path = alt
            else:
                print(f'  MISSING: {name}  ({path})'); continue
        d = np.load(path, allow_pickle=True)
        k = f'{var.upper()}_pred'
        if k not in d.files:
            print(f'  no {k}: {name}'); continue
        F = d[k]
        preds[name] = zc(F) if var=='p' else F

    # legacy cylinder nonoverlap-quadrant (May campaign, same settings: 8x128, 100k, seeds 42/123/456)
    if key == 'cylinder':
        lp = '/sqfs2/cmc/1/home/z6b512/sphinx_project/New_Files/step1_cfd/xpinn_nonoverlap_cylinder_random_results.npz'
        if os.path.exists(lp):
            L = np.load(lp, allow_pickle=True)
            k2 = {'u':'best_U','v':'best_V','p':'best_P'}[var]
            if k2 in L.files:
                F = L[k2]
                if F.shape == F_ref.shape:
                    preds['non-overlap quadrant'] = zc(F) if var=='p' else F
                    print(f'  + legacy nonoverlap-quadrant ({k2}, {F.shape})')
                else:
                    print(f'  legacy shape mismatch {F.shape} vs {F_ref.shape}')

    if not preds:
        print(f'  nothing to plot for {key}/{var}'); return

    n = len(preds)
    fig, axes = plt.subplots(2, n+1, figsize=(4.3*(n+1), 6.6))
    vmax = np.nanpercentile(np.abs(F_ref), 99)
    cmap = 'RdBu_r' if var=='p' else 'viridis'
    if var=='p':
        kw = dict(vmin=-vmax, vmax=vmax)
    else:
        lo = min([np.nanmin(F_ref)] + [np.nanmin(F) for F in preds.values()])
        hi = max([np.nanmax(F_ref)] + [np.nanmax(F) for F in preds.values()])
        kw = dict(vmin=lo, vmax=hi)

    # shared error scale across configurations
    emax = max(np.nanpercentile(np.abs(F-F_ref), 99) for F in preds.values())

    ax = axes[0][0]
    im = ax.pcolormesh(X, Y, F_ref, cmap=cmap, shading='auto', **kw)
    draw_body(ax, cfg, ref); ax.set_title('Reference', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.04)
    axes[1][0].axis('off')
    axes[1][0].text(0.5,0.5,f'{var} field\nshared error scale\nmax = {emax:.3g}',
                    ha='center', va='center', fontsize=10, transform=axes[1][0].transAxes)

    for j,(name,F) in enumerate(preds.items(), start=1):
        ax = axes[0][j]
        im = ax.pcolormesh(X, Y, F, cmap=cmap, shading='auto', **kw)
        draw_body(ax, cfg, ref); ax.set_title(name, fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.04)

        ax = axes[1][j]
        im = ax.pcolormesh(X, Y, np.abs(F-F_ref), cmap='inferno', shading='auto', vmin=0, vmax=emax)
        draw_body(ax, cfg, ref); ax.set_title(f'|error|', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.04)

    for a in axes.ravel():
        if a.has_data():
            a.set_aspect('equal'); a.set_xlim(*cfg['xlim']); a.set_ylim(*cfg['ylim'])
            a.set_xlabel(cfg['xlab'], fontsize=9); a.tick_params(labelsize=8)

    fig.suptitle(f"{cfg['title']}: {var} reconstruction (f = 0.1, seed {SEED})",
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    out=f'{OUT}/fields_{key}_{var}.png'
    plt.savefig(out, dpi=165, bbox_inches='tight'); plt.close()
    print('  wrote', out)

for key,cfg in CASES.items():
    print(key)
    for var in ['u','v','p']:
        build(key,cfg,var)
