import numpy as np, os, glob

ROOT = os.environ.get("SPHINX_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REF = {
 'cylinder':  ROOT+"/cases/cylinder_re40/cylinder_re40_reference.npz",
 'naca_re1k': ROOT+"/cases/naca0012_re1k/naca0012_re1000_aoa5_reference.npz",
 'naca_re10k':ROOT+"/cases/naca0012_re10k/naca0012_re10k_aoa5_reference.npz",
}

def cfg_paths(case):
    nc = case.replace('naca_','naca0012_')
    return {
     'overlap-nearbody':    glob.glob(f"{ROOT}/results/sensitivity_sweep/{case}/frac_0.1/seed_*/metrics.npz"),
     'overlap-quadrant':    glob.glob(f"{ROOT}/cases/*{nc}*/results/A_quadov/frac_0.1/seed_*/metrics.npz"),
     'nonoverlap-nearbody': glob.glob(f"{ROOT}/cases/*{nc}*/results/D_partition/seed_*/metrics*.npz"),
     'nonoverlap-quadrant': glob.glob(f"{ROOT}/cases/*{nc}*/results/nonoverlap/seed_*/metrics_nonoverlap.npz"),
    }

def body_mask(case, X, Y, ref):
    if case=='cylinder':
        cx,cy,R=float(ref['cx']),float(ref['cy']),float(ref['R'])
        return np.sqrt((X-cx)**2+(Y-cy)**2)<=R, (cx,cy,R)
    t=0.12; m=0.01; inr=(X>=-m)&(X<=1+m); xc=np.clip(X,0,1)
    yt=5*t*(0.2969*np.sqrt(xc)-0.126*xc-0.3516*xc**2+0.2843*xc**3-0.1015*xc**4)
    return inr&(np.abs(Y)<yt+m), None

def regions_for(case, X, Y, valid, geom):
    if case=='cylinder':
        cx,cy,R=geom; rr=np.sqrt((X-cx)**2+(Y-cy)**2)
        near=valid&(rr<=3.0*R); wake=valid&(X>cx)&(np.abs(Y-cy)<1.0)&(rr>3.0*R)
        outflow=valid&(X>X.max()-1.5)
    else:
        near=valid&(np.abs(Y)<0.35)&(X>-0.3)&(X<1.8)
        wake=valid&(X>=1.0)&(np.abs(Y)<0.4)&~near
        outflow=valid&(X>X.max()-0.5)
    far=valid&~near&~wake&~outflow
    return [('near-body',near),('wake',wake),('outflow',outflow),('far-field',far),('WHOLE',valid)]

def rel(pred,rf,mask,valid,gauge=False):
    p=pred.astype(float).copy(); r=rf.astype(float).copy()
    if gauge: p=p-np.nanmean(p[valid]); r=r-np.nanmean(r[valid])
    return np.sqrt(np.nanmean((p[mask]-r[mask])**2))/np.sqrt(np.nanmean(r[valid]**2))*100

for case in ['cylinder','naca_re1k','naca_re10k']:
    if not os.path.exists(REF[case]):
        print(f"\n### {case}: REFERENCE NOT FOUND at {REF[case]}"); continue
    ref=np.load(REF[case],allow_pickle=True)
    X,Y=ref['X'],ref['Y']; Uref,Vref,Pref=ref['U'],ref['V'],ref['P']
    body,geom=body_mask(case,X,Y,ref); valid=~body&np.isfinite(Uref)
    regions=regions_for(case,X,Y,valid,geom)
    print(f"\n{'='*70}\n {case.upper()}\n{'='*70}")
    for cfg,files in cfg_paths(case).items():
        if not files:
            print(f"\n  {cfg}: NO FILES FOUND"); continue
        acc={rn:{'u':[],'v':[],'p':[]} for rn,_ in regions}
        for f in sorted(files):
            m=np.load(f,allow_pickle=True); U,V,P=m['U_pred'],m['V_pred'],m['P_pred']
            for rn,mask in regions:
                acc[rn]['u'].append(rel(U,Uref,mask,valid))
                acc[rn]['v'].append(rel(V,Vref,mask,valid))
                acc[rn]['p'].append(rel(P,Pref,mask,valid,gauge=True))
        n=len(files)
        print(f"\n  --- {cfg}  ({n} seed{'s' if n!=1 else ''}) ---")
        print(f"  {'region':<12}{'u':>14}{'v':>14}{'p':>14}")
        for rn,_ in regions:
            u=np.array(acc[rn]['u']);v=np.array(acc[rn]['v']);p=np.array(acc[rn]['p'])
            print(f"  {rn:<12}{u.mean():>7.2f}+-{u.std():<5.2f}{v.mean():>7.2f}+-{v.std():<5.2f}{p.mean():>7.2f}+-{p.std():<5.2f}")
