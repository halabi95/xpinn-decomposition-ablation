import sys, re
base, out = sys.argv[1], sys.argv[2]
src = open(base).read()

# (1) overlap -> 0  (FSI uses 'ovlp')
src = re.sub(r"ovlp\s*=\s*0\.15\s*/\s*L_ref", "ovlp = 0.0  # CONFIG D: partition", src, count=1)

# (2) point_in_subdomain (FSI multi-line) -> partition membership
old_pis = """def point_in_subdomain(x, y, sd):
    return ((x >= sd['x_lo']) & (x <= sd['x_hi']) &
            (y >= sd['y_lo']) & (y <= sd['y_hi']))"""
new_pis = """def point_in_subdomain(x, y, sd):
    nb = subdomains[3]
    in_nb = ((x >= nb['x_lo']) & (x <= nb['x_hi']) &
             (y >= nb['y_lo']) & (y <= nb['y_hi']))
    nm = sd['name']
    if nm == 'Near-body':
        return in_nb
    elif nm == 'Upstream':
        return (~in_nb) & (x < beam_mid_x_n)
    elif nm == 'Top-downstream':
        return (~in_nb) & (x >= beam_mid_x_n) & (y >= cy_n)
    else:
        return (~in_nb) & (x >= beam_mid_x_n) & (y < cy_n)"""
if old_pis not in src: print("ERROR: point_in_subdomain not found"); sys.exit(2)
src = src.replace(old_pis, new_pis, 1)

# (3) sample_interface (keep snap_idx/bbox args; use inside_obstacle exclusion)
new_si = '''def _zone_name_at(x, y):
    nb = subdomains[3]
    if (nb['x_lo'] <= x <= nb['x_hi']) and (nb['y_lo'] <= y <= nb['y_hi']):
        return 'Near-body'
    if x < beam_mid_x_n: return 'Upstream'
    if y >= cy_n: return 'Top-downstream'
    return 'Bot-downstream'

def sample_interface(sd_i, sd_j, n, rng, snap_idx=-1, bbox=None):
    import numpy as _np
    nb = subdomains[3]
    margin = 0.005 / L_ref
    def _excl(x, y):
        if bbox is not None:
            return bool(inside_obstacle_fast(_np.array([x]), _np.array([y]), bbox, margin)[0])
        return bool(inside_obstacle_n(_np.array([x]), _np.array([y]), snap_idx, margin)[0])
    s = {sd_i['name'], sd_j['name']}
    pts, nrm = [], []
    def add(x, y, nx, ny):
        if not _excl(x, y):
            pts.append([x, y]); nrm.append([nx, ny])
    if s == {'Upstream', 'Top-downstream'}:
        for y in rng.uniform(cy_n, subdomains[1]['y_hi'], n*4):
            if not (nb['x_lo'] <= beam_mid_x_n <= nb['x_hi'] and nb['y_lo'] <= y <= nb['y_hi']):
                add(beam_mid_x_n, y, 1.0, 0.0)
    elif s == {'Upstream', 'Bot-downstream'}:
        for y in rng.uniform(subdomains[2]['y_lo'], cy_n, n*4):
            if not (nb['x_lo'] <= beam_mid_x_n <= nb['x_hi'] and nb['y_lo'] <= y <= nb['y_hi']):
                add(beam_mid_x_n, y, 1.0, 0.0)
    elif s == {'Top-downstream', 'Bot-downstream'}:
        for x in rng.uniform(beam_mid_x_n, subdomains[1]['x_hi'], n*4):
            if not (nb['x_lo'] <= x <= nb['x_hi'] and nb['y_lo'] <= cy_n <= nb['y_hi']):
                add(x, cy_n, 0.0, 1.0)
    else:
        other = (s - {'Near-body'}).pop()
        edges = [('x', nb['x_lo'], (nb['y_lo'], nb['y_hi']), (-1e-3, 0.0)),
                 ('x', nb['x_hi'], (nb['y_lo'], nb['y_hi']), (+1e-3, 0.0)),
                 ('y', nb['y_lo'], (nb['x_lo'], nb['x_hi']), (0.0, -1e-3)),
                 ('y', nb['y_hi'], (nb['x_lo'], nb['x_hi']), (0.0, +1e-3))]
        for axis, fixed, (lo, hi), (ox, oy) in edges:
            for t in rng.uniform(lo, hi, n*2):
                if axis == 'x': x, y, nx, ny = fixed, t, 1.0, 0.0
                else:           x, y, nx, ny = t, fixed, 0.0, 1.0
                if _zone_name_at(x+ox, y+oy) == other:
                    add(x, y, nx, ny)
    if len(pts) == 0:
        return _np.zeros((0, 2)), _np.zeros((0, 2))
    P = _np.array(pts); N = _np.array(nrm)
    idx = rng.permutation(len(P))[:n]
    return P[idx], N[idx]'''
m = re.search(r"def sample_interface\(sd_i, sd_j, n, rng, snap_idx=-1, bbox=None\):\n(?:.*\n)+?    return np\.array\(pts\[:n\]\)\n", src)
if not m: print("ERROR: sample_interface not found"); sys.exit(3)
src = src[:m.start()] + new_si + "\n" + src[m.end():]

# (4a) call site keeps snap_idx, bbox
old_call = "            xy_if = sample_interface(sd_i, sd_j, N_intf, rng, snap_idx, bbox)"
new_call = "            xy_if, nrm_if = sample_interface(sd_i, sd_j, N_intf, rng, snap_idx, bbox)"
if old_call not in src: print("WARN(4a): call site not found verbatim")
src = src.replace(old_call, new_call, 1)

# (4b) normal heuristic (FSI multi-line) -> per-point normal
old_norm = """            x_lo_ov = max(sd_i['x_lo'], sd_j['x_lo'])
            x_hi_ov = min(sd_i['x_hi'], sd_j['x_hi'])
            y_lo_ov = max(sd_i['y_lo'], sd_j['y_lo'])
            y_hi_ov = min(sd_i['y_hi'], sd_j['y_hi'])
            normal = (torch.tensor([[1.0, 0.0]], device=device)
                      if (x_hi_ov - x_lo_ov) < (y_hi_ov - y_lo_ov)
                      else torch.tensor([[0.0, 1.0]], device=device))"""
new_norm = "            normal = torch.tensor(nrm_if, dtype=torch.float32, device=device)"
if old_norm not in src: print("WARN(4b): normal heuristic not found verbatim")
src = src.replace(old_norm, new_norm, 1)

open(out, "w").write(src); print("Wrote", out)
