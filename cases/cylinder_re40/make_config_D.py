import sys, re
base, out = sys.argv[1], sys.argv[2]
src = open(base).read()

# (1) overlap = 0
src = re.sub(r"overlap\s*=\s*0\.5", "overlap = 0.0  # CONFIG D: partition", src, count=1)

# (2) partition membership
new_pis = '''def point_in_subdomain(x, y, sd):
    nb = subdomains[3]
    in_nb = ((x >= nb['x_lo']) & (x <= nb['x_hi']) &
             (y >= nb['y_lo']) & (y <= nb['y_hi']))
    nm = sd['name']
    if nm == 'Near-Body':
        return in_nb
    elif nm == 'Upstream':
        return (~in_nb) & (x < x_split)
    elif nm == 'Top-Wake':
        return (~in_nb) & (x >= x_split) & (y >= y_split)
    else:
        return (~in_nb) & (x >= x_split) & (y < y_split)'''
m = re.search(r"def point_in_subdomain\(x, y, sd\):\n    return \(\(x >= sd\['x_lo'\]\).*?\)\)\n", src, flags=re.DOTALL)
if not m: print("ERROR: point_in_subdomain not found"); sys.exit(2)
src = src[:m.start()] + new_pis + "\n" + src[m.end():]

# (3) partition shared-boundary sampler + per-point normals
new_si = '''def _zone_name_at(x, y):
    nb = subdomains[3]
    if (nb['x_lo'] <= x <= nb['x_hi']) and (nb['y_lo'] <= y <= nb['y_hi']):
        return 'Near-Body'
    if x < x_split: return 'Upstream'
    if y >= y_split: return 'Top-Wake'
    return 'Bot-Wake'

def sample_interface(sd_i, sd_j, n, rng):
    nb = subdomains[3]
    s = {sd_i['name'], sd_j['name']}
    pts, nrm = [], []
    def add(x, y, nx, ny):
        if not inside_cylinder(x, y, margin=0.02):
            pts.append([x, y]); nrm.append([nx, ny])
    if s == {'Upstream', 'Top-Wake'}:
        for y in rng.uniform(y_split, subdomains[1]['y_hi'], n*4):
            if not (nb['x_lo'] <= x_split <= nb['x_hi'] and nb['y_lo'] <= y <= nb['y_hi']):
                add(x_split, y, 1.0, 0.0)
    elif s == {'Upstream', 'Bot-Wake'}:
        for y in rng.uniform(subdomains[2]['y_lo'], y_split, n*4):
            if not (nb['x_lo'] <= x_split <= nb['x_hi'] and nb['y_lo'] <= y <= nb['y_hi']):
                add(x_split, y, 1.0, 0.0)
    elif s == {'Top-Wake', 'Bot-Wake'}:
        for x in rng.uniform(x_split, subdomains[1]['x_hi'], n*4):
            if not (nb['x_lo'] <= x <= nb['x_hi'] and nb['y_lo'] <= y_split <= nb['y_hi']):
                add(x, y_split, 0.0, 1.0)
    else:
        other = (s - {'Near-Body'}).pop()
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
        return np.zeros((0, 2)), np.zeros((0, 2))
    P = np.array(pts); N = np.array(nrm)
    idx = rng.permutation(len(P))[:n]
    return P[idx], N[idx]'''
m = re.search(r"def sample_interface\(sd_i, sd_j, n, rng\):\n(?:    .*\n|\n)+?    return np\.array\(pts\[:n\]\)\n", src)
if not m: print("ERROR: sample_interface not found"); sys.exit(3)
src = src[:m.start()] + new_si + "\n" + src[m.end():]

# (4a) capture normals at the call site
old_call = "            xy_if = sample_interface(sd_i, sd_j, N_intf, rng)\n            if len(xy_if) == 0:\n                continue"
new_call = "            xy_if, nrm_if = sample_interface(sd_i, sd_j, N_intf, rng)\n            if len(xy_if) == 0:\n                continue\n            normal = torch.tensor(nrm_if, dtype=torch.float32, device=device)"
if old_call not in src: print("WARN(4a): call site not found verbatim")
src = src.replace(old_call, new_call, 1)

# (4b) remove thin-box normal heuristic
old_norm = """            x_lo_ov = max(sd_i['x_lo'], sd_j['x_lo'])
            x_hi_ov = min(sd_i['x_hi'], sd_j['x_hi'])
            y_lo_ov = max(sd_i['y_lo'], sd_j['y_lo'])
            y_hi_ov = min(sd_i['y_hi'], sd_j['y_hi'])
            normal = (torch.tensor([[1.0, 0.0]], device=device)
                      if (x_hi_ov - x_lo_ov) < (y_hi_ov - y_lo_ov)
                      else torch.tensor([[0.0, 1.0]], device=device))"""
if old_norm not in src: print("WARN(4b): normal heuristic block not found verbatim")
src = src.replace(old_norm, "            # CONFIG D: per-point normal set at call site above", 1)

open(out, "w").write(src)
print("Wrote", out)
