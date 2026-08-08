#!/usr/bin/env python3
"""
make_nonoverlap_general.py -- build a non-overlapping standard-XPINN baseline
from a SPHINX case script that uses the Upstream/Top-Wake/Bot-Wake/Near-Body
overlapping decomposition. Replaces it with 4 non-overlapping quadrants about
(x_split, y_split) and samples interface points on the shared axis lines,
excluding the body interior.

Usage ON SQUID, in the case directory:
    python make_nonoverlap_general.py <script.py> <body_kind>
  where <body_kind> is 'naca' or 'fsi' (controls the body-exclusion function name).

Produces <script>_nonoverlap.py (original untouched).

This is the NACA/FSI analogue of the cylinder non-overlap patch. The only
changes are the decomposition, the interface list, and interface sampling.
Architecture, BC enforcement, PDE loss, optimizer, seeds: all unchanged.
"""
import sys, re

if len(sys.argv) < 2:
    print("usage: python make_nonoverlap_general.py <script.py>"); sys.exit(1)
SRC = sys.argv[1]
DST = SRC.replace('.py', '_nonoverlap.py')

with open(SRC) as f:
    code = f.read()

# ---- 1. Find the split/overlap definition line and force overlap=0 ----
# matches either "x_split = ...; y_split = ...; overlap = 0.3"
# or the FSI form using ovlp
if 'overlap = ' in code:
    code = re.sub(r"overlap = [0-9.]+",
                  "overlap = 0.0  # NON-OVERLAP (standard XPINN, Major 1b)",
                  code, count=1)
if 'ovlp' in code:
    # FSI uses 'ovlp'; set it to 0 wherever it's first defined
    code = re.sub(r"ovlp\s*=\s*[0-9.][0-9.eE+-]*",
                  "ovlp = 0.0  # NON-OVERLAP (standard XPINN, Major 1b)",
                  code, count=1)

# ---- 2. Replace the subdomains dict with clean quadrants ----
# We rebuild quadrants using whatever split vars the script already defines.
# NACA: x_split, y_split, x_min_d/x_max_d/y_min_d/y_max_d
# FSI : beam_mid_x_n, cy_n, x_min_n/x_max_n/y_min_n/y_max_n
sub_match = re.search(r"subdomains\s*=\s*\{.*?\n\}", code, flags=re.S)
if not sub_match:
    print("ERROR: could not find subdomains dict"); sys.exit(1)
old_sub = sub_match.group(0)

if 'x_min_d' in code:  # NACA-style
    new_sub = """subdomains = {
    0: {'name': 'Q-TL', 'x_lo': x_min_d, 'x_hi': x_split, 'y_lo': y_split, 'y_hi': y_max_d},
    1: {'name': 'Q-TR', 'x_lo': x_split, 'x_hi': x_max_d, 'y_lo': y_split, 'y_hi': y_max_d},
    2: {'name': 'Q-BL', 'x_lo': x_min_d, 'x_hi': x_split, 'y_lo': y_min_d, 'y_hi': y_split},
    3: {'name': 'Q-BR', 'x_lo': x_split, 'x_hi': x_max_d, 'y_lo': y_min_d, 'y_hi': y_split},
}"""
elif 'x_min_n' in code:  # FSI-style
    new_sub = """subdomains = {
    0: {'name': 'Q-TL', 'x_lo': x_min_n, 'x_hi': beam_mid_x_n, 'y_lo': cy_n, 'y_hi': y_max_n},
    1: {'name': 'Q-TR', 'x_lo': beam_mid_x_n, 'x_hi': x_max_n, 'y_lo': cy_n, 'y_hi': y_max_n},
    2: {'name': 'Q-BL', 'x_lo': x_min_n, 'x_hi': beam_mid_x_n, 'y_lo': y_min_n, 'y_hi': cy_n},
    3: {'name': 'Q-BR', 'x_lo': beam_mid_x_n, 'x_hi': x_max_n, 'y_lo': y_min_n, 'y_hi': cy_n},
}"""
else:
    print("ERROR: unknown domain-bound variable convention"); sys.exit(1)
code = code.replace(old_sub, new_sub)

# ---- 3. Replace interfaces list with the 4 axis interfaces ----
if_match = re.search(r"interfaces\s*=\s*\[.*?\]", code, flags=re.S)
old_if = if_match.group(0)
new_if = """interfaces = [
    (0, 1, 'vertical'), (2, 3, 'vertical'),
    (0, 2, 'horizontal'), (1, 3, 'horizontal'),
]"""
code = code.replace(old_if, new_if)

# ---- 4. Rewrite the interface-sampling function to sample on the shared line ----
# Both NACA and FSI define it as a function that computes x_lo/x_hi/y_lo/y_hi
# from the two subdomains and returns empty when degenerate. We replace the body.
# Identify body-exclusion fn: NACA uses inside_airfoil? FSI uses inside_solid?
body_fn = None
for cand in ['inside_airfoil', 'inside_body', 'inside_solid', 'inside_cylinder', 'point_in_body']:
    if cand+'(' in code:
        body_fn = cand; break

# locate the sampling function definition (the one with overlap-region logic)
samp = re.search(r"def (sample_interface\w*)\(sd_i, sd_j, n, rng\):.*?\n(?=\ndef |\n# |\n[A-Za-z_]+\s*=)", code, flags=re.S)
if not samp:
    print("WARNING: could not auto-locate interface sampling fn; manual edit needed")
else:
    fn_name = samp.group(1)
    excl = f"    mask = ~{body_fn}(xs, ys)\n" if body_fn else "    mask = np.ones(len(xs), dtype=bool)\n"
    new_fn = f'''def {fn_name}(sd_i, sd_j, n, rng):
    """Sample on the shared edge between non-overlapping quadrants, excluding
    the body interior. Standard XPINN interface (Major 1b)."""
    x_lo = max(sd_i['x_lo'], sd_j['x_lo']); x_hi = min(sd_i['x_hi'], sd_j['x_hi'])
    y_lo = max(sd_i['y_lo'], sd_j['y_lo']); y_hi = min(sd_i['y_hi'], sd_j['y_hi'])
    if abs(x_hi - x_lo) < 1e-9:
        ys = rng.uniform(y_lo, y_hi, n*4); xs = np.full_like(ys, x_lo)
    elif abs(y_hi - y_lo) < 1e-9:
        xs = rng.uniform(x_lo, x_hi, n*4); ys = np.full_like(xs, y_lo)
    else:
        return np.zeros((0, 2))
{excl}    xs, ys = xs[mask], ys[mask]
    m = min(len(xs), n)
    return np.column_stack([xs[:m], ys[:m]]) if m > 0 else np.zeros((0, 2))

'''
    code = code.replace(samp.group(0), new_fn)

# ---- 5. tag output so SPHINX results aren't overwritten ----
code = code.replace("'metrics.npz'", "'metrics_nonoverlap.npz'")

with open(DST, 'w') as f:
    f.write(code)
print(f"Wrote {DST}")
print(f"  body-exclusion fn detected: {body_fn}")
print(f"  interface sampling fn: {samp.group(1) if samp else 'NOT FOUND - CHECK MANUALLY'}")
print("  REVIEW THE DIFF before running. Then add checkpointing + smoke-test.")
