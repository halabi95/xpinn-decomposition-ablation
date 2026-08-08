import re, glob, os

base_files = []
for case in ['cylinder_re40','naca0012_re1k','naca0012_re10k']:
    for f in glob.glob(f'cases/{case}/sphinx_*.py'):
        b = os.path.basename(f)
        if b.endswith('_D.py') or 'nonoverlap' in b or 'ckpt' in b or 'make_' in b or '_w0' in b:
            continue
        base_files.append(f)

NEW_WIDTHS = [0.5, 0.7]

for src in sorted(base_files):
    with open(src) as fh:
        code = fh.read()
    m = re.search(r'overlap = ([0-9.]+)', code)
    if not m:
        print(f'SKIP (no overlap= line): {src}')
        continue
    cur = float(m.group(1))
    for w in NEW_WIDTHS:
        if abs(w - cur) < 1e-9:
            print(f'skip {os.path.basename(src)} w={w} (== current, already have it)')
            continue
        new_code = re.sub(r'overlap = [0-9.]+', f'overlap = {w}  # WIDTH SWEEP', code, count=1)
        dst = src.replace('.py', f'_w{int(w*100):03d}.py')
        with open(dst, 'w') as fh:
            fh.write(new_code)
        print(f'wrote {os.path.basename(dst)}  (overlap {cur} -> {w})')
