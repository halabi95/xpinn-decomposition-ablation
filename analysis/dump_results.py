import numpy as np, glob, csv

rows = []
for f in glob.glob('**/metrics*.npz', recursive=True):
    try:
        r = np.load(f, allow_pickle=True)
        row = dict(path=f,
                   u=float(r['u_err']), v=float(r['v_err']), p=float(r['p_err']))
        for k in ['interface_frac','seed','n_points','case_tag']:
            row[k] = (float(r[k]) if k != 'case_tag' else str(r[k])) if k in r.files else ''
        rows.append(row)
    except Exception as e:
        rows.append(dict(path=f, u='', v='', p='', interface_frac='', seed='',
                         n_points='', case_tag=f'ERROR {e}'))

keys = ['path','case_tag','interface_frac','seed','n_points','u','v','p']
with open('all_results.csv','w',newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
    for r in rows: w.writerow({k: r.get(k,'') for k in keys})

print(f'wrote all_results.csv with {len(rows)} rows')

from collections import Counter
c = Counter('/'.join(r['path'].split('/')[:3]) for r in rows)
for k, n in sorted(c.items()):
    print(f'  {k:60} {n:4d}')
