import sys, re
base, out = sys.argv[1], sys.argv[2]
src = open(base).read()
new_subdomains = """subdomains = {
    0: {'name': 'Q-TL',
        'x_lo': x_min_n, 'x_hi': beam_mid_x_n + ovlp,
        'y_lo': cy_n - ovlp, 'y_hi': y_max_n},
    1: {'name': 'Q-TR',
        'x_lo': beam_mid_x_n - ovlp, 'x_hi': x_max_n,
        'y_lo': cy_n - ovlp, 'y_hi': y_max_n},
    2: {'name': 'Q-BL',
        'x_lo': x_min_n, 'x_hi': beam_mid_x_n + ovlp,
        'y_lo': y_min_n, 'y_hi': cy_n + ovlp},
    3: {'name': 'Q-BR',
        'x_lo': beam_mid_x_n - ovlp, 'x_hi': x_max_n,
        'y_lo': y_min_n, 'y_hi': cy_n + ovlp},
}"""
new_interfaces = """interfaces = [
    (0, 1, 'vertical'), (2, 3, 'vertical'),
    (0, 2, 'horizontal'), (1, 3, 'horizontal'),
    (0, 3, 'overlap'), (1, 2, 'overlap'),
]"""
m = re.search(r"subdomains\s*=\s*\{.*?\n\}", src, flags=re.DOTALL)
if not m: print("ERROR: no subdomains block"); sys.exit(2)
src = src[:m.start()] + new_subdomains + src[m.end():]
m2 = re.search(r"interfaces\s*=\s*\[.*?\n\]", src, flags=re.DOTALL)
if not m2: print("ERROR: no interfaces block"); sys.exit(3)
src = src[:m2.start()] + new_interfaces + src[m2.end():]
if "'Near-body'" in src or "'Near-Body'" in src: print("WARN: Near-body still present")
open(out,"w").write(src); print("Wrote", out)
