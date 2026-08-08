# Near-body zoning and interface overlap in XPINNs for sparse-data flow reconstruction

Code and reference solutions for the paper of the same name.

The study is a controlled 2x2 ablation crossing subdomain zoning (near-body
versus quadrant) with interface overlap (overlapping versus non-overlapping),
evaluated on four steady benchmarks.

## Layout

    cases/<case>/           one directory per benchmark
      sphinx_<case>.py         overlap, near-body zoning
      sphinx_<case>_A.py       overlap, quadrant zoning
      sphinx_<case>_D.py       non-overlap, near-body zoning
      sphinx_<case>_nonoverlap.py   non-overlap, quadrant zoning
      make_*.py                generators deriving the above from the base script
      *_reference.npz          CFD reference solution
    analysis/               post-processing and the RBF baseline
    jobs/                   example PBS submission script
    all_results.csv         every run: case, configuration, f, seed, u/v/p error

## Benchmarks and reference solvers

| case | Re | solver |
|---|---|---|
| cylinder_re40 | 40 | FEniCS |
| naca0012_re1k | 1,000 | FEniCS |
| naca0012_re10k | 10,000 | SU2, k-omega SST |
| fsi1_re20 | 20 | turtleFSI |

Reference fields are stored on structured grids for the three external-flow
cases and on the original unstructured mesh for the fluid-structure case.

## Running a case

    python3 sphinx_naca_re1k.py --interface-frac 0.1 --seed 42 \
        --n-epochs 100000 --output-dir results/frac_0.1/seed_42

The non-overlapping configurations take `--interface-frac 0.0`, since they have
no overlap region. Each run writes `metrics.npz` containing the relative L2
errors and the predicted fields.

Requires PyTorch, NumPy and SciPy. Runs were performed on NVIDIA A100 and
Quadro RTX 6000 GPUs.

## Note on scope

`all_results.csv` contains every error value reported in the paper. The raw
per-run output directories are not included here, as they total several hundred
megabytes; the CSV is generated from them by `analysis/dump_results.py`.

The cylinder non-overlapping quadrant configuration was produced by
`cases/cylinder_re40/xpinn_nonoverlap_cylinder.py`, from an earlier campaign
using the same architecture, epoch count, seeds and measurement budget as the
other configurations.
