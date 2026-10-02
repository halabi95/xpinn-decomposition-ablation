# Non-overlapping near-body zoning is the most reliable of four decompositions for sparse-data flow reconstruction with extended physics-informed neural networks

Code, reference solutions and results accompanying the manuscript of the same
name by H. El Halabi and Y. Takahashi (under review).

The study is a controlled 2x2 ablation crossing subdomain zoning (near-body
versus quadrant) with interface overlap (overlapping versus non-overlapping)
in extended physics-informed neural networks, evaluated on four steady
benchmarks at a matched measurement budget, for u, v and p, over three seeds.

## Layout

    cases/<case>/                 one directory per benchmark
      sphinx_<case>.py              overlapping, near-body zoning
      sphinx_<case>_A.py            overlapping, quadrant zoning
      sphinx_<case>_D.py            non-overlapping, near-body zoning
      sphinx_<case>_nonoverlap.py   non-overlapping, quadrant zoning
      make_*.py                     generators that derive the variants from the base script
      *_reference.npz               CFD reference solution
    cases/cylinder_re40/xpinn_nonoverlap_cylinder.py
                                  non-overlapping quadrant run for the cylinder (see Scope)
    analysis/                     post-processing, region-resolved and surface analyses, RBF baseline
    data/                         results not contained in all_results.csv (see Scope)
    jobs/                         example PBS submission script
    all_results.csv               one row per run

## Benchmarks and reference solvers

| case | Re | reference solver |
|---|---|---|
| cylinder_re40 | 40 | FEniCS, laminar |
| naca0012_re1k | 1,000 | FEniCS, laminar |
| naca0012_re10k | 10,000 | SU2, k-omega SST (eddy viscosity supplied to the reconstruction) |
| fsi1_re20 | 20 | turtleFSI, final converged snapshot |

The three external-flow references are stored on structured evaluation grids;
the fluid-structure reference is stored on its unstructured mesh. For both
airfoil cases the airfoil is at zero geometric incidence and the 5 degree
angle of attack is imposed through the freestream direction (U_x, U_y in the
reference file).

## Running a case

    python3 sphinx_naca_re1k.py --interface-frac 0.1 --seed 42 \
        --n-epochs 100000 --output-dir results/frac_0.1/seed_42

The non-overlapping configurations are run with `--interface-frac 0.0`.
Runs write `metrics.npz`, or `metrics_nonoverlap.npz` for the `_nonoverlap`
scripts, containing the relative L2 errors and the predicted fields.

Notes on what the scripts do:

- Each measurement supplies u, v and p, noise-free. For the grid cases the
  locations are drawn at random and the values interpolated linearly from the
  reference grid; for FSI1 they are taken at random fluid nodes.
- `--interface-frac f` places a fraction f of the budget inside the subdomain
  shared regions for the cylinder and airfoil cases. In the `fsi1_re20`
  scripts it instead draws that fraction, with replacement, from the
  fluid-structure interface nodes on the cylinder and beam surfaces.
- Errors are relative L2 errors over the fluid points of the evaluation grid.
  The pressure error is computed on the pressure itself, without removing its
  domain mean, as in the paper.

Requires PyTorch, NumPy, SciPy and Matplotlib. Production runs used single
NVIDIA A100 (40 GB) GPUs; a Quadro RTX 6000 was used only for short tests.

## all_results.csv

Columns: `path`, `case_tag`, `interface_frac`, `seed`, `n_points`, `u`, `v`, `p`
(errors in %). The configuration is identified from the path:
`sensitivity_sweep` overlapping near-body, `A_quadov` overlapping quadrant,
`D_partition` non-overlapping near-body, `nonoverlap` non-overlapping quadrant,
`width_sweep/<zoning>/w0XXX` the matched-width study, `n_scaling_f01` the
budget sweep, `rbf_baseline` the interpolation baseline.

Not used in the paper: the 12 `width_sweep` rows without a `nearbody` or
`quadrant` folder (a superseded first sweep) and the 15 `n_scaling_f03_legacy`
rows (an earlier sweep at f = 0.3).

The file is generated from the per-run outputs by `analysis/dump_results.py`.

## Scope

Together with `data/`, these files contain every value reported in the paper.

- The cylinder non-overlapping quadrant configuration was run by
  `cases/cylinder_re40/xpinn_nonoverlap_cylinder.py` in an earlier campaign
  with the same architecture, epoch count, seeds and measurement budget. Its
  results are in `data/xpinn_nonoverlap_cylinder_random_results.npz` under the
  `SPHINX_X_N200` keys (a legacy name); the stored fields are those of its
  lowest-error seed (456).
- Region-resolved errors (Supplementary Table S4) are in
  `data/regional_errors.txt`, produced by `analysis/regional_analysis.py`.
- Surface pressures (Fig. 4, Table 5) are in `data/surface_cp.npz`, produced
  by `analysis/compute_forces.py`.

The raw per-run output directories are not included because of their size;
the analysis scripts regenerate the values above from them.
