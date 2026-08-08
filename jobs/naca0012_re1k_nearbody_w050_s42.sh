#!/bin/bash
#PBS -N W2_naca0012_re1k_nearbody_w050_s42
#PBS -q SQUID
#PBS --group=hp260073
#PBS -l elapstim_req=120:00:00
#PBS -l gpunum_job=1
#PBS -l memsz_job=20gb
#PBS -j o
#PBS -o /sqfs2/cmc/1/home/z6b512/sphinx_project/GITHUB_CODE/SPHINX-main/cases/naca0012_re1k/results/width_sweep/nearbody/w0050/frac_0.1/seed_42/job.o
#PBS -V
source /sqfs/work/hp240004/z6b512/miniforge/etc/profile.d/conda.sh
conda activate /sqfs/work/hp240004/z6b512/conda_env/sphinx
cd /sqfs2/cmc/1/home/z6b512/sphinx_project/GITHUB_CODE/SPHINX-main/cases/naca0012_re1k
python3 sphinx_naca_re1k_w050.py  --interface-frac 0.1 --seed 42 --n-epochs 100000 --output-dir results/width_sweep/nearbody/w0050/frac_0.1/seed_42
