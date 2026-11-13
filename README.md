# Deep Heteroskedastic Regression: Post-Hoc Variance Estimation from Latent Representations

Code for the NeurIPS 2026 paper [*Deep Heteroskedastic Regression: Post-Hoc Variance Estimation from Latent Representations*](https://arxiv.org/pdf/2603.01750).  

The method fits linear variance heads on the intermediate representations of a trained mean-only network, using a hold-out dataset, and combines them in a Gaussian scale mixture. This repository contains the code needed to reproduce the main results of the paper (Tables 1–2 and Figures 2–6).

## Setup

```bash
conda create -n pdhr python=3.12
conda activate pdhr
pip install -r requirements.txt
pip install -e .
```

The OMol25 experiments (UMA and AllScAIP) additionally need `fairchem-core`. The features were extracted with the AllScAIP fork of fairchem at commit `9ffa688`, with the patch in `fairchem_energy_only.patch` applied (it adds the `energy_only` inference setting, fairchem pull request #1714):

```bash
cd ..
git clone https://github.com/EricZQu/fairchem.git
cd fairchem
git checkout 9ffa688
git apply ../PracticalDeepHeteroskedasticRegression/fairchem_energy_only.patch
pip install -e packages/fairchem-core
pip install numpy==1.26.4
cd ../PracticalDeepHeteroskedasticRegression
```

`fairchem-core` upgrades numpy to version 2, with which the mixture ECE of the ensembles differs slightly from the paper, so the last command reinstalls numpy 1.26.4, the version the paper's results were computed with. pip then warns that `fairchem-core` requires numpy 2; the warning can be ignored.

## Data

- **QM9** is downloaded by PyTorch Geometric to `data/`.
- **UTKFace** is downloaded with `gdown` to `data/utkface/`.
- **California Housing** is downloaded with scikit-learn to `data/california_housing/`.
- **Year** is downloaded from the UCI repository to `data/year/`.

These four datasets can be downloaded with

```bash
python scripts/download_data.py
```

- **OMol25**: place the validation split of OMol25 (`data0000.aselmdb`, ..., `data0079.aselmdb` and `metadata.npz`) in `data/omol25/val/raw/`. The UMA model (`uma-s-1p1`) is downloaded by fairchem from Hugging Face, which requires access to the model and a Hugging Face token in the environment variable `HF_TOKEN`. The AllScAIP checkpoint is not publicly available; we obtained it directly from the authors of AllScAIP. It is expected at `checkpoints/omol_all_sm_NeAnNoSi_ft_40E20F_fixed.pt`.

## Running the experiments

The scripts in `scripts/` build a list of runs and execute the run with index `SLURM_ARRAY_TASK_ID`, so they can be submitted as SLURM array jobs:

```bash
sbatch scripts/qm9_01_baselines.sh
```

The UMA feature extraction and the UTKFace experiments need a GPU with 24 GB of memory.

Scripts with the same prefix must be run in order, since later scripts load the models saved by earlier ones in `models/`. The post-hoc variance heads are fitted on top of the homoskedastic models from the `*_01_baselines.sh` scripts, and the ensembles combine the variance heads fitted on the individual representations.

| Script | Experiment |
| --- | --- |
| `qm9_01_baselines.sh` | Homoskedastic, Naive NLL, Natural NLL, β-NLL and Faithful models on QM9 |
| `qm9_02_post_hoc.sh` | Post-hoc variance heads on each representation and on all representations |
| `qm9_03_post_hoc_ensemble.sh` | Post-hoc variance ensemble |
| `qm9_04_wd_sweep.sh`, `qm9_05_wd_sweep_ensemble.sh` | Weight decay sweep |
| `qm9_06_wd_sweep_mlp.sh` | MLP variance head |
| `qm9_07_learning_curve_baselines.sh`, `qm9_08_learning_curve_post_hoc.sh`, `qm9_09_learning_curve_post_hoc_ensemble.sh` | Hold-out dataset size |
| `utkface_01_baselines.sh`, `utkface_02_post_hoc.sh`, `utkface_03_post_hoc_ensemble.sh` | UTKFace |
| `tabular_01_baselines.sh`, `tabular_02_post_hoc.sh`, `tabular_03_post_hoc_ensemble.sh` | California Housing and Year |
| `omol25_01_create_features.sh` | Extracts the UMA and AllScAIP predictions and representations on OMol25 |
| `omol25_02_homo.sh`, `omol25_03_post_hoc.sh`, `omol25_04_post_hoc_ensemble.sh` | UMA and AllScAIP |
| `omol25_05_learning_curve.sh`, `omol25_06_learning_curve_ensemble.sh` | Hold-out dataset size for UMA |

## Tables and figures

Once the experiments have finished, the tables and figures are created with

| Result | Command | Required scripts |
| --- | --- | --- |
| Table 1 | `python tables/combined_table.py` | `qm9_01`–`03`, `utkface_*`, `tabular_*` |
| Table 2 | `python tables/omol25_table.py` | `omol25_01`–`04` |
| Figure 2 | `python plots/combined_feature_nll_rank.py` | `qm9_01`–`03`, `utkface_*`, `omol25_01`–`04` |
| Figure 3 (left) | `python plots/learning_curve.py` | `qm9_01`, `qm9_07`–`09` |
| Figure 3 (right) | `python plots/learning_curve_uma.py` | `omol25_01`, `omol25_05`–`06` |
| Figure 4 | `python plots/wd_sweep.py` | `qm9_01`–`05` |
| Figure 5 | `python plots/wd_sweep_mlp_table.py` | `qm9_01`–`06` |
| Figure 6 | `python plots/calib_plots_pretrained.py` | `omol25_01`–`04` |

The tables are written to `tables/` and the figures to `plots/`.
