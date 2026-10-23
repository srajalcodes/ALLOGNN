# AlloGNN Reproducibility Experiments

This directory contains scripts for reproducing selected experiments and analyses associated with **AlloGNN: A Multi-Modal Graph Neural Network with Residue-Level Gated Fusion for Allosteric Binding Site Prediction in Human Protein Kinases**.

The scripts cover ablation experiments, sequence-only baseline evaluation, bootstrap confidence intervals, and structure-level performance variability.

## Experimental Workflows

| Script | Purpose |
|---|---|
| `run_ablation_study.py` | Trains and evaluates six model configurations to examine the contributions of individual modalities and gated fusion. |
| `run_sequence_only_baseline.py` | Trains and evaluates a sequence-only baseline using the predefined train, validation, and test splits. |
| `run_bootstrap_ci.py` | Computes residue-level performance metrics and chain-level bootstrap confidence intervals. |
| `run_structure_variability.py` | Evaluates performance across individual Type IV test structures and summarizes between-structure variability. |
| `train_eval_seqonly_same_split.py` | Provides an alternative sequence-only training and evaluation workflow. Its relationship to the final reported baseline must be verified. |

## Prerequisites

Before running the scripts, ensure that you have:

- Installed the dependencies specified in the repository's `environment.yml`.
- Downloaded and extracted the required dataset and precomputed ESM2 embeddings.
- Prepared the train, validation, and held-out test split files.
- Obtained the exact model checkpoint required by each evaluation script.

See the [main repository README](../../README.md) and the [Zenodo dataset](https://doi.org/10.5281/zenodo.21978244) for setup and data access.

## Running the Experiments

Run commands from the **repository root**, not from this directory.

```bash
python scripts/reproducibility/run_ablation_study.py
```

```bash
python scripts/reproducibility/run_sequence_only_baseline.py
```

```bash
python scripts/reproducibility/run_bootstrap_ci.py
```

```bash
python scripts/reproducibility/run_structure_variability.py
```

The alternative sequence-only workflow can be run with:

```bash
python scripts/reproducibility/train_eval_seqonly_same_split.py
```

Each script may require specific data files, checkpoints, and output directories. Review its configuration before execution.

## Reproducibility Notes

- Use the same data splits, model configuration, checkpoint, and evaluation protocol as the corresponding manuscript experiment.
- Select model checkpoints using the validation set. Do not use the held-out test set for model selection.
- Preserve experiment logs and generated outputs so that reported metrics can be traced to their source runs.
- Verify that the results produced by each script agree with the corresponding manuscript tables or figures before claiming exact reproduction.
- The sequence-only workflows should not be treated as interchangeable until the provenance of the reported baseline has been established.

**Status:** These scripts provide reproducibility workflows for selected experiments. Their agreement with every reported manuscript result has not yet been independently confirmed.