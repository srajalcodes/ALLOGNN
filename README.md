# 🧬 ALLOGNN: Multi-Modal Graph Neural Network with Residue-Level Gated Fusion

Official PyTorch implementation of **"ALLOGNN: A Multi-Modal Graph Neural Network with Residue-Level Gated Fusion for Allosteric Binding Site Prediction in Human Protein Kinases."**

ALLOGNN is a multi-modal deep learning framework for residue-level prediction of active and allosteric binding sites in human protein kinases. The framework integrates evolutionary sequence information, structural topology, and local surface and biophysical features through a learned residue-level gating mechanism.

ALLOGNN is designed to address limitations of sequence-only binding-site prediction for structurally and evolutionarily diverse allosteric sites.

---

## Overview

ALLOGNN integrates three complementary biological modalities:

- **Sequence Stream (Evolutionary Context):** Extracts deep protein representations using the frozen **ESM2-650M** protein language model.
- **Structure Stream (Spatial Topology):** Propagates sequence information over a residue-level Cα contact graph using **Graph Attention Convolution (GATConv)**.
- **Surface Stream (Local Biophysical Profiling):** Encodes local physical characteristics, including relative solvent-accessible surface area (SASA), crystallographic B-factor, and secondary structure.
- **Residue-Level Gated Fusion:** A learned softmax gating mechanism dynamically assigns modality-specific weights to individual residues.

<p align="center">
  <img src="figures/architecture.png" width="850" alt="ALLOGNN Architecture">
</p>

ALLOGNN predicts five kinase binding-site classes:

1. **Kinase Type I** — Orthosteric
2. **Kinase Type I.5** — Back-pocket
3. **Kinase Type II** — DFG-out
4. **Kinase Type III** — Proximal allosteric
5. **Kinase Type IV** — Distal allosteric

---

## 📄 Paper & Dataset

**Paper:**  
*ALLOGNN: A Multi-Modal Graph Neural Network with Residue-Level Gated Fusion for Allosteric Binding Site Prediction in Human Protein Kinases*

**Dataset and Pre-trained Model:**  
[Zenodo — 10.5281/zenodo.21978244](https://doi.org/10.5281/zenodo.21978244)

Refer to the Zenodo record for the released dataset, precomputed resources, and model artifacts.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/srajalcodes/ALLOGNN.git
cd ALLOGNN
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate allognn
```

### 3. Download the precomputed dataset

```bash
python download_data.py
```

The download script retrieves the ALLOGNN archive from Zenodo and extracts its contents. The archive is approximately 14.8 GB; check the Zenodo record for the current file size and contents.

### 4. Evaluate the pre-trained model

```bash
python evaluation.py
```

The evaluation script loads the configured model checkpoint and evaluates it using the configured test split. The released evaluation setup is intended to use the held-out gene-cold test set containing 812 chains.

---

## 🗂️ Dataset Structure

The project uses the following resource layout. The files available after downloading depend on the contents of the Zenodo release.

```text
ALLOGNN/
│
├── data/
│   ├── raw/
│   │   └── kincore/
│   │       └── kincore_metadata.tab
│   │
│   └── processed/
│       ├── embeddings/
│       │   └── esm2_embeddings.h5
│       │
│       ├── graphs/
│       │   ├── *.pt
│       │   └── ...
│       │
│       └── splits/
│           ├── train_ids.txt
│           ├── val_ids.txt
│           └── test_ids.txt
│
├── output/
│   └── checkpoints/
│       └── best_allo_*.pt
│
├── results/
│
├── scripts/
│   └── reproducibility/
│       ├── README.md
│       ├── run_ablation_study.py
│       ├── run_bootstrap_ci.py
│       ├── run_sequence_only_baseline.py
│       ├── run_structure_variability.py
│       └── train_eval_seqonly_same_split.py
│
├── evaluation.py
├── train.py
├── download_data.py
├── environment.yml
└── README.md
```

The dataset release is associated with precomputed protein graphs, ESM2-650M residue representations, surface features, metadata, dataset splits, and model artifacts. Consult the Zenodo record to verify the exact contents of the published archive.

---

## 🔬 Reproducibility Experiments

The repository provides scripts for reproducing selected experiments and analyses associated with ALLOGNN.

Detailed descriptions, prerequisites, and execution instructions are available in the [reproducibility documentation](scripts/reproducibility/README.md).

Run commands from the repository root.

### Ablation Study

```bash
python scripts/reproducibility/run_ablation_study.py
```

This workflow trains and evaluates six model configurations to investigate the contributions of individual modalities and the residue-level gated fusion mechanism.

### Sequence-Only Baseline

```bash
python scripts/reproducibility/run_sequence_only_baseline.py
```

This workflow trains and evaluates a sequence-only baseline using the configured training, validation, and test splits.

### Bootstrap Confidence Intervals

```bash
python scripts/reproducibility/run_bootstrap_ci.py
```

This workflow computes performance metrics and chain-level bootstrap confidence intervals.

### Structure-Level Performance Variability

```bash
python scripts/reproducibility/run_structure_variability.py
```

This workflow evaluates performance across individual Type IV test structures and summarizes between-structure variability.

### Alternative Sequence-Only Workflow

```bash
python scripts/reproducibility/train_eval_seqonly_same_split.py
```

This provides an alternative sequence-only training and evaluation workflow. Its results should not be treated as interchangeable with the primary baseline until the provenance of the manuscript's reported result has been established.

**Reproducibility note:** Exact agreement between all script outputs and the manuscript's reported results has not yet been independently confirmed. Check the relevant configurations, data splits, checkpoint provenance, and generated outputs before claiming exact reproduction. Model selection should use the validation set, not the held-out test set.

---

## 🧪 Training

To train ALLOGNN using the prepared dataset:

```bash
python train.py
```

The training pipeline integrates sequence, structural, and surface representations through the residue-level gated fusion architecture.

Ensure that the required input data, dependencies, and configuration files are available before starting training.

---

## 📊 Results and Evaluation

ALLOGNN is evaluated across five kinase binding-site classes spanning orthosteric and allosteric binding regions.

The evaluation workflow reports class-wise performance metrics, including AUROC and AUPR. Additional analysis workflows support ablation studies, bootstrap confidence intervals, and structure-level performance variability.

See the [reproducibility documentation](scripts/reproducibility/README.md) for details of the available analysis workflows.

*Quantitative comparisons will be documented after the reported metrics have been reconciled with their corresponding checkpoints, evaluation scripts, and experimental logs.*

---

## 🧬 Binding-Site Classes

| Site Class | Description |
|---|---|
| **Type I** | Orthosteric ATP-binding site |
| **Type I.5** | Back-pocket binding site |
| **Type II** | DFG-out conformation / adjacent site |
| **Type III** | Proximal allosteric site |
| **Type IV** | Distal allosteric site |

---

## 📚 Citation

If you use ALLOGNN, its dataset, or the associated code in your research, please cite the following work:

```bibtex
@inproceedings{ALLOGNN2026,
  title={ALLOGNN: A Multi-Modal Graph Neural Network with Residue-Level Gated Fusion for Allosteric Binding Site Prediction in Human Protein Kinases},
  author={Tiwari, Srajal and Sharma, Dolly},
  year={2026}
}
```

### Dataset Citation

```bibtex
@dataset{tiwari2026ALLOGNN,
  author       = {Tiwari, Srajal and Sharma, Dolly},
  title        = {ALLOGNN: Data and Pre-Trained Models for Allosteric Binding Site Prediction in Human Protein Kinases},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.21978244}
}
```

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📬 Contact

For questions regarding the ALLOGNN implementation, dataset, or reproducibility materials, please open an issue in this repository.

**GitHub:**  
https://github.com/srajalcodes/ALLOGNN

**Zenodo:**  
https://doi.org/10.5281/zenodo.21978244
