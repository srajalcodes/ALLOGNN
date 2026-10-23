"""
ALLOGNN per-structure variability analysis for the fixed held-out test set.

This version deliberately uses STRUCTURE/CHAIN-level variability, not gene-level
variability. Reviewer 1 explicitly allows per-gene OR per-structure variability.
It therefore avoids reconstructing gene assignments from ambiguous PDB metadata.

Run from the AlloGNN project root:
    python per_structure_variability.py

Uses the same test split and checkpoint as confidence_intervals_test_composition.py.
No retraining and no changes to split files.

Outputs:
  results/tables/type_iv_per_structure_metrics.csv
  results/tables/type_iv_per_structure_variability_summary.csv
  results/tables/type_iv_per_structure_variability_summary.json
"""

import os
os.environ["PYG_HOME"] = "data/pyg_cache"
os.environ["TORCH_GEOMETRIC_HOME"] = "data/pyg_cache"
os.environ["TORCH_HOME"] = "data/torch_cache"

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from src.data.dataset import AlloDataset
from src.models.allognn import AlloGNN

GRAPH_DIR = "data/processed/graphs"
EMBEDDINGS = "data/processed/embeddings/esm2_embeddings.h5"
TEST_SPLIT = "data/processed/splits/test_ids.txt"
CHECKPOINT = "output/checkpoints/best_allo_20260811_104436.pt"
BATCH_SIZE = 16
EXPECTED_TEST_CHAINS = 812
TYPE_IV_ID = 4
OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def flat_numpy(x):
    return x.detach().cpu().numpy().reshape(-1)


def calculate_metrics(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    auroc = np.nan
    aupr = np.nan
    if len(y) and np.unique(y).size == 2:
        auroc = float(roc_auc_score(y, p))
    if len(y) and y.sum() > 0:
        aupr = float(average_precision_score(y, p))
    return auroc, aupr


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "n_structures_evaluable": 0, "median": None, "q1": None, "q3": None,
            "iqr": None, "min": None, "max": None, "mean": None, "std": None,
        }
    q1, q3 = np.percentile(values, [25, 75])
    return {
        "n_structures_evaluable": int(len(values)),
        "median": float(np.median(values)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nCheckpoint: {CHECKPOINT}")

    if not Path(CHECKPOINT).exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not Path(TEST_SPLIT).exists():
        raise FileNotFoundError(f"Missing test split: {TEST_SPLIT}")

    test_ids = [
        line.strip()
        for line in Path(TEST_SPLIT).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(test_ids) != len(set(test_ids)):
        raise RuntimeError("Duplicate IDs in test_ids.txt.")
    if EXPECTED_TEST_CHAINS and len(test_ids) != EXPECTED_TEST_CHAINS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_CHAINS} test chains; found {len(test_ids)}."
        )

    dataset = AlloDataset(GRAPH_DIR, EMBEDDINGS, TEST_SPLIT)
    ids = list(dataset.valid_ids)
    if len(ids) != len(test_ids) or set(ids) != set(test_ids):
        raise RuntimeError("Loaded dataset IDs do not exactly match test_ids.txt.")
    if ids != test_ids:
        print("WARNING: dataset order differs from test_ids.txt; using dataset.valid_ids order.")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = AlloGNN().to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    rows = []
    offset = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(loader, 1):
            batch = batch.to(device)
            output = model(batch)
            pred = flat_numpy(output["binding"]).astype(float)
            y = flat_numpy(batch.y).astype(int)
            node_types = flat_numpy(batch.site_type[batch.batch]).astype(int)
            ptr = flat_numpy(batch.ptr).astype(int)

            if not (len(pred) == len(y) == len(node_types)):
                raise RuntimeError(f"Prediction/label/type length mismatch in batch {batch_i}")

            for j in range(len(ptr) - 1):
                start, end = ptr[j], ptr[j + 1]
                chain_id = ids[offset + j]
                if end <= start:
                    raise RuntimeError(f"{chain_id}: empty graph")
                types = np.unique(node_types[start:end])
                if len(types) != 1:
                    raise RuntimeError(f"{chain_id}: inconsistent site types {types}")
                if int(types[0]) != TYPE_IV_ID:
                    continue

                yy = y[start:end]
                pp = pred[start:end]
                auroc, aupr = calculate_metrics(yy, pp)
                rows.append({
                    "chain_id": chain_id,
                    "site_type": "Kinase Type IV",
                    "n_residues": int(len(yy)),
                    "n_positive_residues": int(yy.sum()),
                    "n_negative_residues": int(len(yy) - yy.sum()),
                    "positive_prevalence": float(yy.mean()),
                    "AUROC": auroc,
                    "AUPR": aupr,
                })

            offset += len(ptr) - 1
            if batch_i % 10 == 0 or offset == len(ids):
                print(f"Processed {offset}/{len(ids)} chains")

    if offset != len(ids):
        raise RuntimeError(f"Processed {offset} chains; expected {len(ids)}.")
    if not rows:
        raise RuntimeError(f"No Type IV chains found with site_type ID {TYPE_IV_ID}.")

    per_structure = pd.DataFrame(rows).sort_values("chain_id")
    per_structure_path = OUT_DIR / "type_iv_per_structure_metrics.csv"
    per_structure.to_csv(per_structure_path, index=False)

    summary_rows = []
    for metric in ["AUROC", "AUPR"]:
        summary_rows.append({
            "site_type": "Kinase Type IV",
            "metric": metric,
            "n_type_iv_structures": int(len(per_structure)),
            **summarize(per_structure[metric].to_numpy()),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "type_iv_per_structure_variability_summary.csv"
    summary.to_csv(summary_path, index=False)

    json_path = OUT_DIR / "type_iv_per_structure_variability_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": CHECKPOINT,
            "test_split": TEST_SPLIT,
            "analysis_unit": "individual held-out test chain/structure",
            "site_type": "Kinase Type IV",
            "n_test_chains": len(ids),
            "n_type_iv_structures": int(len(per_structure)),
            "method": (
                "Calculate residue-level AUROC/AUPR separately within each Type IV test "
                "chain, then summarize the distribution across chains. This is descriptive "
                "between-structure variability; it is not a confidence interval for pooled "
                "residue-level AUROC/AUPR."
            ),
            "per_structure_metrics": per_structure.to_dict(orient="records"),
            "variability_summary": summary.to_dict(orient="records"),
        }, f, indent=2)

    print("\n=== TYPE IV PER-STRUCTURE METRICS ===")
    print(per_structure.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== TYPE IV PER-STRUCTURE VARIABILITY SUMMARY ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved:")
    print(f"  {per_structure_path}")
    print(f"  {summary_path}")
    print(f"  {json_path}")
    print(
        "\nInterpretation: per-structure scores describe variation across individual "
        "test chains. Keep the existing pooled metrics and chain-bootstrap CIs."
    )


if __name__ == "__main__":
    main()
