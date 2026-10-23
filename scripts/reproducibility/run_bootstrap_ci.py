"""
ALLOGNN test-set composition + chain-level bootstrap confidence intervals.

Run from the AlloGNN project root:
    python confidence_intervals_test_composition.py

IMPORTANT: Set CHECKPOINT to the exact checkpoint used for the final manuscript's
main test results. Logs previously associated Type IV AUPR=0.853 with
best_allo_20260731_135907.pt. Verify this before running.
"""
import os
os.environ["PYG_HOME"] = "data/pyg_cache"
os.environ["TORCH_GEOMETRIC_HOME"] = "data/pyg_cache"
os.environ["TORCH_HOME"] = "data/torch_cache"

import json
from pathlib import Path
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
N_BOOTSTRAPS = 2000
SEED = 42
EXPECTED_TEST_CHAINS = 812
OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TYPE_NAMES = {
    0: "Kinase Type I",
    1: "Kinase Type I.5",
    2: "Kinase Type II",
    3: "Kinase Type III",
    4: "Kinase Type IV",
}

def to_numpy(x):
    return x.detach().cpu().numpy().reshape(-1)

def metrics(y, p):
    out = {"AUROC": np.nan, "AUPR": np.nan}
    if len(y) and np.unique(y).size == 2:
        out["AUROC"] = float(roc_auc_score(y, p))
    if len(y) and np.sum(y) > 0:
        out["AUPR"] = float(average_precision_score(y, p))
    return out

def chain_bootstrap_ci(chains, n_bootstraps=2000, seed=42):
    """Sample chains with replacement, keeping all residues within each chain."""
    rng = np.random.default_rng(seed)
    ids = list(chains)
    aucs, auprs = [], []
    if len(ids) < 2:
        return {
            "AUROC_CI95_low": None, "AUROC_CI95_high": None,
            "AUPR_CI95_low": None, "AUPR_CI95_high": None,
            "valid_AUROC_bootstraps": 0, "valid_AUPR_bootstraps": 0,
        }

    for _ in range(n_bootstraps):
        draw = rng.integers(0, len(ids), size=len(ids))
        yb = np.concatenate([chains[ids[i]]["y"] for i in draw])
        pb = np.concatenate([chains[ids[i]]["p"] for i in draw])
        if np.unique(yb).size == 2:
            aucs.append(roc_auc_score(yb, pb))
        if yb.sum() > 0:
            auprs.append(average_precision_score(yb, pb))

    def bounds(vals):
        if not vals:
            return None, None
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)

    auc_lo, auc_hi = bounds(aucs)
    aupr_lo, aupr_hi = bounds(auprs)
    return {
        "AUROC_CI95_low": auc_lo, "AUROC_CI95_high": auc_hi,
        "AUPR_CI95_low": aupr_lo, "AUPR_CI95_high": aupr_hi,
        "valid_AUROC_bootstraps": len(aucs),
        "valid_AUPR_bootstraps": len(auprs),
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nCheckpoint: {CHECKPOINT}")
    if not Path(CHECKPOINT).exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    split_ids = [s.strip() for s in Path(TEST_SPLIT).read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(split_ids) != len(set(split_ids)):
        raise RuntimeError("Duplicate IDs in test split.")

    dataset = AlloDataset(GRAPH_DIR, EMBEDDINGS, TEST_SPLIT)
    ids = list(dataset.valid_ids)
    if len(ids) != len(split_ids):
        missing = sorted(set(split_ids) - set(ids))
        raise RuntimeError(f"Loaded {len(ids)} graphs, expected {len(split_ids)}. Missing: {missing[:20]}")
    if EXPECTED_TEST_CHAINS and len(ids) != EXPECTED_TEST_CHAINS:
        raise RuntimeError(f"Expected {EXPECTED_TEST_CHAINS} test chains; loaded {len(ids)}.")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = AlloGNN().to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    # Each record stores all residues for one test chain.
    chains = {}
    offset = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(loader, 1):
            batch = batch.to(device)
            output = model(batch)
            pred = to_numpy(output["binding"])
            y = to_numpy(batch.y).astype(np.int64)
            node_types = to_numpy(batch.site_type[batch.batch]).astype(np.int64)
            ptr = to_numpy(batch.ptr).astype(int)

            if not (len(pred) == len(y) == len(node_types)):
                raise RuntimeError(f"Prediction/label/type length mismatch in batch {batch_i}")

            for j in range(len(ptr) - 1):
                start, end = ptr[j], ptr[j + 1]
                chain_id = ids[offset + j]
                chain_y, chain_p = y[start:end], pred[start:end].astype(float)
                chain_types = np.unique(node_types[start:end])
                if len(chain_types) != 1:
                    raise RuntimeError(f"{chain_id}: inconsistent node site types {chain_types}")
                site_type = int(chain_types[0])
                if site_type not in TYPE_NAMES:
                    raise RuntimeError(f"{chain_id}: unrecognized site_type={site_type}")
                if len(chain_y) == 0:
                    raise RuntimeError(f"{chain_id}: empty graph")
                chains[chain_id] = {"site_type_id": site_type, "y": chain_y, "p": chain_p}
            offset += len(ptr) - 1
            if batch_i % 10 == 0 or offset == len(ids):
                print(f"Processed {offset}/{len(ids)} chains")

    if offset != len(ids) or len(chains) != len(ids):
        raise RuntimeError(f"Processed {offset} graphs but expected {len(ids)}")

    groups = {"Overall (pooled)": chains}
    for tid, name in TYPE_NAMES.items():
        groups[name] = {cid: r for cid, r in chains.items() if r["site_type_id"] == tid}

    rows, details = [], {}
    for i, (name, subset) in enumerate(groups.items()):
        if not subset:
            continue
        y = np.concatenate([r["y"] for r in subset.values()])
        p = np.concatenate([r["p"] for r in subset.values()])
        point = metrics(y, p)
        ci = chain_bootstrap_ci(subset, N_BOOTSTRAPS, SEED + i)
        n_res = len(y)
        n_pos = int(y.sum())
        row = {
            "site_type": name,
            "n_chains": len(subset),
            "n_residues": n_res,
            "n_positive_residues": n_pos,
            "n_negative_residues": int(n_res - n_pos),
            "positive_prevalence": float(n_pos / n_res),
            "AUROC": point["AUROC"],
            "AUROC_CI95_low": ci["AUROC_CI95_low"],
            "AUROC_CI95_high": ci["AUROC_CI95_high"],
            "AUPR": point["AUPR"],
            "AUPR_CI95_low": ci["AUPR_CI95_low"],
            "AUPR_CI95_high": ci["AUPR_CI95_high"],
            "valid_AUROC_bootstraps": ci["valid_AUROC_bootstraps"],
            "valid_AUPR_bootstraps": ci["valid_AUPR_bootstraps"],
        }
        rows.append(row)
        details[name] = {
            **row,
            "checkpoint": CHECKPOINT,
            "bootstrap_method": "Percentile cluster bootstrap; sample chains with replacement and retain all residues per sampled chain",
            "n_bootstraps_requested": N_BOOTSTRAPS,
            "seed": SEED + i,
        }

    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / "test_composition_and_bootstrap_ci.csv"
    summary.to_csv(summary_path, index=False)

    chain_rows = []
    for cid, r in chains.items():
        yy = r["y"]
        chain_rows.append({
            "chain_id": cid,
            "site_type_id": r["site_type_id"],
            "site_type": TYPE_NAMES[r["site_type_id"]],
            "n_residues": len(yy),
            "n_positive_residues": int(yy.sum()),
            "n_negative_residues": int(len(yy) - yy.sum()),
            "positive_prevalence": float(yy.mean()),
        })
    chain_path = OUT_DIR / "test_chain_composition_audit.csv"
    pd.DataFrame(chain_rows).to_csv(chain_path, index=False)

    json_path = OUT_DIR / "test_composition_and_bootstrap_ci.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": CHECKPOINT,
            "test_split": TEST_SPLIT,
            "n_test_chains": len(chains),
            "n_bootstraps": N_BOOTSTRAPS,
            "seed": SEED,
            "bootstrap_method": "Percentile cluster bootstrap by chain",
            "results": details,
        }, f, indent=2)

    cols = ["site_type", "n_chains", "n_residues", "n_positive_residues",
            "positive_prevalence", "AUROC", "AUROC_CI95_low", "AUROC_CI95_high",
            "AUPR", "AUPR_CI95_low", "AUPR_CI95_high"]
    print("\n=== TEST-SET COMPOSITION + 95% CHAIN-BOOTSTRAP CIs ===")
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {chain_path}")
    print(f"  {json_path}")
    print("\nCheck that CHECKPOINT exactly matches the checkpoint used for the manuscript results.")

if __name__ == "__main__":
    main()
