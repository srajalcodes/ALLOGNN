"""
Train and evaluate a sequence-only AlloGNN baseline on the existing gene-cold split.
Uses train_ids.txt for training, val_ids.txt for checkpoint selection, and
test_ids.txt for one final held-out evaluation. Does not modify existing files.
"""
import os
os.environ["PYG_HOME"] = "data/pyg_cache"
os.environ["TORCH_GEOMETRIC_HOME"] = "data/pyg_cache"
os.environ["TORCH_HOME"] = "data/torch_cache"

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from src.data.dataset import AlloDataset
from src.training.trainer import AlloGNNTrainer
from src.models.allognn import AlloGNN

SEED = 42
EXPECTED_TEST_CHAINS = 812
EPOCHS = 20
GRAPH_DIR = "data/processed/graphs"
EMBEDDINGS = "data/processed/embeddings/esm2_embeddings.h5"
SPLIT_DIR = Path("data/processed/splits")
OUTPUT_DIR = Path("output/sequence_only_baseline")
CHECKPOINT_PATH = OUTPUT_DIR / "best_sequence_only_gene_cold.pt"
RESULTS_PATH = OUTPUT_DIR / "sequence_only_gene_cold_results.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SeqOnlyAlloGNN(AlloGNN):
    """Sequence-only architecture matching SeqOnlyAlloGNN in run_ablations.py."""
    def forward(self, data):
        seq_emb = self.seq_proj(data.x_seq)
        struct_emb = torch.zeros(
            (data.num_nodes, 128), device=seq_emb.device, dtype=seq_emb.dtype
        )
        surf_emb = torch.zeros(
            (data.num_nodes, 32), device=seq_emb.device, dtype=seq_emb.dtype
        )
        fused, _ = self.fusion(seq_emb, struct_emb, surf_emb)
        shared = self.shared(fused)
        return {
            "binding": torch.sigmoid(self.binding_head(shared).squeeze(-1)),
            "type": self.type_head(shared),
            "gate_weights": torch.zeros(
                (data.num_nodes, 3), device=seq_emb.device, dtype=seq_emb.dtype
            ),
        }


def load_dataset(split_name):
    split_path = SPLIT_DIR / f"{split_name}_ids.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Required split file not found: {split_path}")
    dataset = AlloDataset(GRAPH_DIR, EMBEDDINGS, str(split_path))
    ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate IDs found in {split_path}.")
    if len(dataset.valid_ids) != len(ids):
        missing = sorted(set(ids) - set(dataset.valid_ids))
        raise RuntimeError(
            f"{split_name} split has IDs without graph files: {len(missing)} missing; "
            f"examples={missing[:10]}. Fix data/split mismatch before evaluation."
        )
    print(f"{split_name}: {len(ids)} IDs in split; {len(dataset)} loadable chains.")
    return dataset, set(dataset.valid_ids)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; seed: {SEED}")

    train_dataset, train_ids = load_dataset("train")
    val_dataset, val_ids = load_dataset("val")
    test_dataset, test_ids = load_dataset("test")

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Train, validation, and test split IDs must be disjoint.")
    if len(test_dataset) != EXPECTED_TEST_CHAINS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_CHAINS} test chains but loaded {len(test_dataset)}. "
            "Stopping to avoid silently evaluating on a different test set."
        )

    print(
        f"Verified split sizes — train: {len(train_dataset)}, "
        f"validation: {len(val_dataset)}, test: {len(test_dataset)}"
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = SeqOnlyAlloGNN().to(device)
    optimizer = AdamW(model.parameters(), lr=0.0008, weight_decay=1.2e-6)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=2)
    cosine = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[2])
    trainer = AlloGNNTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler, device=device,
        pos_weight=8.26, gamma=1.06, alpha=1.0, beta=0.065
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_aupr = float("-inf")
    best_epoch = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, _, _ = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.evaluate(val_loader, epoch)
        val_aupr = val_metrics.get("aupr_type4")
        if val_aupr is None or not np.isfinite(val_aupr):
            raise RuntimeError("Validation Type IV AUPR unavailable; inspect validation labels.")

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} "
            f"| val_TypeIV_AUPR={val_aupr:.6f}"
        )

        if val_aupr > best_val_aupr:
            best_val_aupr = float(val_aupr)
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_class": "SeqOnlyAlloGNN",
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "best_validation_type4_aupr": best_val_aupr,
                    "train_chains": len(train_dataset),
                    "validation_chains": len(val_dataset),
                    "test_chains": len(test_dataset),
                },
                CHECKPOINT_PATH,
            )

    print(f"\nBest epoch selected on validation set: {best_epoch}")
    print(f"Best validation Type IV AUPR: {best_val_aupr:.6f}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = trainer.evaluate(test_loader, best_epoch)
    if "aupr_type4" not in test_metrics:
        raise RuntimeError("Test Type IV AUPR unavailable; inspect test site_type labels.")

    results = {
        "model": "sequence-only AlloGNN ablation",
        "seed": SEED,
        "best_epoch_selected_on_validation": best_epoch,
        "best_validation_type4_aupr": best_val_aupr,
        "train_chains": len(train_dataset),
        "validation_chains": len(val_dataset),
        "test_chains": len(test_dataset),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "metric_definition": (
            "Pooled residue-level metrics from AlloGNNTrainer.evaluate(); Type IV "
            "is site_type == 4; AUPR uses average_precision_score over all Type IV "
            "test residues."
        ),
        "checkpoint": str(CHECKPOINT_PATH),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nFINAL HELD-OUT TEST RESULT")
    print(f"Sequence-only Type IV AUPR: {test_metrics['aupr_type4']:.6f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Results: {RESULTS_PATH}")
    print("The test set was not used for checkpoint selection.")


if __name__ == "__main__":
    main()
