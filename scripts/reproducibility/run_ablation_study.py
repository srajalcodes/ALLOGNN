"""Train and evaluate six ALLOGNN ablations on the fixed gene-cold split."""
import os
os.environ["PYG_HOME"] = "data/pyg_cache"
os.environ["TORCH_GEOMETRIC_HOME"] = "data/pyg_cache"
os.environ["TORCH_HOME"] = "data/torch_cache"

import csv, json, logging, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from src.data.dataset import AlloDataset
from src.training.trainer import AlloGNNTrainer
from src.models.allognn import AlloGNN

SEED = 42
EPOCHS = 75
PATIENCE = 15
EXPECTED_TEST_CHAINS = 812
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1.2e-6
POS_WEIGHT = 8.26
FOCAL_GAMMA = 1.06
ALPHA = 1.0
BETA = 0.065
GRAPH_DIR = "data/processed/graphs"
EMBEDDINGS = "data/processed/embeddings/esm2_embeddings.h5"

SPLIT_DIR = Path("data/processed/splits")
OUTPUT_DIR = Path("output/ablation_full_gene_cold")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(split_name):
    split_path = SPLIT_DIR / f"{split_name}_ids.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    ids = [x.strip() for x in split_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate IDs in {split_path}")
    dataset = AlloDataset(GRAPH_DIR, EMBEDDINGS, str(split_path))
    valid_ids = set(getattr(dataset, "valid_ids", []))
    if valid_ids and len(valid_ids) != len(ids):
        raise RuntimeError(f"{split_name}: split IDs and loaded graph IDs do not match")
    if len(dataset) != len(ids):
        raise RuntimeError(f"{split_name}: expected {len(ids)} chains, loaded {len(dataset)}")
    return dataset, set(ids)


class ModalityAblationAlloGNN(AlloGNN):
    def __init__(self, active_modalities):
        super().__init__()
        self.active_modalities = tuple(active_modalities)

    def forward(self, data):
        seq_emb = self.seq_proj(data.x_seq)
        if "structure" in self.active_modalities:
            struct_input = seq_emb if "sequence" in self.active_modalities else torch.zeros_like(seq_emb)
            struct_emb = self.struct_gat(struct_input, data.edge_index, data.edge_attr)
        else:
            struct_emb = torch.zeros((data.num_nodes, 128), device=seq_emb.device, dtype=seq_emb.dtype)
        surf_emb = self.surf_proj(data.surf_x) if "surface" in self.active_modalities else torch.zeros(
            (data.num_nodes, 32), device=seq_emb.device, dtype=seq_emb.dtype
        )
        if "sequence" not in self.active_modalities:
            seq_emb = torch.zeros_like(seq_emb)

        fusion = self.fusion
        streams = [
            F.gelu(fusion.seq_proj(seq_emb)),
            F.gelu(fusion.struct_proj(struct_emb)),
            F.gelu(fusion.surf_proj(surf_emb)),
        ]
        names = ("sequence", "structure", "surface")
        active = [name in self.active_modalities for name in names]
        streams = [v if keep else torch.zeros_like(v) for v, keep in zip(streams, active)]
        logits = fusion.gate[:-1](torch.cat(streams, dim=-1))
        active_mask = torch.tensor(active, device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(~active_mask.unsqueeze(0), float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        fused_stream = weights[:, 0:1] * streams[0] + weights[:, 1:2] * streams[1] + weights[:, 2:3] * streams[2]
        fused = fusion.norm(fusion.out_proj(fused_stream))
        shared = self.shared(fused)
        return {
            "binding": torch.sigmoid(self.binding_head(shared).squeeze(-1)),
            "type": self.type_head(shared),
            "gate_weights": weights,
        }


class NoGatingAlloGNN(AlloGNN):
    def __init__(self):
        super().__init__()
        self.naive_fusion = nn.Sequential(
            nn.Linear(256 + 128 + 32, 256), nn.LayerNorm(256), nn.GELU()
        )

    def forward(self, data):
        seq_emb = self.seq_proj(data.x_seq)
        struct_emb = self.struct_gat(seq_emb, data.edge_index, data.edge_attr)
        surf_emb = self.surf_proj(data.surf_x)
        fused = self.naive_fusion(torch.cat([seq_emb, struct_emb, surf_emb], dim=-1))
        shared = self.shared(fused)
        return {
            "binding": torch.sigmoid(self.binding_head(shared).squeeze(-1)),
            "type": self.type_head(shared),
            "gate_weights": torch.zeros((data.num_nodes, 3), device=seq_emb.device, dtype=seq_emb.dtype),
        }


def experiments():
    return [
        ("full_allognn", "Full ALLOGNN", AlloGNN, BETA),
        ("sequence_only", "Sequence-only", lambda: ModalityAblationAlloGNN(("sequence",)), BETA),
        ("structure_surface", "Structure + surface", lambda: ModalityAblationAlloGNN(("structure", "surface")), BETA),
        ("sequence_structure", "Sequence + structure", lambda: ModalityAblationAlloGNN(("sequence", "structure")), BETA),
        ("no_gating", "No gating (concatenation)", NoGatingAlloGNN, BETA),
        ("without_auxiliary_loss", "Without auxiliary loss", AlloGNN, 0.0),
    ]


def make_trainer(model, device, beta):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=2)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[2])
    return AlloGNNTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler, device=device,
        pos_weight=POS_WEIGHT, gamma=FOCAL_GAMMA, alpha=ALPHA, beta=beta
    )

def clean_metrics(metrics):
    return {k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(float(v))}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT_DIR / "ablation_run.log", encoding="utf-8"),
            logging.StreamHandler(),
        ], force=True,
    )
    log = logging.getLogger("allognn_ablation")
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device=%s; seed=%d; epochs=%d", device, SEED, EPOCHS)

    train_ds, train_ids = load_dataset("train")
    val_ds, val_ids = load_dataset("val")
    test_ds, test_ids = load_dataset("test")
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Train, validation, and test IDs must be disjoint")
    if len(test_ds) != EXPECTED_TEST_CHAINS:
        raise RuntimeError(f"Expected {EXPECTED_TEST_CHAINS} test chains; loaded {len(test_ds)}")

    log.info("Verified split sizes: train=%d, validation=%d, test=%d", len(train_ds), len(val_ds), len(test_ds))
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    summary = []

    for run_id, name, model_factory, beta in experiments():
        set_seed(SEED)
        run_dir = OUTPUT_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "best_validation_checkpoint.pt"
        history_path = run_dir / "epoch_history.json"
        result_path = run_dir / "final_test_results.json"
        log.info("Starting %s; auxiliary-loss beta=%.3f", name, beta)

        model = model_factory().to(device)
        trainer = make_trainer(model, device, beta)
        best_val_aupr, best_epoch = float("-inf"), None
        epochs_without_improvement = 0
        epochs_run = 0
        history = []

        for epoch in range(1, EPOCHS + 1):
            epochs_run = epoch
            train_loss, binding_loss, type_loss = trainer.train_epoch(train_loader, epoch)
            val_metrics = trainer.evaluate(val_loader, epoch)
            val_aupr = val_metrics.get("aupr_type4")
            if val_aupr is None or not np.isfinite(float(val_aupr)):
                raise RuntimeError(f"{name}: validation Type IV AUPR unavailable at epoch {epoch}")
            history.append({
                "epoch": epoch,
                "train_total_loss": float(train_loss),
                "train_binding_loss": float(binding_loss),
                "train_auxiliary_loss": float(type_loss),
                "validation_metrics": clean_metrics(val_metrics),
            })
            log.info("%s | epoch %02d/%02d | train loss=%.5f | validation Type IV AUPR=%.6f",
                     name, epoch, EPOCHS, train_loss, val_aupr)
            if float(val_aupr) > best_val_aupr:
                best_val_aupr, best_epoch = float(val_aupr), epoch
                torch.save({
                    "model_state_dict": model.state_dict(), "experiment_id": run_id,
                    "experiment_name": name, "seed": SEED, "best_epoch": best_epoch,
                    "best_validation_type4_aupr": best_val_aupr, "beta": beta,
                    "train_chains": len(train_ds), "validation_chains": len(val_ds),
                    "test_chains": len(test_ds),
                }, checkpoint_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            if epochs_without_improvement >= PATIENCE:
                log.info("%s | early stopping at epoch %d; no validation Type IV AUPR improvement for %d consecutive epochs",
                         name, epoch, PATIENCE)
                break

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = clean_metrics(trainer.evaluate(test_loader, best_epoch))
        result = {
            "experiment_id": run_id, "experiment_name": name, "seed": SEED,
            "epochs_run": epochs_run, "best_epoch_selected_on_validation": best_epoch,
            "best_validation_type4_aupr": best_val_aupr,
            "train_chains": len(train_ds), "validation_chains": len(val_ds),
            "test_chains": len(test_ds), "test_metrics": test_metrics,
            "checkpoint": str(checkpoint_path), "history": str(history_path),
            "auxiliary_loss_beta": beta,
            "metric_protocol": "Best checkpoint selected by validation Type IV AUPR; final metrics computed on the fixed held-out test split.",
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary.append(result)
        log.info("Completed %s; best epoch=%d; test Type IV AUPR=%s", name, best_epoch, test_metrics.get("aupr_type4"))
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (OUTPUT_DIR / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUTPUT_DIR / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["experiment_id", "experiment_name", "seed", "best_epoch_selected_on_validation",
                  "best_validation_type4_aupr", "test_aupr_type4", "test_aupr_overall",
                  "test_auroc_overall", "checkpoint"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in summary:
            metrics = item["test_metrics"]
            writer.writerow({
                "experiment_id": item["experiment_id"], "experiment_name": item["experiment_name"],
                "seed": item["seed"], "best_epoch_selected_on_validation": item["best_epoch_selected_on_validation"],
                "best_validation_type4_aupr": item["best_validation_type4_aupr"],
                "test_aupr_type4": metrics.get("aupr_type4", ""),
                "test_aupr_overall": metrics.get("aupr_overall", ""),
                "test_auroc_overall": metrics.get("auroc_overall", ""),
                "checkpoint": item["checkpoint"],
            })
    log.info("Completed all experiments. Outputs saved under %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
