import os
os.environ['PYG_HOME'] = 'data/pyg_cache'
os.environ['TORCH_GEOMETRIC_HOME'] = 'data/pyg_cache'
os.environ['TORCH_HOME'] = 'data/torch_cache'

import csv
import datetime
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.dataset import AlloDataset
from src.models.allognn import AlloGNN
from src.training.trainer import AlloGNNTrainer

TABLE_PATH = Path("results/tables/same_split_baselines.csv")
TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)


class SeqOnlyAlloGNN(AlloGNN):
    """Sequence-only control: structure + surface streams zeroed out."""
    def forward(self, data):
        seq_emb = self.seq_proj(data.x_seq)
        struct_emb = torch.zeros((data.num_nodes, 128), device=seq_emb.device)
        surf_emb = torch.zeros((data.num_nodes, 32), device=seq_emb.device)
        fused, _ = self.fusion(seq_emb, struct_emb, surf_emb)
        shared = self.shared(fused)
        return {
            'binding': torch.sigmoid(self.binding_head(shared).squeeze(-1)),
            'type': self.type_head(shared),
            'gate_weights': torch.zeros((data.num_nodes, 3), device=seq_emb.device)
        }


def append_result_row(row: dict):
    file_exists = TABLE_PATH.exists()
    with open(TABLE_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Device: {device} | Run: {timestamp}")

    # Same loaders/splits as the main model
    train_dataset = AlloDataset("data/processed/graphs", "data/processed/embeddings/esm2_embeddings.h5", "data/processed/splits/train_ids.txt")
    val_dataset = AlloDataset("data/processed/graphs", "data/processed/embeddings/esm2_embeddings.h5", "data/processed/splits/val_ids.txt")
    test_dataset = AlloDataset("data/processed/graphs", "data/processed/embeddings/esm2_embeddings.h5", "data/processed/splits/test_ids.txt")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Same Optuna hyperparameters as the main model (run_log_20260811_104436.json)
    opt_lr = 0.0008070161049782942
    opt_wd = 1.2243130080634888e-06
    opt_gamma = 1.0631476498675683
    opt_pos_weight = 8.264681596629156
    opt_beta = 0.0654674015416209

    model = SeqOnlyAlloGNN().to(device)
    optimizer = AdamW(model.parameters(), lr=opt_lr, weight_decay=opt_wd)
    max_epochs = 75
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=70, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    trainer = AlloGNNTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler, device=device,
        pos_weight=opt_pos_weight, gamma=opt_gamma, alpha=1.0, beta=opt_beta
    )

    patience, epochs_no_improve, best_t4 = 15, 0, 0.0
    ckpt_path = Path("output/checkpoints") / f"best_seqonly_{timestamp}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        loss, bce, ce = trainer.train_epoch(train_loader, epoch)
        metrics = trainer.evaluate(val_loader, epoch)
        t4 = metrics.get('aupr_type4', 0.0)
        print(f"Ep {epoch:02d}/{max_epochs} | Loss {loss:.4f} | T4_AUPR {t4:.4f}")

        if t4 > best_t4:
            best_t4 = t4
            torch.save(model.state_dict(), ckpt_path)
            epochs_no_improve = 0
            print(f"   -> saved best (T4 AUPR {t4:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stop at epoch {epoch}.")
                break

    # Evaluate best checkpoint on the real 812-chain test set
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    all_preds, all_labels, all_types = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            all_preds.append(out['binding'].cpu())
            all_labels.append(batch.y.cpu())
            all_types.append(batch.site_type[batch.batch].cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    types = torch.cat(all_types).numpy()

    tmap = {0: 'Type_I', 1: 'Type_I.5', 2: 'Type_II', 3: 'Type_III', 4: 'Type_IV'}
    print("\n=== ESM2-ONLY, SAME SPLIT, TEST RESULTS ===")
    for tid, tname in tmap.items():
        mask = (types == tid)
        if mask.sum() > 0 and labels[mask].sum() > 0:
            auroc = roc_auc_score(labels[mask], preds[mask])
            aupr = average_precision_score(labels[mask], preds[mask])
            print(f"{tname:<10} AUROC={auroc:.3f} AUPR={aupr:.3f}")
            append_result_row({
                "model": "ESM2-only (ours, same split)",
                "checkpoint": str(ckpt_path),
                "target_site": tname,
                "n_residues": int(mask.sum()),
                "n_positive": int(labels[mask].sum()),
                "AUROC": round(float(auroc), 4),
                "AUPR": round(float(aupr), 4),
                "timestamp": timestamp,
            })

    print(f"\nResults appended to: {TABLE_PATH}")


if __name__ == "__main__":
    main()