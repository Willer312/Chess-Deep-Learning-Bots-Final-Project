import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from bots.policy_cnn import PolicyResNet


def train_policy_cnn(
    data_path: str = "data/train_dataset.npz",
    out_path: str = "weights/policy_resnet.pt",
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PolicyResNet on {device}")

    raw = np.load(data_path)
    planes = torch.from_numpy(raw["planes"]).float()
    moves = torch.from_numpy(raw["moves"]).long()

    dataset = TensorDataset(planes, moves)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = PolicyResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)

        scheduler.step()
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        top1_correct = 0
        top3_correct = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * x_batch.size(0)

                preds = logits.argmax(dim=-1)
                top1_correct += (preds == y_batch).sum().item()

                top3_preds = logits.topk(3, dim=-1).indices
                top3_correct += (top3_preds == y_batch.unsqueeze(1)).any(dim=1).sum().item()

        val_loss /= val_size
        top1_acc = (top1_correct / val_size) * 100.0
        top3_acc = (top3_correct / val_size) * 100.0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Top-1 Acc: {top1_acc:.2f}% | Top-3 Acc: {top3_acc:.2f}%"
        )

        if top1_acc > best_val_acc:
            best_val_acc = top1_acc
            torch.save(model.state_dict(), out_path)

    print(f"Saved best model (Val Top-1: {best_val_acc:.2f}%) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train_dataset.npz")
    parser.add_argument("--out", type=str, default="weights/policy_resnet.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_policy_cnn(
        data_path=args.data,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )