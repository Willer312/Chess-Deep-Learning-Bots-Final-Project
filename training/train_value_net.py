import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from bots.value_search import ValueMLP


def train_value_net(
    data_path: str = "data/train_dataset.npz",
    out_path: str = "weights/value_mlp.pt",
    epochs: int = 15,
    batch_size: int = 512,
    lr: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training ValueMLP on {device}")

    raw = np.load(data_path)
    planes = raw["planes"].astype(np.float32)
    flat_pieces = planes[:, :12].reshape(len(planes), -1)
    turn_flag = planes[:, 16, 0, 0:1]
    features = torch.from_numpy(np.concatenate([flat_pieces, turn_flag], axis=1))
    outcomes = torch.from_numpy(raw["outcomes"]).float()

    dataset = TensorDataset(features, outcomes)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = ValueMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)

        scheduler.step()
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        sign_correct = 0
        decisive_total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * x_batch.size(0)

                decisive_mask = y_batch != 0.0
                if decisive_mask.any():
                    sign_correct += ((preds[decisive_mask] > 0) == (y_batch[decisive_mask] > 0)).sum().item()
                    decisive_total += int(decisive_mask.sum().item())

        val_loss /= val_size
        sign_acc = (sign_correct / decisive_total * 100.0) if decisive_total > 0 else 0.0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | "
            f"Outcome Sign Acc: {sign_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_path)

    print(f"Saved best ValueMLP (Val MSE: {best_val_loss:.4f}) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train_dataset.npz")
    parser.add_argument("--out", type=str, default="weights/value_mlp.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_value_net(
        data_path=args.data,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )