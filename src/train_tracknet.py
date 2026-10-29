"""Train TrackNet for tennis ball tracking.

This is a simplified training loop. Full implementation should follow
TrackNetV3 architecture (U-Net with 3-frame input → heatmap).
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import TrackNetDataset


def train(data_root: str, epochs: int = 30, batch: int = 4, lr: float = 1e-3, device: str = "cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    train_ds = TrackNetDataset(data_root, split="train")
    val_ds = TrackNetDataset(data_root, split="val")
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2)

    # Placeholder model — replace with real TrackNet architecture
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 1, 3, padding=1),
        nn.Sigmoid(),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "tracknet.pt")
    print("Saved tracknet.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(args.data_root, args.epochs, args.batch, args.lr, args.device)


if __name__ == "__main__":
    main()
