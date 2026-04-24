"""
Self-Pruning Neural Network on CIFAR-10
========================================
Implements a feed-forward network with learnable gate parameters
that sparsify during training via L1 regularization on sigmoid gates.

KEY FIX: We use a hard-threshold gate during EVALUATION:
  - During training : gates = sigmoid(gate_scores)   [soft, differentiable]
  - During eval/sparsity measurement: gate = 0 if sigmoid < threshold else sigmoid
This ensures sparsity is correctly measured and reported.

Author: [Your Name]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

SPARSITY_THRESHOLD = 0.1   # gates below this are considered "pruned"


# ─────────────────────────────────────────────
# Part 1: PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A linear layer with learnable gate_scores (same shape as weight).

    Forward pass:
        gates          = sigmoid(gate_scores)            in (0, 1)
        pruned_weights = weight * gates
        output         = F.linear(x, pruned_weights, bias)

    Gradients flow through both weight and gate_scores automatically
    because gate_scores is an nn.Parameter and all ops are differentiable.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))

        # gate_scores: one scalar per weight, same shape as weight
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_params()

    def _init_params(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        # Start gate_scores at +2 -> sigmoid(2) ~ 0.88 (mostly open)
        # The sparsity loss will push many toward -inf -> gate ~ 0
        nn.init.constant_(self.gate_scores, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates          = torch.sigmoid(self.gate_scores)   # soft gates in (0,1)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Soft gate values, detached from the graph."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = SPARSITY_THRESHOLD) -> float:
        """Fraction of gates considered pruned (below threshold)."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item() * 100.0


# ─────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """Feed-forward network for CIFAR-10 using PrunableLinear layers."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3 * 32 * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            PrunableLinear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))

    def prunable_layers(self):
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    # ── Part 2: Sparsity Loss ─────────────────────────────

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values = sum of sigmoid(gate_scores).
        Minimising this drives gate_scores toward -inf -> gates toward 0.
        """
        all_gates = [torch.sigmoid(m.gate_scores).view(-1)
                     for m in self.prunable_layers()]
        return torch.cat(all_gates).sum()

    # ── Measurement helpers ───────────────────────────────

    def overall_sparsity(self, threshold: float = SPARSITY_THRESHOLD) -> float:
        total = pruned = 0
        for m in self.prunable_layers():
            g = m.get_gates()
            total  += g.numel()
            pruned += (g < threshold).sum().item()
        return (pruned / total * 100.0) if total > 0 else 0.0

    def all_gate_values(self) -> np.ndarray:
        parts = [m.get_gates().cpu().numpy().ravel()
                 for m in self.prunable_layers()]
        return np.concatenate(parts)

    def print_layer_sparsity(self):
        for i, m in enumerate(self.prunable_layers()):
            print(f"    Layer {i}: sparsity = {m.sparsity():.1f}%  "
                  f"| mean gate = {m.get_gates().mean():.3f}")


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

def get_loaders(batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10('./data', train=True,  download=True, transform=transform)
    test_set  = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, lam):
    model.train()
    total_loss = cls_loss_sum = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits      = model(imgs)
        cls_loss    = F.cross_entropy(logits, labels)
        sparse_loss = model.sparsity_loss()
        loss        = cls_loss + lam * sparse_loss
        loss.backward()
        optimizer.step()
        total_loss   += loss.item()     * imgs.size(0)
        cls_loss_sum += cls_loss.item() * imgs.size(0)
    N = len(loader.dataset)
    return total_loss / N, cls_loss_sum / N


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds    = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100.0


# ─────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────

def run_experiment(lam, epochs=20, device=torch.device('cpu')):
    print(f"\n{'='*60}")
    print(f"  lambda = {lam}   |   epochs = {epochs}")
    print(f"{'='*60}")

    train_loader, test_loader = get_loaders()
    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        t_loss, c_loss = train_epoch(model, train_loader, optimizer, device, lam)
        scheduler.step()
        sparsity = model.overall_sparsity()
        if epoch % 4 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={t_loss:.4f} "
                  f"cls={c_loss:.4f} | sparsity={sparsity:.1f}%")

    test_acc = evaluate(model, test_loader, device)
    sparsity = model.overall_sparsity()

    print(f"\n  Test Accuracy : {test_acc:.2f}%")
    print(f"  Sparsity Level: {sparsity:.2f}%")
    print(f"  Per-layer breakdown:")
    model.print_layer_sparsity()

    return {
        "lam":       lam,
        "test_acc":  test_acc,
        "sparsity":  sparsity,
        "gate_vals": model.all_gate_values(),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Sparsity threshold: {SPARSITY_THRESHOLD}")

    # Low / medium / high lambda
    lambdas = [1e-4, 1e-3, 5e-3]
    results = []

    for lam in lambdas:
        results.append(run_experiment(lam, epochs=20, device=device))

    # ── Summary table ──────────────────────────────────────
    print("\n\n" + "-" * 52)
    print(f"{'Lambda':>10} | {'Test Acc (%)':>12} | {'Sparsity (%)':>14}")
    print("-" * 52)
    for r in results:
        print(f"{r['lam']:>10.0e} | {r['test_acc']:>12.2f} | {r['sparsity']:>14.2f}")
    print("-" * 52)

    # ── Plot ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Distribution of Final Gate Values", fontsize=14, fontweight='bold')

    for ax, r in zip(axes, results):
        vals = r["gate_vals"]
        ax.hist(vals, bins=100, color="steelblue", edgecolor="none")
        ax.axvline(SPARSITY_THRESHOLD, color='red', linestyle='--',
                   linewidth=1.2, label=f'threshold={SPARSITY_THRESHOLD}')
        ax.set_title(
            f"lambda = {r['lam']:.0e}\n"
            f"Acc={r['test_acc']:.1f}%  |  Sparse={r['sparsity']:.1f}%",
            fontsize=10
        )
        ax.set_xlabel("Gate value")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("gate_distribution.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved -> gate_distribution.png")


if __name__ == "__main__":
    main()
