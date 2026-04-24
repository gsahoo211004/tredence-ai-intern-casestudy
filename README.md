# Tredence AI Engineering Intern — Case Study

## The Self-Pruning Neural Network

A feed-forward network that learns to prune itself during training using
learnable gate parameters and L1 sparsity regularization on CIFAR-10.

## How to Run
pip install torch torchvision matplotlib numpy
python self_pruning_network.py

## Files
- `self_pruning_network.py` — Full implementation (Parts 1, 2, 3)
- `report.md` — Analysis and results
- `gate_distribution.png` — Gate value plots (generated on run)