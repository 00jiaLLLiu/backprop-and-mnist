# Backprop from Scratch & MNIST Classifiers

Two deep-learning fundamentals exercises (CSI 5340 Deep Learning, Homework 2): hand-derived backpropagation verified against autograd, and an empirical study of BatchNorm vs Dropout on MNIST.

## Question 1 — Manual Forward/Backward Propagation

A 5-layer computation graph with matrix multiplies, a sigmoid, and an addition:

```
y = Ax → u = σ(y);  v = Bx → z = u + v → w = Cz → L = ||w||
```

- Each layer implements `forward`/`backward` by hand (`Question1/calcu_cmp.py`); every gradient numerically verified against PyTorch autograd to prove the derivation correct.
- Training curves recorded for a single input (`loss_oneX.py`, 100 epochs) and for N=15 inputs (`loss_NX.py`, 150 epochs); loss plots in `Question1/`.

## Question 2 — Softmax, MLP, CNN on MNIST

Three classifiers trained on MNIST (auto-downloaded via torchvision, SGD lr=0.01), each in 6 structural variants — base, Dropout(0.2/0.5/0.8), BatchNorm, BatchNorm+Dropout — for **18 full training runs** with per-step loss/accuracy logging:

- `Question2/SoftmaxRegression/` — softmax regression (cross-entropy)
- `Question2/MLP/` — 784-512-256-128-64-10 fully-connected network
- `Question2/CNN/` — 2-layer convolutional network
- Per-variant training logs in `MLP.csv` / `softmax.csv` / `CNN.csv`; loss/accuracy plots in `Question2/loss_acc/` (`plot.py` regenerates them).

## Key Findings

- **BatchNorm is the clear winner everywhere**: softmax >90% accuracy with loss <1.6; MLP hits 99% / 0.002 loss in 5 epochs; CNN reaches 99.6% / 0.01 loss. It speeds convergence *and* improves the final result.
- **Dropout hurts more than it helps here**: dropout 0.8 is the worst in every model (softmax collapses to 35% accuracy / 2.18 loss). Even 0.2 is barely distinguishable from no Dropout — on MNIST with these architectures, randomly killing 80% of activations just destroys signal.
- **BN + Dropout together = 1+1<2**: the combination never beats BatchNorm alone. Two regularizers don't stack; pick the one that works.
- **Same settings, CNN ≈ MLP > softmax** — unsurprising, but the comparison also shows the MLP predicts fastest (~0.001s) while being the only model that actually needs training; the learning-free baseline from the texture project generalizes fine without any.

## How to Run

```bash
pip install -r requirements.txt
cd Question2/MLP && python MLP.py        # one variant; see _bn/_do/_noboth for others
# MNIST downloads automatically on first run
cd ../loss_acc && python plot.py         # regenerate comparison plots
```

## Repository Structure

```
├── Question1/              # manual backprop: layers, gradient check, loss curves
│   ├── calcu_cmp.py
│   ├── loss_oneX.py
│   └── loss_NX.py
├── Question2/              # softmax / MLP / CNN × {base, dropout, BN, both}
│   ├── SoftmaxRegression/
│   ├── MLP/
│   ├── CNN/
│   ├── *.csv               # training logs per variant
│   └── loss_acc/           # comparison plots + plot.py
└── report/
    └── Assignment2.pdf
```
