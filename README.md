# Physics-Informed Neural Networks (PINNs) — Reproduction & Extension

Reproduction and experimental extension of the seminal PINN framework applied to the **1D viscous Burgers' equation**, following:

> Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707.

---

## Tasks

| # | Task | Summary |
|---|------|---------|
| 1 | **Paper Understanding** | Mathematical foundations — PINN architecture, composite loss formulation, automatic differentiation via GradientTape |
| 2 | **Reproduction** | Baseline TensorFlow 2.x implementation of the continuous-time PINN for Burgers' equation |
| 3 | **Extension** | Hybrid Adam + L-BFGS optimization, weighted boundary loss (5×), hard boundary enforcement, increased collocation points |

---

## Results

| Configuration | Relative L² Error | Change |
|---|---|---|
| Raissi et al. (2019) — original | ≈ 0.010 | reference |
| Exp 1 — Baseline PINN | 0.578 | — |
| Exp 2 — Hybrid Optimizer | 0.511 | −11.6% |
| Exp 3 — Full Improved | 0.512 | ≈ no change |

The reproduction gap (~51×) is attributed to optimization non-convexity, spectral bias, and constrained compute vs. the original hardware environment.

---

## Repository Structure

```
├── train.py               # Training script (all experiments)
├── inference.py           # Load checkpoint and run inference
├── config.yaml            # Hyperparameters and experiment settings
├── requirements.txt
├── data/
│   └── sample_data.csv    # Sample collocation/boundary points
├── notebooks/
│   └── 01_inference_demo.ipynb
├── src/
│   ├── model.py           # PINN network architecture
│   ├── dataset.py         # Collocation point sampling
│   └── utils.py           # Loss functions, metrics, plotting
├── results/
│   ├── baseline_metrics.json
│   ├── improved_metrics.json
│   └── training_log.csv
└── checkpoints/           # Saved model weights
```

---

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Inference

```bash
python inference.py --checkpoint checkpoints/best_model
```

---

## Key Findings

- Composite PINN loss can converge to low values while the PDE solution remains inaccurate — training loss is not a reliable proxy for solution quality
- L-BFGS provides modest improvement but inherits the quality of Adam pre-training
- Spectral bias makes sharp shock fronts (low ν) particularly difficult to capture with standard feedforward architectures

---

## Citation

```bibtex
@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E},
  journal={Journal of Computational Physics},
  volume={378},
  pages={686--707},
  year={2019}
}
```
