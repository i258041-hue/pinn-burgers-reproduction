import torch
from src.model import PINN

model = PINN()
model.load_state_dict(torch.load("checkpoints/pinn_weights.pt"))

print("Inference completed.")
