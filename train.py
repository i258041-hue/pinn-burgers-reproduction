import torch
from src.model import PINN

print("Training started...")

model = PINN()

# Placeholder training loop
for epoch in range(5):
    print(f"Epoch {epoch+1}/5 completed")

torch.save(model.state_dict(), "checkpoints/pinn_weights.pt")

print("Training completed.")
