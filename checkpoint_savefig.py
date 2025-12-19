import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
import os
import time
from torch import nn, Tensor

# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: 
        return torch.sigmoid(x) * x

# Model class
class MLP(nn.Module):
    def __init__(self, input_dim: int = 1, time_dim: int = 1, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(input_dim)

        self.main = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim),
            Swish(),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    Swish(),
                ) for _ in range(num_layers - 2)
            ],
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        # x = self.layer_norm(x)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)
        
        return output.reshape(*sz)

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t, **extras)

def print_gpu_memory(epoch=None, step=None):
    """Prints the current GPU memory usage.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        
        prefix = f"| Epoch {epoch}" if epoch is not None else "| Memory"
        if step is not None: prefix += f", Step {step}"
        
        print(f"{prefix} | GPU Alloc: {allocated:.1f}MB | Reserv: {reserved:.1f}MB | Max: {max_allocated:.1f}MB")
        
        torch.cuda.reset_peak_memory_stats()
    else:
        print("GPU not available.")

def get_elapsed_time(start_time):
    elapsed = time.time() - start_time
    return str(datetime.timedelta(seconds=int(elapsed)))

global_start_time = time.time()

EPOCH_TO_LOAD = 10000
checkpoint_path = f'checkpoints_LRfixed/model_epoch_{EPOCH_TO_LOAD}.pth'

# 2. Inizializza il modello (stessa architettura del training)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vf_eval = MLP(input_dim=1, time_dim=1, hidden_dim=128, num_layers=6).to(device)

# 3. Carica i pesi dal file .pth
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vf_eval.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint at epoch {EPOCH_TO_LOAD} loaded successfully.")
else:
    print(f"Error: the file {checkpoint_path} does not exist.")

# Using the model for sampling
vf_eval.eval() 
wrapped_vf = WrappedModel(vf_eval)
solver = ODESolver(velocity_model=wrapped_vf)

source_dist = torch.distributions.Uniform(-3.0, 3.0)
n_samples = 100000
T = torch.linspace(0, 1, 200).to(device)
x_init = source_dist.sample((n_samples,)).unsqueeze(1).to(device)

with torch.no_grad():
    sol = solver.sample(
        time_grid=T, 
        x_init=x_init, 
        method='dopri5',
        step_size=None,
        return_intermediates=True
    )
sol = sol.cpu().numpy().squeeze()

realistic_dataset = np.load("deltaphimoredata.npy")
dphi = realistic_dataset[:, 2]

# Plot the evolution of the distribution at selected time points
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
time_indices = np.linspace(0, len(T)-1, 10, dtype=int)

for idx, t_idx in enumerate(time_indices):
    axes[idx].hist(sol[t_idx], bins=100, density=True, alpha=0.7, color='cyan')
    axes[idx].set_title(f't = {T[t_idx].cpu().item():.2f}')
    axes[idx].set_xlabel('dphi')
    axes[idx].set_ylabel('Density')
    axes[idx].grid(alpha=0.3)
    # Usa i limiti del dataset originale per coerenza visiva
    axes[idx].set_xlim(dphi.min() - 0.5, dphi.max() + 0.5)

plt.suptitle(f'Distribution Evolution - Epoch {EPOCH_TO_LOAD}', fontsize=14, y=1.00)
plt.tight_layout()
plt.show() 
plt.savefig(f'evolution_epoch_{EPOCH_TO_LOAD}_LRfixed.png', dpi=300)

# Target vs Generated at final time
plt.figure(figsize=(10, 6))

plt.hist(dphi, bins=100, density=True, alpha=0.5, color='red', label='Target (Real)')
plt.hist(sol[-1, :], bins=100, density=True, alpha=0.7, color='cyan', label=f'Generated (Epoch {EPOCH_TO_LOAD})')
plt.title(f'Comparison at Epoch {EPOCH_TO_LOAD}')
plt.legend()
plt.savefig(f'comparison_epoch_{EPOCH_TO_LOAD}_LRfixed.png', dpi=300)
plt.close()