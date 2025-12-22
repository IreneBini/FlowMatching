""" FLOW MATCHING FOR 1D DISTRIBUTION LEARNING (dphi)

DESCRIPTION:
This script implements a Continuous Normalizing Flow (CNF) using the Flow
Matching paradigm to learn and generate a 1-dimensional distribution (dphi).
It maps a simple source distribution (Uniform or Gaussian) to a target
"realistic" distribution derived from a NumPy dataset.

KEY COMPONENTS:
1. Model Architecture: 
   - A Multi-Layer Perceptron (MLP) with Swish activations.
   - Designed to predict the velocity field v(x, t).
2. Probability Path:
   - Uses AffineProbPath with an Optimal Transport (CondOT) scheduler.
   - Defines the linear trajectory between noise (t=0) and data (t=1).
3. Training:
   - Supports both fixed and variable learning rates (StepLR).
   - Implements dynamic batching (switching to full-batch after initial epochs).
   - Tracks GPU memory usage and saves periodic checkpoints.
4. Inference & Evaluation:
   - Solves the learned Neural ODE using the 'dopri5' solver.
   - Generates 100,000 samples and visualizes the evolution of the 
     distribution over time-steps.

USAGE:
Run via CLI with optional arguments:
    python <script_name>.py --name "experiment_name" --variable_lr

OUTPUTS:
- Checkpoints (.pth) saved in experiment-specific folders.
- Training loss plot (log scale).
- Temporal evolution plots of the distribution.
- Final comparison histogram (Generated vs. Target).
"""

import argparse
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
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

def save_evaluation_plots(epoch, model, source_dist, target_data, device, base_dir):
    """Generates samples and saves comparison plots for a specific epoch.
    """
    epoch_dir = os.path.join(base_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Setup solver
    wrapped_model = WrappedModel(model)
    solver = ODESolver(velocity_model=wrapped_model)
    T = torch.linspace(0, 1, 100).to(device)
    
    n_samples = 50000 
    x_init = source_dist.sample((n_samples,)).unsqueeze(1).to(device)
    
    sol = solver.sample(
        time_grid=T,
        x_init=x_init,
        method='dopri5',
        step_size=None,
        return_intermediates=True
        ).cpu().numpy().squeeze()

    # Plot 1: Evoluzione temporale
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    indices = np.linspace(0, len(T)-1, 10, dtype=int)
    for idx, t_idx in enumerate(indices):
        axes[idx].hist(sol[t_idx], bins=100, density=True, alpha=0.7, color='cyan')
        axes[idx].set_title(f't = {T[t_idx].item():.2f}')
        axes[idx].set_xlabel('dphi')
        axes[idx].set_ylabel('Density')
        axes[idx].grid(alpha=0.3)
        axes[idx].set_xlim(target_data.min() - 0.5, target_data.max() + 0.5)
    plt.suptitle('Distribution Evolution (Realistic dphi)', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(epoch_dir, 'dphi_distribution_evolution.png'))
    plt.close()

    final_positions = sol[-1, :]

    # Plot 2: Confronto Finale
    plt.figure(figsize=(8, 5))
    plt.hist(target_data, bins=100, density=True, alpha=0.5, color='red', label='Target (Data)')
    plt.hist(final_positions, bins=100, density=True, alpha=0.7, color='cyan', label='Generated')
    plt.xlabel('dphi', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title(f'Comparison at Epoch {epoch}', fontsize=12)
    plt.savefig(os.path.join(epoch_dir, 'final_comparison.png'))
    plt.close()

def save_loss_plot(losses, epoch, base_dir):
    """Saves the training loss plot up to the current epoch.
    """
    epoch_dir = os.path.join(base_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 4))
    plt.plot(losses, alpha=0.3, color='gray', label='Batch Loss')
    
    # Compute and plot moving average
    if len(losses) > 100:
        ma = np.convolve(losses, np.ones(100)/100, mode='valid')
        plt.plot(ma, linewidth=2, color='blue', label='Moving Average (100)')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Training Loss up to Epoch {epoch}')
    plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(epoch_dir, 'loss_history.png'), dpi=200)
    plt.close()

def create_experiment_summary(checkpoint_dir, name, args, dphi, total_params, hidden_dim, num_layers, epochs, bs, lr, LRfixed, step_size=None, gamma=None):
    summary_path = os.path.join(checkpoint_dir, "experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write("==================\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment Name: {name}\n")
        f.write("-" * 20 + "\n")
        f.write("DATASET INFO:\n")
        f.write(f"Target File:         deltaphimoredata.npy\n")
        f.write(f"Dataset Size:        {len(dphi)} samples\n")
        f.write(f"Target Variable:     dphi\n")
        f.write("-" * 20 + "\n")
        f.write("MODEL ARCHITECTURE:\n")
        f.write(f"Type:                MLP\n")
        f.write(f"Layers:              {num_layers}\n")
        f.write(f"Hidden Dim:          {hidden_dim}\n")
        f.write(f"Activation:          Swish (Sigmoid * x)\n")
        f.write(f"Total Parameters:    {total_params:,}\n")
        f.write("-" * 20 + "\n")
        f.write("FLOW PARAMETERS:\n")
        f.write(f"Source Dist:         {args.source_dist.capitalize()}\n")
        if args.source_dist == "uniform":
            f.write("Source Range:        [-3.0, 3.0]\n")
        else:
            f.write("Source Params:       Mean=0, Std=1\n")
        f.write("Path Type:           AffineProbPath\n")
        f.write("Scheduler:           CondOTScheduler (Optimal Transport)\n")
        f.write("-" * 20 + "\n")
        f.write("TRAINING CONFIGURATION:\n")
        f.write(f"Epochs:              {epochs}\n")
        f.write(f"Initial Batch Size:  {bs}\n")
        f.write("Dynamic Batching:    Full-dataset after epoch 2\n")
        f.write(f"Optimizer:           Adam (lr={lr})\n")
        f.write(f"LR Strategy:         {'Fixed' if LRfixed else 'StepLR'}\n")
        if not LRfixed:
            f.write(f"Scheduler Params:    StepSize={step_size}, Gamma={gamma}\n")
        f.write("-" * 48 + "\n")
        f.write("EVALUATION CONFIG (In-Training):\n")
        f.write("ODE Solver:          dopri5\n")
        f.write("Eval Samples:        50,000\n")
        f.write("==================\n")

    print(f"--- Summary file created at: {summary_path} ---")   

global_start_time = time.time()

parser = argparse.ArgumentParser(description="Training Flow Matching per dphi")
parser.add_argument("--name", type=str, default="t_uniform", help="Nome dell'esperimento/cartella")
parser.add_argument("--variable_lr", action="store_true", help="Se presente, usa il learning rate variabile (default: Fixed)")
parser.add_argument("--source_dist", type=str, default="uniform", choices=["uniform", "gaussian"], help="Distribuzione sorgente: 'uniform' o 'gaussian'")

args = parser.parse_args()

name = args.name
LRfixed = not args.variable_lr  # Se variable_lr è True, LRfixed diventa False

checkpoint_dir = f"checkpoints_{name}"
os.makedirs(checkpoint_dir, exist_ok=True)

realistic_dataset = np.load("deltaphimoredata.npy")
dphi = realistic_dataset[:, 2]

# Initialize new model for dphi
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

lr = 0.001
epochs = 10001
bs = 20480
print_every = 100
save_every = 100
hidden_dim = 128
num_layers = 6

vf4 = MLP(input_dim=1, time_dim=1, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
path = AffineProbPath(scheduler=CondOTScheduler())
optim4 = torch.optim.Adam(vf4.parameters(), lr=lr)

if LRfixed:
    print("Using fixed learning rate.")
else:
    print("Using variable learning rate schedule.")
    # scheduler parameters
    step_size = 100 # LR decrease every 100 epochs
    gamma = 0.5 # LR decrease factor
    scheduler = torch.optim.lr_scheduler.StepLR(optim4, step_size=step_size, gamma=gamma)

# print model parameters count
total_params = sum(p.numel() for p in vf4.parameters())
print(f"Model Parameters: {total_params}")

# start distribution: uniform -3, 3. The range is chosen to cover the dphi data range.
if args.source_dist == "gaussian":
    source_dist = torch.distributions.Normal(0.0, 1.0)
elif args.source_dist == "uniform":
    source_dist = torch.distributions.Uniform(-3.0, 3.0)

# Create summary file
create_experiment_summary(checkpoint_dir, name, args, dphi, total_params, hidden_dim, num_layers, epochs, bs, lr, LRfixed, step_size if not LRfixed else None, gamma if not LRfixed else None)

# Training loop
losses_exp4 = []
start_time = time.time()

# Create a custom sampler for dphi distribution
dphi_tensor = torch.from_numpy(dphi).float()

# Create a DataLoader over the full dataset and iterate in batches
dataset = torch.utils.data.TensorDataset(dphi_tensor.unsqueeze(1))
# loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

for epoch in range(epochs):
    if epoch == 0:
        loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    elif epoch == 2:
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    epoch_loss = 0.0
    num_batches = 0

    for batch in loader:
        optim4.zero_grad()
        
        # Batch from dataset
        x_1 = batch[0].to(device)  # shape (batch_curr, 1)
        batch_curr = x_1.size(0)
        
        # Start samples and times sized to current batch
        x_0 = source_dist.sample((batch_curr,)).unsqueeze(1).to(device)
        t = torch.rand(batch_curr).to(device) # time sampling randomly from [0, 1] (continuous)
        
        # Sample probability path
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        out = vf4(path_sample.x_t, path_sample.t)
        
        # Flow matching loss
        loss = torch.pow(out - path_sample.dx_t, 2).mean()
        
        # Optimizer step
        loss.backward()
        optim4.step()
            
        losses_exp4.append(loss.item())
        epoch_loss += loss.item()
        num_batches += 1

    if not LRfixed:
        scheduler.step() # scheduler step at the end of the epoch

    # Log progress
    if epoch % print_every == 0:
        avg_loss = epoch_loss / num_batches

        # Total time elapsed
        total_time_str = get_elapsed_time(global_start_time)
        elapsed = time.time() - global_start_time

        if LRfixed:
            print(f'| Epoch {epoch:6d} | loss {avg_loss:8.3f} | {elapsed * 1000 / print_every:5.2f} ms/epoch | Total Time {total_time_str}')
        else:
            current_lr = scheduler.get_last_lr()[0]
            print(f'| Epoch {epoch:6d} | LR: {current_lr:.6f} | loss {avg_loss:8.3f} | {elapsed * 1000 / print_every:5.2f} ms/epoch | Total Time {total_time_str}')
        
        print_gpu_memory(epoch=epoch)

    # save checkpoint and evaluation plots
    if epoch > 0 and epoch % save_every == 0:
        # Create epoch directory
        epoch_dir = os.path.join(checkpoint_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(epoch_dir, f'model_epoch_{epoch}.pth')
        
        # Save checkpoint
        save_dict = {
            'epoch': epoch,
            'model_state_dict': vf4.state_dict(),
            'optimizer_state_dict': optim4.state_dict(),
            'loss': avg_loss,
        }
        if not LRfixed:
            save_dict['lr'] = scheduler.get_last_lr()[0]
            
        torch.save(save_dict, checkpoint_path)
        print(f"--- Checkpoint saved: {checkpoint_path} ---")

        # Generate evaluation plots
        vf4.eval() # Set to evaluation mode
        save_evaluation_plots(epoch, vf4, source_dist, dphi, device, checkpoint_dir)
        print(f"--- Evaluation plots saved for epoch {epoch} ---")
        
        vf4.train() # Return to training mode

        # 4. Updated Loss Plot Generation
        print(f"Total iterations: {len(losses_exp4)}")
        save_loss_plot(losses_exp4, epoch, checkpoint_dir)
        print(f"--- Loss plot saved for epoch {epoch} ---")
    
    # Remove references to batch tensors to free memory at the end of each epoch
    del batch, x_1, x_0, t, out, loss
    torch.cuda.empty_cache()
