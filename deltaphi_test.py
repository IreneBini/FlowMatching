import argparse
import datetime
import flow_matching
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

global_start_time = time.time()

parser = argparse.ArgumentParser(description="Training Flow Matching per dphi")
parser.add_argument("--name", type=str, default="t_uniform", help="Nome dell'esperimento/cartella")
parser.add_argument("--variable_lr", action="store_true", help="Se presente, usa il learning rate variabile (default: Fixed)")

args = parser.parse_args()

# Assegnazione variabili
name = args.name
LRfixed = not args.variable_lr  # Se variable_lr è True, LRfixed diventa False

# Se vuoi che il nome della cartella rifletta automaticamente la scelta del LR:
suffix = "LRfixed" if LRfixed else "LRvariable"
checkpoint_dir = f"checkpoints_{name}_{suffix}"
os.makedirs(checkpoint_dir, exist_ok=True)

realistic_dataset = np.load("deltaphimoredata.npy")
dphi = realistic_dataset[:, 2]

# Initialize new model for dphi
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

lr = 0.001
epochs = 100001
bs = 20480
print_every = 100
save_every = 500

vf4 = MLP(input_dim=1, time_dim=1, hidden_dim=128, num_layers=6).to(device)
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
source_dist = torch.distributions.Uniform(-3.0, 3.0)

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

    # Alla fine dell'epoca
    del batch, x_1, x_0, t, out, loss  # Rimuove i riferimenti ai tensori del batch
    torch.cuda.empty_cache()

    # Log progress
    if epoch % print_every == 0:
        if not LRfixed:
            current_lr = scheduler.get_last_lr()[0]

        avg_loss = epoch_loss / num_batches

        # Tempo totale dall'inizio dello script
        total_time_str = get_elapsed_time(global_start_time)
        elapsed = time.time() - global_start_time

        if LRfixed:
            print(f'| Epoch {epoch:6d} | loss {avg_loss:8.3f} | {elapsed * 1000 / print_every:5.2f} ms/epoch | Total Time {total_time_str}')
        else:
            print(f'| Epoch {epoch:6d} | LR: {current_lr:.6f} | loss {avg_loss:8.3f} | {elapsed * 1000 / print_every:5.2f} ms/epoch | Total Time {total_time_str}')
        
        print_gpu_memory(epoch=epoch)

    # save checkpoint every 100 epochs
    if epoch > 0 and epoch % save_every == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
        if LRfixed:
            torch.save({
                'epoch': epoch,
                'model_state_dict': vf4.state_dict(),
                'optimizer_state_dict': optim4.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': vf4.state_dict(),
                'optimizer_state_dict': optim4.state_dict(),
                'loss': avg_loss,
                'lr': scheduler.get_last_lr()[0]
            }, checkpoint_path)
        print(f"---> Checkpoint saved: {checkpoint_path}")

# Plot training loss
print(f"Total iterations: {len(losses_exp4)}")
plt.figure(figsize=(10, 4))
plt.plot(losses_exp4, alpha=0.6)
plt.plot(np.convolve(losses_exp4, np.ones(100)/100, mode='valid'), linewidth=2, label='Moving Average (100)')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Realistic dphi Training Loss')
plt.legend()
plt.grid(alpha=0.3)
plt.yscale('log')
plt.savefig(os.path.join(checkpoint_dir, f'dphi_training_loss_{name}.png'), dpi=300)
plt.close()


# Sample trajectories
wrapped_vf4 = WrappedModel(vf4)
solver4 = ODESolver(velocity_model=wrapped_vf4)
T = torch.linspace(0, 1, 200).to(device)
n_samples = 100000
x_init_exp4 = source_dist.sample((n_samples,)).unsqueeze(1).to(device)
sol_exp4 = solver4.sample(
    time_grid=T, 
    x_init=x_init_exp4, 
    method='dopri5', 
    step_size=None, 
    return_intermediates=True
)
sol_exp4 = sol_exp4.cpu().numpy().squeeze()

print(f"Generated trajectories shape: {sol_exp4.shape}")

# Visualize evolution over time
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
time_indices = np.linspace(0, len(T)-1, 10, dtype=int)

for idx, t_idx in enumerate(time_indices):
    axes[idx].hist(sol_exp4[t_idx], bins=100, density=True, alpha=0.7, color='cyan')
    axes[idx].set_title(f't = {T[t_idx].cpu().item():.2f}')
    axes[idx].set_xlabel('dphi')
    axes[idx].set_ylabel('Density')
    axes[idx].grid(alpha=0.3)
    # Set xlim based on data range
    axes[idx].set_xlim(dphi.min() - 0.5, dphi.max() + 0.5)

plt.suptitle('Distribution Evolution (Realistic dphi)', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, f'dphi_distribution_evolution_{name}.png'), dpi=300)
plt.close()

final_positions_exp4 = sol_exp4[-1, :]
plt.hist(dphi, bins=100, density=True, alpha=0.5, color='red', label='Target')
plt.hist(final_positions_exp4, bins=100, density=True, alpha=0.7, color='cyan', label='Generated')
plt.xlabel('dphi', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Final Distribution: Generated vs Target', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(checkpoint_dir, f'final_dphi_distribution_{name}.png'), dpi=300)
plt.close()