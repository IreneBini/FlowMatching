""" FLOW MATCHING FOR 1D DISTRIBUTION LEARNING (dphi)

DESCRIPTION:
This script implements a Continuous Normalizing Flow (CNF) using the
Flow Matching paradigm to learn and generate a 1-dimensional
distribution (dphi). It maps a simple source distribution (Uniform or
Gaussian) to a target "realistic" distribution derived from a NumPy
dataset.

KEY COMPONENTS:
1. Model Architecture: 
   - A Multi-Layer Perceptron (MLP) with Swish activations.
   - Designed to predict the velocity field v(x, t).
2. Probability Path:
   - Uses AffineProbPath with an Optimal Transport (CondOT) scheduler.
   - Defines the linear trajectory between noise (t=0) and data (t=1).
3. Training:
   - Supports both fixed and variable learning rates (StepLR).
   - Implements dynamic batching (switching to full-batch after initial
     epochs).
   - Tracks GPU memory usage and saves periodic checkpoints.
4. Inference & Evaluation:
   - Solves the learned Neural ODE using the 'dopri5' solver.
   - Generates 500000 samples and visualizes the evolution of the 
     distribution over time-steps.

USAGE EXAMPLE:
    - Use Gaussian source distribution with clamping:
        - python deltaphi_test.py --name gaus --source_dist gaussian --clamp
    - Use Uniform source distribution with variable learning rate:
        - python deltaphi_test.py --name uniform_varLR --source_dist uniform --variable_lr
    - Resume training from a specific checkpoint:
        - python deltaphi_test.py --resume "checkpoints/epoch_500/model_epoch_500.pth"

OUTPUTS:
For each evaluation epoch, the script saves in the checkpoint directory:
- Checkpoints (.pth).
- Training loss and metrics evolution plot.
- Temporal evolution plots of the distribution.
- Final comparison histogram (Generated vs Target).
"""

import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn.functional as F

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from scipy import stats
from torch import nn, Tensor

# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: 
        return torch.sigmoid(x) * x

# Model class
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int = 1,
            time_dim: int = 1,
            hidden_dim: int = 128,
            num_layers: int = 4,
            dropout: float = 0.0
        ):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(input_dim)

        self.main = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim),
            Swish(),
            nn.Dropout(dropout),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    Swish(),
                    nn.Dropout(dropout),
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
        # # 1. Calcola l'output del modello (la velocità)
        # v = self.model(x, t, **extras)
        # v = torch.clamp(v, min=-100.0, max=100.0)
        # if torch.isnan(v).any() or torch.isinf(v).any():
        #     print(f"\n[!!!] ERRORE NUMERICO rilevato a t = {t.mean().item():.4f}")
        #     print(f"Norma di x: {torch.norm(x).item():.4f}")
        #     print(f"Valori NaN in v: {torch.isnan(v).sum().item()}")
        #     print(f"Valori Inf in v: {torch.isinf(v).sum().item()}")
        #     raise ValueError("Rilevati NaN/Inf nell'output del modello")
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
        
        print(
            f"| GPU Alloc: {allocated:.1f}MB | "
            f"Reserv: {reserved:.1f}MB | Max: {max_allocated:.1f}MB"
        )
        
        torch.cuda.reset_peak_memory_stats()
    else:
        print("GPU not available.")

def get_elapsed_time(start_time):
    elapsed = time.time() - start_time
    return str(datetime.timedelta(seconds=int(elapsed)))

def save_evaluation_plots(
        epoch,
        sol,
        T,
        target_data,
        base_dir,
        dim_index=0,
        variable_name='dphi'
    ):
    """Saves comparison plots for a specific epoch.
    Args:
        epoch (int): Current epoch number.
        sol (np.ndarray): Solution array from the ODE solver.
        T (torch.Tensor): Time grid used in the solver.
        target_data (np.ndarray): Target distribution data.
        base_dir (str): Base directory to save plots.
    """
    epoch_dir = os.path.join(base_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    data = target_data[:, dim_index]
    sol = sol[:, :, dim_index]

    final_positions = sol[-1, :]

    # Plot 1: Temporal Evolution
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    indices = np.linspace(0, len(T)-1, 10, dtype=int)
    for idx, t_idx in enumerate(indices):
        axes[idx].hist(
            sol[t_idx],
            bins=100,
            density=True,
            alpha=0.7,
            color='cyan'
        )
        axes[idx].set_title(f't = {T[t_idx].item():.2f}')
        axes[idx].set_xlabel(variable_name)
        axes[idx].set_ylabel('Density')
        axes[idx].grid(alpha=0.3)
        axes[idx].set_xlim(data.min() - 0.5, data.max() + 0.5)
    plt.suptitle('Distribution Evolution', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(epoch_dir, variable_name + '_distribution_evolution.png'))
    plt.close()

    # Plot 2: Final Comparison
    plt.figure(figsize=(8, 5))
    plt.hist(
        data,
        bins=50,
        density=True,
        alpha=0.5,
        color='red',
        label='Target'
    )
    plt.hist(
        final_positions,
        bins=50,
        density=True,
        alpha=0.7,
        color='cyan',
        label='Generated'
    )
    plt.xlabel(variable_name, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(data.min() - 0.5, data.max() + 0.5)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title(f'Comparison at Epoch {epoch}', fontsize=12)
    plt.savefig(os.path.join(epoch_dir, variable_name + '_final_comparison.png'), dpi=200)
    plt.close()

def correlation_plot_2d(
    epoch,
    target_data,
    sol,
    base_dir,
    variable_names=('phi1', 'phi2'),
    index = (0, 1)
):
    """Plots the 2D correlation between the two variables for target and generated data.
    """
    epoch_dir = os.path.join(base_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    gen_final = sol[-1]

    plt.figure(figsize=(12, 5))

    # Target Data
    plt.subplot(1, 2, 1)
    plt.hist2d(
        target_data[:, index[0]],
        target_data[:, index[1]],
        bins=50,
        density=True,
        cmap='Reds'
    )
    plt.colorbar(label='Density')
    plt.xlabel(variable_names[0], fontsize=12)
    plt.ylabel(variable_names[1], fontsize=12)
    plt.title('Target Data Correlation', fontsize=14)
    plt.grid(alpha=0.3)

    # Generated Data
    plt.subplot(1, 2, 2)
    plt.hist2d(
        gen_final[:, index[0]],
        gen_final[:, index[1]],
        bins=50,
        density=True,
        cmap='Blues'
    )
    plt.colorbar(label='Density')
    plt.xlabel(variable_names[0], fontsize=12)
    plt.ylabel(variable_names[1], fontsize=12)
    plt.title('Generated Data Correlation', fontsize=14)
    plt.grid(alpha=0.3)

    plt.suptitle(f'Correlation Comparison at Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(epoch_dir, variable_names[0] + '_' + variable_names[1] + '_correlation.png'), dpi=200)
    plt.close()

def compute_dphi(
    phi1, 
    phi2
):
    """Computes the dphi value given phi1 and phi2.
    """
    dphi = phi2 - phi1
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return dphi

def compute_dphi_from_pxpy(
    px1,
    py1,
    px2,
    py2
):
    """Computes dphi from px and py values.
    """
    phi1 = np.arctan2(py1, px1)
    phi2 = np.arctan2(py2, px2)
    dphi = compute_dphi(phi1, phi2)
    return dphi

def plot_dphi(
    epoch,
    dphi_data,
    dphi_gen,
    base_dir
):
    """Plots the dphi distribution.
    """
    epoch_dir = os.path.join(base_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(
        dphi_data,
        bins=50,
        density=True,
        alpha=0.7,
        color='red'
    )
    plt.hist(
        dphi_gen,
        bins=50,
        density=True,
        alpha=0.7,
        color='cyan'
    )
    plt.xlabel('dphi', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'dphi Distribution at Epoch {epoch}', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(epoch_dir, 'dphi_distribution.png'), dpi=200)
    plt.close()

# def plot_dphi_from_phi1phi2(
#     epoch,
#     target_data,
#     sol,
#     base_dir
# ):
#     """Plots the dphi distribution derived from phi1 and phi2 data.
#     """
#     epoch_dir = os.path.join(base_dir, f"epoch_{epoch}")
#     os.makedirs(epoch_dir, exist_ok=True)

#     gen_final = sol[-1] # Shape: (N, 2)

#     dphi_data = compute_dphi(target_data[:, 0], target_data[:, 1])
#     dphi_gen = compute_dphi(gen_final[:, 0], gen_final[:, 1])


#     plt.figure(figsize=(8, 5))
#     plt.hist(
#         dphi_data,
#         bins=50,
#         density=True,
#         alpha=0.7,
#         color='red'
#     )
#     plt.hist(
#         dphi_gen,
#         bins=50,
#         density=True,
#         alpha=0.7,
#         color='cyan'
#     )
#     plt.xlabel('dphi', fontsize=12)
#     plt.ylabel('Density', fontsize=12)
#     plt.title(f'dphi Distribution at Epoch {epoch}', fontsize=14)
#     plt.grid(alpha=0.3)
#     plt.savefig(os.path.join(epoch_dir, 'dphi_distribution_from_phi1phi2.png'), dpi=200)
#     plt.close()

# def save_evaluation_plots_2d(
#         epoch, 
#         sol, 
#         T, 
#         target_data, 
#         base_dir
#     ):
#     """Saves 2D comparison plots for a specific epoch.
#     """
#     epoch_dir = os.path.join(base_dir, f"epoch_{epoch}")
#     os.makedirs(epoch_dir, exist_ok=True)

#     gen_final = sol[-1] # Shape: (N, 2)

#     fig = plt.figure(figsize=(18, 6))
    
#     ax1 = fig.add_subplot(1, 2, 1)
#     ax1.hist(target_data[:, 0], bins=50, density=True, alpha=0.5, label='Target', color='red')
#     ax1.hist(gen_final[:, 0], bins=50, density=True, alpha=0.5, label='Generated', color='cyan')
#     ax1.set_title(f'phi1 comparison at Epoch {epoch}', fontsize=12)
#     ax1.set_xlabel('phi1')
#     ax1.set_ylabel('density')
#     ax1.legend()

#     ax2 = fig.add_subplot(1, 2, 2)
#     ax2.hist(target_data[:, 1], bins=50, density=True, alpha=0.5, label='Target', color='red')
#     ax2.hist(gen_final[:, 1], bins=50, density=True, alpha=0.5, label='Generated', color='cyan')
#     ax2.set_title(f'phi2 comparison at Epoch {epoch}', fontsize=12)
#     ax2.set_xlabel('phi2')
#     ax2.set_ylabel('density')
#     ax2.legend()

#     plt.savefig(os.path.join(epoch_dir, 'final_comparison_2d.png'), dpi=200)
#     plt.close()

#     # plot_dphi_from_phi1phi2(
#     #     epoch,
#     #     target_data[:, 0],
#     #     target_data[:, 1],
#     #     gen_final[:, 0],
#     #     gen_final[:, 1],
#     #     base_dir
#     # )

# def plot_flow_dynamics_1d(
#         epoch,
#         model,
#         solver,
#         source_dist,
#         target_data,
#         device,
#         base_dir
#     ):
#     """Plots the flow dynamics including the velocity field and sample
#     trajectories.
#     """
#     epoch_dir = os.path.join(base_dir, f"epoch_{epoch}")
#     os.makedirs(epoch_dir, exist_ok=True)

#     # Create grid for velocity field
#     t_range = torch.linspace(0, 1, 20).to(device)
#     x_range = torch.linspace(target_data.min() - 1, target_data.max() + 1, 50).to(device)
#     T_grid, X_grid = torch.meshgrid(t_range, x_range, indexing='ij')
    
#     with torch.no_grad():
#         v_pred = model(X_grid.reshape(-1, 1), T_grid.reshape(-1, 1))
#         V = v_pred.reshape(T_grid.shape).cpu().numpy()

#     # Generate sample trajectories
#     n_traj = 50
#     x_init = source_dist.sample((n_traj,)).unsqueeze(1).to(device)
#     t_eval = torch.linspace(0, 1, 100).to(device)
    
#     with torch.no_grad():
#         trajectories = solver.sample(
#             time_grid=t_eval,
#             x_init=x_init,
#             method='dopri5',
#             step_size=None,
#             return_intermediates=True
#         )
#         trajectories = trajectories.cpu().numpy().squeeze()

#     # Plotting
#     plt.figure(figsize=(12, 7))
#     t_np = t_range.cpu().numpy()
#     x_np = x_range.cpu().numpy()

#     # Compute max absolute velocity for normalization
#     v_max_abs = np.abs(V).max()

#     strm = plt.streamplot(
#         t_np,
#         x_np,
#         np.ones_like(V.T),
#         V.T,
#         color=V.T,
#         cmap='coolwarm',
#         norm=plt.Normalize(vmin=-v_max_abs, vmax=v_max_abs)
#         #alpha=0.3
#     )
    
#     for i in range(n_traj):
#         plt.plot(
#             t_eval.cpu().numpy(),
#             trajectories[:, i],
#             color='black',
#             alpha=0.2,
#             lw=1
#         )
    
#     plt.scatter(
#         np.ones(n_traj),
#         trajectories[-1, :],
#         color='red',
#         s=10,
#         label='Generated (t=1)',
#         zorder=5
#     )
#     plt.scatter(
#         np.zeros(n_traj),
#         trajectories[0, :],
#         color='blue',
#         s=10,
#         label='Source (t=0)',
#         zorder=5
#     )

#     plt.title(f'Flow Dynamics & Velocity Field - Epoch {epoch}')
#     plt.xlabel('Time $t$')
#     plt.ylabel('Position $x$ (dphi)')
#     plt.xlim(-0.05, 1.05)
#     plt.colorbar(strm.lines, label='Velocity $v(x, t)$')
#     plt.legend()
#     plt.grid(alpha=0.2)
    
#     plt.savefig(os.path.join(epoch_dir, 'flow_dynamics.png'), dpi=200)
#     plt.close()

# def save_loss_and_metrics_plot(
#         current_epoch,
#         epochs,
#         w_history_val,
#         w_history_train,
#         ks_history_val,
#         ks_history_train,
#         ks_pvalue_history_val,
#         ks_pvalue_history_train,
#         losses_train,
#         losses_val,
#         base_dir,
#         figname = 'loss_metrics_plots',
#         eval_name='',
#         ma_n_metrics = 20,
#         ma_n_loss = 10
#     ):
#     """Saves the evolution plot of loss and evaluation metrics.
#     """
#     epoch_dir = os.path.join(base_dir, f"epoch_{current_epoch}")
#     os.makedirs(epoch_dir, exist_ok=True)

#     if len(epochs) != len(w_history_val):
#         nn = len(epochs) - len(w_history_val)
#         epochs = epochs[nn:]

#     fig, axes = plt.subplots(2, 2, figsize=(18, 10)) 
#     axes = axes.flatten()
    
#     # 0. Wasserstein
#     axes[0].plot(epochs, w_history_val, marker='.', color='blue', linewidth=2, alpha=0.5, label='Validation')
#     if len(w_history_val) >= ma_n_metrics:
#         ma = np.convolve(w_history_val, np.ones(ma_n_metrics)/ma_n_metrics, mode='valid')
#         ma_epochs = epochs[:-ma_n_metrics+1]
#         axes[0].plot(ma_epochs, ma, linewidth=1, linestyle='--', color='blue', label='Moving Average validation')
#     axes[0].plot(epochs, w_history_train, marker='.', color='cyan', linewidth=2, alpha=0.5, label='Training')
#     if len(w_history_train) >= ma_n_metrics:
#         ma = np.convolve(w_history_train, np.ones(ma_n_metrics)/ma_n_metrics, mode='valid')
#         ma_epochs = epochs[:-ma_n_metrics+1]
#         axes[0].plot(ma_epochs, ma, linewidth=1, linestyle='--', color='cyan', label='Moving Average training')
#     axes[0].set_ylabel('Wasserstein Distance')
#     axes[0].set_xlabel('Epoch')
#     axes[0].set_ylim(0, 0.07)
#     axes[0].legend()
#     axes[0].set_title('Evolution of Wasserstein Distance')

#     # 1. KS Statistic
#     axes[1].plot(epochs, ks_history_val, marker='.', color='red', linewidth=2, alpha=0.5, label='Validation')
#     if len(ks_history_val) >= ma_n_metrics:
#         ma = np.convolve(ks_history_val, np.ones(ma_n_metrics)/ma_n_metrics, mode='valid')
#         ma_epochs = epochs[:-ma_n_metrics+1]
#         axes[1].plot(ma_epochs, ma, linewidth=1, linestyle='--', color='red', label='Moving Average validation')
#     axes[1].plot(epochs, ks_history_train, marker='.', color='orange', linewidth=2, alpha=0.5, label='Training')
#     if len(ks_history_train) >= ma_n_metrics:
#         ma = np.convolve(ks_history_train, np.ones(ma_n_metrics)/ma_n_metrics, mode='valid')
#         ma_epochs = epochs[:-ma_n_metrics+1]
#         axes[1].plot(ma_epochs, ma, linewidth=1, linestyle='--', color='orange', label='Moving Average training')
#     axes[1].set_ylabel('KS Statistic')
#     axes[1].set_xlabel('Epoch')
#     axes[1].legend()
#     axes[1].set_title('Evolution of KS Statistic')

#     # 2. KS p-value
#     axes[2].plot(epochs, ks_pvalue_history_val, marker='.', color='green', linewidth=2, alpha=0.5, label='Validation')
#     if len(ks_pvalue_history_val) >= ma_n_metrics:
#         ma = np.convolve(ks_pvalue_history_val, np.ones(ma_n_metrics)/ma_n_metrics, mode='valid')
#         ma_epochs = epochs[:-ma_n_metrics+1]
#         axes[2].plot(ma_epochs, ma, linewidth=1, linestyle='--', color='green', label='Moving Average validation')
#     axes[2].plot(epochs, ks_pvalue_history_train, marker='.', color='palegreen', linewidth=2, alpha=0.5, label='Training')
#     if len(ks_pvalue_history_train) >= ma_n_metrics:
#         ma = np.convolve(ks_pvalue_history_train, np.ones(ma_n_metrics)/ma_n_metrics, mode='valid')
#         ma_epochs = epochs[:-ma_n_metrics+1]
#         axes[2].plot(ma_epochs, ma, linewidth=1, linestyle='--', color='palegreen', label='Moving Average training')
#     axes[2].set_ylabel('KS p-value')
#     axes[2].set_xlabel('Epoch')
#     axes[2].set_yscale('log')
#     axes[2].set_ylim(1e-10, 1.0)
#     axes[2].legend()
#     axes[2].set_title('Evolution of KS p-value')

#     # 3. Loss Plots
#     if eval_name == '':
#         iterations = range(len(losses_train))
#     else:
#         iterations = epochs
#     axes[3].plot(iterations, losses_train, color='purple', linewidth=2, alpha=0.3, label='Training')
#     if len(losses_train) > ma_n_loss:
#         ma = np.convolve(losses_train, np.ones(ma_n_loss)/ma_n_loss, mode='valid')
#         ma_iter = iterations[:-ma_n_loss+1]
#         axes[3].plot(ma_iter, ma, linewidth=1, linestyle='--', color='purple', label='Moving Average training')
#     axes[3].plot(iterations, losses_val, color='tab:purple', linewidth=2, alpha=0.3, label='Validation')
#     if len(losses_val) > ma_n_loss:
#         ma = np.convolve(losses_val, np.ones(ma_n_loss)/ma_n_loss, mode='valid')
#         ma_iter = iterations[:-ma_n_loss+1]
#         axes[3].plot(ma_iter, ma, linewidth=1, linestyle='--', color='tab:purple', label='Moving Average validation')
#     axes[3].set_ylabel('Loss')
#     axes[3].set_xlabel('Epoch')
#     axes[3].legend()
#     axes[3].set_title('Evolution of Loss')

#     plt.tight_layout()
#     plt.savefig(os.path.join(epoch_dir, figname + eval_name +'.png'), dpi=200)
#     plt.close()

def save_loss_and_metrics_plot(
        current_epoch,
        epochs,
        w_mean_val, w_std_val,
        w_mean_train, w_std_train,
        ks_mean_val, ks_std_val,
        ks_mean_train, ks_std_train,
        ks_p_mean_val, ks_p_std_val, 
        ks_p_mean_train, ks_p_std_train,
        losses_train,
        losses_val,
        base_dir,
        figname = 'loss_metrics_plots',
        eval_name='',
    ):
    """Saves plots with error bars (mean ± std).
    """

    epoch_dir = os.path.join(base_dir, f"epoch_{current_epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    if len(epochs) != len(w_mean_val):
        nn = len(epochs) - len(w_mean_val)
        epochs = epochs[nn:]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10)) 
    axes = axes.flatten()
    
    # Helper per plottare media + barre di errore
    def plot_metric(ax, x, y, ystd, color, label):
        ax.plot(x, y, color=color, linewidth=1, alpha=0.3)
        ax.errorbar(x, y, yerr=ystd, fmt='o', color=color, label=label, markersize=4, capsize=3, elinewidth=1, alpha=0.8)

    # 0. Wasserstein
    plot_metric(axes[0], epochs, w_mean_val, w_std_val, 'blue', 'Validation')
    plot_metric(axes[0], epochs, w_mean_train, w_std_train, 'cyan', 'Training')
    axes[0].set_ylabel('Wasserstein Distance')
    axes[0].set_title('Wasserstein Distance')
    axes[0].set_ylim(0, 0.07) # Opzionale: sbloccalo se vuoi un range fisso

    # 1. KS Statistic
    plot_metric(axes[1], epochs, ks_mean_val, ks_std_val, 'red', 'Validation')
    plot_metric(axes[1], epochs, ks_mean_train, ks_std_train, 'orange', 'Training')
    axes[1].set_ylabel('KS Statistic')
    axes[1].set_title('KS Statistic')
    axes[1].set_ylim(0, 0.03)

    # 2. KS p-value
    plot_metric(axes[2], epochs, ks_p_mean_val, ks_p_std_val, 'green', 'Validation')
    plot_metric(axes[2], epochs, ks_p_mean_train, ks_p_std_train, 'palegreen', 'Training')
    axes[2].set_ylabel('KS p-value')
    axes[2].set_yscale('log')
    axes[2].set_ylim(1e-10, 1.0)
    axes[2].set_title('KS p-value')

    # 3. Loss
    iterations = epochs if eval_name != '' else range(len(losses_train))
    axes[3].plot(iterations, losses_train, color='purple', label='Training', alpha=0.6)
    axes[3].plot(iterations, losses_val, color='tab:purple', label='Validation', alpha=0.6)
    axes[3].set_ylabel('Loss')
    axes[3].set_title('Evolution of Loss')

    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.legend(fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(epoch_dir, figname + eval_name +'.png'), dpi=200)
    plt.close()

def create_experiment_summary(
        checkpoint_dir,
        name,
        args,
        dataset_size,
        total_params,
        hidden_dim,
        num_layers,
        epochs,
        bs,
        lr,
        start_epoch,
        dataset_name="deltaphimoredata.npy",
        data_target_var="dphi"
    ):
    """Creates a summary file documenting the experiment configuration.
    """
    summary_path = os.path.join(checkpoint_dir, "experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=" * 20 + "\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment Name: {name}\n")
        if args.resume:
            f.write(f"Resume from epoch: {start_epoch}\n")
        f.write("-" * 20 + "\n")
        f.write("DATASET INFO:\n")
        f.write(f"Target File:         {dataset_name}\n")
        f.write(f"Dataset Size:        {dataset_size} samples\n")
        f.write(f"Train/Val Split:     80% / 20%\n")
        f.write(f"Target Variable:     {data_target_var}\n")
        f.write("-" * 20 + "\n")
        f.write("MODEL ARCHITECTURE:\n")
        f.write(f"Type:                MLP\n")
        f.write(f"Layers:              {num_layers}\n")
        f.write(f"Hidden Dim:          {hidden_dim}\n")
        f.write(f"Activation:          Swish (Sigmoid * x)\n")
        f.write(f"Dropout:             {args.dropout}\n")
        f.write(f"Total Parameters:    {total_params:,}\n")
        f.write("-" * 20 + "\n")
        f.write("FLOW PARAMETERS:\n")
        f.write(f"Source Dist:         {args.source_dist.capitalize()}\n")
        if args.source_dist == "uniform":
            f.write("Source Range:        [-3.0, 3.0]\n")
        else:
            f.write("Source Params:       Mean=0, Std=1\n")
        if args.clamp:
            f.write("Clamping:            Enabled [-3.0, 3.0]\n")
        f.write("Path Type:           AffineProbPath\n")
        f.write("Scheduler:           CondOTScheduler (Optimal Transport)\n")
        f.write("-" * 20 + "\n")
        f.write("TRAINING CONFIGURATION:\n")
        f.write(f"Epochs:              {epochs}\n")
        f.write(f"Batch Size:          {bs}\n")
        # f.write(f"Dynamic Batching:    Full-dataset after epoch 2\n")
        f.write(f"Optimizer:           Adam (lr={lr})\n")
        f.write(f"LR Strategy:         {'Fixed' if args.variable_lr == False else 'StepLR'}\n")
        if args.variable_lr:
            f.write(f"Scheduler Params:    StepSize={args.step_size}, Gamma={args.gamma}\n")
        f.write(f"Loss Function:       MSE {'+ Cosine Similarity' if args.cosine_similarity_loss else ''}\n")
        if args.cosine_similarity_loss:
            f.write(f"lambda               {args.coeff}\n")
        f.write("-" * 20 + "\n")
        f.write("EVALUATION CONFIG (In-Training):\n")
        f.write("ODE Solver:          dopri5\n")
        f.write("Eval Samples:        100000\n")
        f.write("=" * 20 + "\n")

    print(f"--- Summary file created at: {summary_path} ---")

def plot_train_val_dist(
    train_data,
    val_data,
    base_dir,
    namefig = 'train_val_split_check.png',
    var_name = 'dphi'
):
    """Plots and saves the training and validation distribution comparison."
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(train_data, bins=50, density=True, alpha=0.5, color='tab:blue', label=f'Train ({len(train_data)} samples)')
    plt.hist(val_data, bins=50, density=True, alpha=0.5, color='tab:orange', label=f'Validation ({len(val_data)} samples)')
    
    plt.xlabel(var_name, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Comparison: Training vs Validation Distribution', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(base_dir, namefig)
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    global_start_time = time.time()
    start_time = time.time()
    print(f"starting at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Training Flow Matching")
    parser.add_argument(
        "--name",
        type=str,
        default="t_uniform",
        help="Experiment name for checkpoint directory"
    )
    parser.add_argument(
        "--variable_lr",
        action="store_true",
        help="If set, uses variable learning rate schedule; otherwise fixed LR"
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=10000,
        help=""
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help=""
    )
    parser.add_argument(
        "--source_dist",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian"],
        help="Source distribution: 'uniform' or 'gaussian'"
    )
    parser.add_argument(
        "--clamp",
        action="store_true",
        help="If set, clamps samples of the distribution to the range [-3, 3]"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to .pth file to resume training"
    )
    parser.add_argument(
        "--resume_lr",
        action="store_true",
        help="If set, resume from checkpoint with a new lr"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="If set, only evaluates existing checkpoints without training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20480,
        help="Batch size for first 2 epochs of training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100001,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=200,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of layers in the MLP"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of the MLP"
    )
    parser.add_argument(
        "--cosine_similarity_loss",
        action="store_true",
        help="If set, uses cosine similarity loss sum with MSE loss"
    )

    parser.add_argument(
        "--coeff",
        type=float,
        default=1.0,
        help=""
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout value"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help=""
    )

    args = parser.parse_args()

    name = args.name
    LRfixed = not args.variable_lr

    lr = args.lr
    bs = args.batch_size
    epochs = args.epochs
    save_every = args.save_every
    num_layers = args.num_layers
    hidden_dim = args.hidden_dim

    cos_sim_check = args.cosine_similarity_loss
    coeff = args.coeff

    dropout = args.dropout

    checkpoint_dir = f"checkpoints_{name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load dataset
    realistic_dataset = np.load("pxpy.npy")
    data = realistic_dataset[:, 2:6]  # Extract px1, py1, px2, py2
    
    # print(f"shape data: {data.shape}")
    # print(f"first 5 samples:\n {data[:5, :]}")

    list_var_names = ['px1', 'py1', 'px2', 'py2']

    # Shuffle and split dataset
    seed_base = 5
    np.random.seed(seed_base)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]

    ntrain = int(0.8 * len(data))
    data_train = data[:ntrain]
    data_val = data[ntrain:]

    for i in range(data_train.shape[1]):
        plot_train_val_dist(data_train[:, i], data_val[:, i], checkpoint_dir, f"variable{i+1}_train_val_split.png")
    
    # Define source distribution N dimensional
    if args.source_dist == "gaussian":
        loc = torch.zeros(data.shape[1]).to(device)
        covariance_matrix = torch.eye(data.shape[1]).to(device)
        source_dist = torch.distributions.MultivariateNormal(loc, covariance_matrix)

    elif args.source_dist == "uniform":
        low = torch.full((data.shape[1],), -3.0).to(device)
        high = torch.full((data.shape[1],), 3.0).to(device)
        source_dist = torch.distributions.Uniform(low, high)

    torch.manual_seed(1)
    n_eval_run_for_epoch = 20
    n_samples_eval = 100000
    x_init_eval_list = []

    for _ in range(n_eval_run_for_epoch):
        x_init_eval = source_dist.sample((n_samples_eval,))
        if args.clamp:
            x_init_eval = torch.clamp(x_init_eval, -3.0, 3.0)
        x_init_eval_list.append(x_init_eval)
    
    T_eval = torch.linspace(0, 1, 100)

    # Initialize model, path, optimizer
    vf4 = MLP(
        input_dim=data.shape[1],
        time_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    path = AffineProbPath(scheduler=CondOTScheduler())
    optim4 = torch.optim.Adam(vf4.parameters(), lr=lr)

    if LRfixed:
        print("Using fixed learning rate.")
    else:
        print("Using variable learning rate schedule.")
        # scheduler parameters
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim4,
            step_size=args.step_size,
            gamma=args.gamma
        )
    
    # print model parameters count
    total_params = sum(p.numel() for p in vf4.parameters())
    print(f"Model Parameters: {total_params}")

    # Histories for evaluation metrics
    losses_train = []
    losses_val = []
    
    w_val_hist = [[] for _ in range(data_train.shape[1])]
    w_train_hist = [[] for _ in range(data_train.shape[1])]
    ks_stat_val_hist = [[] for _ in range(data_train.shape[1])]
    ks_stat_train_hist = [[] for _ in range(data_train.shape[1])]
    ks_pval_val_hist = [[] for _ in range(data_train.shape[1])]
    ks_pval_train_hist = [[] for _ in range(data_train.shape[1])]

    w_val_hist_std = [[] for _ in range(data_train.shape[1])]
    w_train_hist_std = [[] for _ in range(data_train.shape[1])]
    ks_stat_val_hist_std = [[] for _ in range(data_train.shape[1])]
    ks_stat_train_hist_std = [[] for _ in range(data_train.shape[1])]
    ks_pval_val_hist_std = [[] for _ in range(data_train.shape[1])]
    ks_pval_train_hist_std = [[] for _ in range(data_train.shape[1])]

    w_val_hist_dphi = []
    w_train_hist_dphi = []
    ks_stat_val_hist_dphi = []
    ks_stat_train_hist_dphi = []
    ks_pval_val_hist_dphi = []
    ks_pval_train_hist_dphi = []

    w_val_hist_dphi_std = []
    w_train_hist_dphi_std = []
    ks_stat_val_hist_dphi_std = []
    ks_stat_train_hist_dphi_std = []
    ks_pval_val_hist_dphi_std = []
    ks_pval_train_hist_dphi_std = []

    eval_epochs = []

    best_eval_metric = float('inf')
    patience_counter = 0
    custom_patience = args.patience

    if args.eval_only:
        print("EVALUATION ONLY MODE")
        # Find all checkpoints in the directory
        checkpoint_paths = []
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                if file.endswith('.pth') and 'model_epoch' in file:
                    checkpoint_paths.append(os.path.join(root, file))
        
        # Sort numerically based on epoch number in filename
        checkpoint_paths.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))
        
        # # Optionally, subsample checkpoints for quicker evaluation
        # checkpoint_paths = checkpoint_paths[::5]
        
        for cp_path in checkpoint_paths:
            print(f"--- Processing Checkpoint: {cp_path} ---")
            checkpoint = torch.load(cp_path, map_location=device, weights_only=False)
            epoch = checkpoint['epoch']

            # Load model state
            vf4.load_state_dict(checkpoint['model_state_dict'])
            vf4.eval()

            # Generate samples and evaluate
            with torch.no_grad():
                n_samples_eval = 100000 
                x_init_eval = source_dist.sample((n_samples_eval,)).to(device)
                if args.clamp:
                    x_init_eval = torch.clamp(x_init_eval, -3.0, 3.0)
                
                wrapped_model_eval = WrappedModel(vf4)
                solver_eval = ODESolver(velocity_model=wrapped_model_eval)
                T_eval = torch.linspace(0, 1, 100).to(device)

                sol_eval = solver_eval.sample(
                    time_grid=T_eval,
                    x_init=x_init_eval,
                    method='dopri5',
                    return_intermediates=True,
                    atol = 1e-5,
                    rtol = 1e-5
                )
                sol = sol_eval.cpu().numpy()

            eval_epochs.append(epoch)

            losses_train.append(checkpoint.get('loss_history_train', [0])[-1])
            losses_val.append(checkpoint.get('loss_history_val', [0])[-1])

            for i in range(data_train.shape[1]):
                # Wasserstein
                if 'w_history_val' in checkpoint:
                    w_val_hist[i].append(checkpoint['w_history_val'][i][-1])
                if 'w_history_train' in checkpoint:
                    w_train_hist[i].append(checkpoint['w_history_train'][i][-1])
                
                # KS Statistic
                if 'ks_stat_history_val' in checkpoint:
                    ks_stat_val_hist[i].append(checkpoint['ks_stat_history_val'][i][-1])
                if 'ks_stat_history_train' in checkpoint:
                    ks_stat_train_hist[i].append(checkpoint['ks_stat_history_train'][i][-1])
                
                # KS p-value
                if 'ks_pvalue_history_val' in checkpoint:
                    ks_pval_val_hist[i].append(checkpoint['ks_pvalue_history_val'][i][-1])
                if 'ks_pvalue_history_train' in checkpoint:
                    ks_pval_train_hist[i].append(checkpoint['ks_pvalue_history_train'][i][-1])


            if 'w_history_val_dphi' in checkpoint:
                w_val_hist_dphi.append(checkpoint['w_history_val_dphi'][-1])
            if 'w_history_train_dphi' in checkpoint:
                w_train_hist_dphi.append(checkpoint['w_history_train_dphi'][-1])
            
            # KS Statistic dphi
            if 'ks_stat_history_val_dphi' in checkpoint:
                ks_stat_val_hist_dphi.append(checkpoint['ks_stat_history_val_dphi'][-1])
            if 'ks_stat_history_train_dphi' in checkpoint:
                ks_stat_train_hist_dphi.append(checkpoint['ks_stat_history_train_dphi'][-1])
            
            # KS p-value dphi
            if 'ks_pvalue_history_val_dphi' in checkpoint:
                ks_pval_val_hist_dphi.append(checkpoint['ks_pvalue_history_val_dphi'][-1])
            if 'ks_pvalue_history_train_dphi' in checkpoint:
                ks_pval_train_hist_dphi.append(checkpoint['ks_pvalue_history_train_dphi'][-1])

        # After processing all checkpoints, save plots for the last epoch
        for i in range(data_train.shape[1]):
            save_loss_and_metrics_plot(
                current_epoch=eval_epochs[-1],
                epochs=eval_epochs,
                w_history_val=w_val_hist[i],
                w_history_train=w_train_hist[i],
                ks_history_val=ks_stat_val_hist[i],
                ks_history_train=ks_stat_train_hist[i],
                ks_pvalue_history_val=ks_pval_val_hist[i],
                ks_pvalue_history_train=ks_pval_train_hist[i],
                losses_train=losses_train,
                losses_val=losses_val,
                base_dir=checkpoint_dir,
                figname=f'loss_metrics_plots_variable{i+1}',
                eval_name="_evalonly"
            )
            save_evaluation_plots(
                    epoch,
                    sol,
                    T_eval.cpu(),
                    data_val,
                    checkpoint_dir,
                    dim_index=i,
                    variable_name=f'variable{i+1}'
                )

        dphi_data_val = compute_dphi_from_pxpy(data_val[:, 0], data_val[:, 1], data_val[:, 2], data_val[:, 3])
        dphi_gen = compute_dphi_from_pxpy(
            sol[-1][:, 0],
            sol[-1][:, 1],
            sol[-1][:, 2],
            sol[-1][:, 3]
        )
        plot_dphi(epoch, dphi_data_val, dphi_gen, checkpoint_dir)

        save_loss_and_metrics_plot(
            current_epoch=eval_epochs[-1],
            epochs=eval_epochs,
            w_history_val=w_val_hist_dphi,
            w_history_train=w_train_hist_dphi,
            ks_history_val=ks_stat_val_hist_dphi,
            ks_history_train=ks_stat_train_hist_dphi,
            ks_pvalue_history_val=ks_pval_val_hist_dphi,
            ks_pvalue_history_train=ks_pval_train_hist_dphi,
            losses_train=losses_train,
            losses_val=losses_val,
            base_dir=checkpoint_dir,
            figname='loss_metrics_plots_dphi',
            eval_name="_evalonly"
        )
        correlation_plot_2d(epoch, data_val, sol, checkpoint_dir, variable_names=['px1', 'py1'], index=(0, 1))
        correlation_plot_2d(epoch, data_train, sol, checkpoint_dir, variable_names=['px2', 'py2'], index=(2, 3))

        print("EVALUATION COMPLETED")

    else:
        print("TRAINING MODE")

        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume and os.path.isfile(args.resume):
            print(f"--- Loading checkpoint: {args.resume} ---")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            
            # load model and optimizer states
            vf4.load_state_dict(checkpoint['model_state_dict'])
            optim4.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            losses_train = checkpoint.get('loss_history_train', [])
            losses_val = checkpoint.get('loss_history_val', [])
            eval_epochs = checkpoint.get('eval_epochs', [])

            input_dim = data_train.shape[1]
            
            # Mappatura delle metriche da gestire
            metrics_map = {
                'w_history_train': w_train_hist,
                'w_history_val': w_val_hist,
                'w_history_train_std': w_train_hist_std,
                'w_history_val_std': w_val_hist_std,
                'ks_stat_history_train': ks_stat_train_hist,
                'ks_stat_history_val': ks_stat_val_hist,
                'ks_stat_history_train_std': ks_stat_train_hist_std,
                'ks_stat_history_val_std': ks_stat_val_hist_std,
                'ks_pvalue_history_train': ks_pval_train_hist,
                'ks_pvalue_history_val': ks_pval_val_hist,
                'ks_pvalue_history_train_std': ks_pval_train_hist_std,
                'ks_pvalue_history_val_std': ks_pval_val_hist_std
            }

            for key, local_list in metrics_map.items():
                if key in checkpoint:
                    data_from_cp = checkpoint[key]
                    for i in range(min(len(data_from_cp), input_dim)):
                        local_list[i] = data_from_cp[i]
            
            dphi_metrics = {
                'w_history_train_dphi': w_train_hist_dphi,
                'w_history_val_dphi': w_val_hist_dphi,
                'w_history_train_dphi_std': w_train_hist_dphi_std,
                'w_history_val_dphi_std': w_val_hist_dphi_std,
                'ks_stat_history_train_dphi': ks_stat_train_hist_dphi,
                'ks_stat_history_val_dphi': ks_stat_val_hist_dphi,
                'ks_stat_history_train_dphi_std': ks_stat_train_hist_dphi_std,
                'ks_stat_history_val_dphi_std': ks_stat_val_hist_dphi_std,
                'ks_pvalue_history_train_dphi': ks_pval_train_hist_dphi,
                'ks_pvalue_history_val_dphi': ks_pval_val_hist_dphi,
                'ks_pvalue_history_train_dphi_std': ks_pval_train_hist_dphi_std,
                'ks_pvalue_history_val_dphi_std': ks_pval_val_hist_dphi_std
            }

            for key, local_list in dphi_metrics.items():
                if key in checkpoint:
                    local_list.clear()
                    local_list.extend(checkpoint[key])

            if 'best_eval_metric' in checkpoint:
                best_eval_metric = checkpoint['best_eval_metric']
            
            if 'patience' in checkpoint:
                patience_counter = checkpoint['patience']
            
            if args.resume_lr:
                lr = args.lr
            else:
                if 'lr' in checkpoint:
                    lr = checkpoint['lr']
                else:
                    lr = args.lr

            if not LRfixed:
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                else:
                    for _ in range(start_epoch):
                        scheduler.step()
                    
            print(f"--- Resumed from epoch {start_epoch} with LR={lr} ---")

        # Create summary file
        create_experiment_summary(
            checkpoint_dir,
            name,
            args,
            len(data),
            total_params,
            hidden_dim,
            num_layers,
            epochs,
            bs, 
            lr,
            start_epoch,
            "pxpy.npy",
            "px1, py1, px2, py2"
        )

        print(f"best metric value: {best_eval_metric}")
        print(f"patience: {patience_counter}")

        # Create a custom sampler for dphi distribution
        data_train_tensor = torch.from_numpy(data_train).float()

        # Create a DataLoader over the full dataset and iterate in batches
        dataset = torch.utils.data.TensorDataset(data_train_tensor)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=bs,
            shuffle=True
        )

        for epoch in range(start_epoch, epochs):
            torch.manual_seed(seed_base + epoch) # For reproducibility also when restart the training from a checkpoint 
    
            epoch_loss_accum = 0.0
            num_batches = 0

            for batch in loader:
                optim4.zero_grad()
                
                # Batch from dataset
                x_1 = batch[0].to(device) # shape (batch_curr, 2)
                batch_curr = x_1.size(0)
                
                # Start samples and times sized to current batch
                x_0 = source_dist.sample((batch_curr,)).to(device) # ho tolto un unsqueeze(1) per avere shape (batch_curr, 2)
                if args.clamp == True:
                    x_0 = torch.clamp(x_0, -3.0, 3.0) # Clamp for uniform distribution

                t = torch.rand(batch_curr, 1).to(device) # Uniform(0, 1) times
                
                # Sample probability path
                path_sample = path.sample(t=t.flatten(), x_0=x_0, x_1=x_1)
                out = vf4(path_sample.x_t, path_sample.t)
                
                # Flow matching loss
                loss_cfm = torch.pow(out - path_sample.dx_t, 2).mean()

                if cos_sim_check:
                    cos_sim = F.cosine_similarity(out, path_sample.dx_t, dim=1, eps=1e-8)
                    loss_sim = (1.0 - cos_sim).mean()
                    loss = loss_cfm + coeff * loss_sim
                else:
                    loss = loss_cfm

                # Optimizer step
                loss.backward()
                # Add gradient clipping
                #torch.nn.utils.clip_grad_norm_(vf4.parameters(), max_norm=1.0)
                # Update weights
                optim4.step()
                    
                epoch_loss_accum += loss.item()
                num_batches += 1

            # Validation loss computation
            x_1_v = torch.from_numpy(data_val).float().to(device)
            vf4.eval()
            with torch.no_grad():
                x_0_v = source_dist.sample((x_1_v.size(0),)).to(device)
                if args.clamp: 
                    x_0_v = torch.clamp(x_0_v, -3.0, 3.0)

                t_v = torch.rand(x_1_v.size(0), 1).to(device)
                path_v = path.sample(t=t_v.flatten(), x_0=x_0_v, x_1=x_1_v)
                
                out_v = vf4(path_v.x_t, path_v.t)
                v_loss_cfm = torch.pow(out_v - path_v.dx_t, 2).mean().item()
                if cos_sim_check:
                    v_loss_sim = (1.0 - F.cosine_similarity(out_v, path_v.dx_t, dim=1, eps=1e-8)).mean().item()
                    v_loss = v_loss_cfm + coeff * v_loss_sim
                else:
                    v_loss = v_loss_cfm
                losses_val.append(v_loss)

            vf4.train()

            if not LRfixed:
                scheduler.step() # scheduler step at the end of the epoch

            avg_loss = epoch_loss_accum / num_batches
            losses_train.append(avg_loss)

            # save checkpoint and evaluation plots
            if epoch > 0 and epoch % save_every == 0:
                # Evaluation: Generate samples and compute metrics
                print("-" * 100)
                print(f"Epoch {epoch}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                vf4.eval()
                with torch.no_grad():
                    run_w_val = [[] for _ in range(data_train.shape[1])]
                    run_w_train = [[] for _ in range(data_train.shape[1])]
                    run_ks_stat_val = [[] for _ in range(data_train.shape[1])]
                    run_ks_stat_train = [[] for _ in range(data_train.shape[1])]
                    run_ks_pval_val = [[] for _ in range(data_train.shape[1])]
                    run_ks_pval_train = [[] for _ in range(data_train.shape[1])]

                    run_w_val_dphi = []
                    run_w_train_dphi = []
                    run_ks_stat_val_dphi = []
                    run_ks_stat_train_dphi = []
                    run_ks_pval_val_dphi = []
                    run_ks_pval_train_dphi = []

                    for jj in range(n_eval_run_for_epoch):
                        wrapped_model_eval = WrappedModel(vf4)
                        solver_eval = ODESolver(velocity_model=wrapped_model_eval)

                        x_init_run = x_init_eval_list[jj]

                        sol_eval = solver_eval.sample(
                            time_grid=T_eval.to(device),
                            x_init=x_init_run.to(device),
                            step_size=None,
                            method='dopri5',
                            return_intermediates=True,
                            atol = 1e-5,
                            rtol = 1e-5
                            )

                        sol = sol_eval.cpu().numpy()
                        final_pos = sol[-1] # For metric computation I need only solver at final time   

                        # Compute wasserstein distance and KS statistic for each features (for needed for this)
                        for i in range(data_train.shape[1]):
                            # Wasserstein
                            w_train_run = stats.wasserstein_distance(data_train[:, i], final_pos[:, i])
                            w_val_run = stats.wasserstein_distance(data_val[:, i], final_pos[:, i])
                            
                            # Kolmogorov-Smirnov
                            ks_s_t_run, ks_p_t_run = stats.ks_2samp(data_train[:, i], final_pos[:, i])
                            ks_s_v_run, ks_p_v_run = stats.ks_2samp(data_val[:, i], final_pos[:, i])

                            run_w_train[i].append(w_train_run)
                            run_w_val[i].append(w_val_run)
                            run_ks_stat_train[i].append(ks_s_t_run)
                            run_ks_pval_train[i].append(ks_p_t_run)
                            run_ks_stat_val[i].append(ks_s_v_run)
                            run_ks_pval_val[i].append(ks_p_v_run)

                        # Compute dphi metrics
                        dphi_data_train = compute_dphi_from_pxpy(
                            data_train[:, 0],
                            data_train[:, 1],
                            data_train[:, 2],
                            data_train[:, 3]
                        )
                        dphi_data_val = compute_dphi_from_pxpy(
                            data_val[:, 0],
                            data_val[:, 1],
                            data_val[:, 2],
                            data_val[:, 3]
                        )
                        dphi_gen = compute_dphi_from_pxpy(
                            final_pos[:, 0],
                            final_pos[:, 1],
                            final_pos[:, 2],
                            final_pos[:, 3]
                        )
                        # dphi_data_train = compute_dphi(data_train[:, 0], data_train[:, 1])
                        # dphi_data_val = compute_dphi(data_val[:, 0], data_val[:, 1])
                        # dphi_gen = compute_dphi(final_pos[:, 0], final_pos[:, 1])

                        # metrics for dphi
                        w_dphi_train_run = stats.wasserstein_distance(dphi_data_train, dphi_gen)
                        ks_s_dphi_train_run, ks_p_dphi_train_run = stats.ks_2samp(dphi_data_train, dphi_gen)

                        w_dphi_val_run = stats.wasserstein_distance(dphi_data_val, dphi_gen)
                        ks_s_dphi_val_run, ks_p_dphi_val_run = stats.ks_2samp(dphi_data_val, dphi_gen)

                        run_w_val_dphi.append(w_dphi_val_run)
                        run_w_train_dphi.append(w_dphi_train_run)
                        run_ks_stat_val_dphi.append(ks_s_dphi_val_run)
                        run_ks_stat_train_dphi.append(ks_s_dphi_train_run)
                        run_ks_pval_val_dphi.append(ks_p_dphi_val_run)
                        run_ks_pval_train_dphi.append(ks_p_dphi_train_run)

                    ##########################################
                    # Qui devo calcolare le medie e metterle in quelle liste li, devo anche fare le liste per gli rms
                    ##########################################
                    for i in range(data_train.shape[1]):

                        w_train_hist[i].append(np.mean(run_w_train[i]))
                        w_train_hist_std[i].append(np.std(run_w_train[i]))
                        w_val_hist[i].append(np.mean(run_w_val[i]))
                        w_val_hist_std[i].append(np.std(run_w_val[i]))

                        ks_stat_train_hist[i].append(np.mean(run_ks_stat_train[i]))
                        ks_stat_train_hist_std[i].append(np.std(run_ks_stat_train[i]))
                        ks_stat_val_hist[i].append(np.mean(run_ks_stat_val[i]))
                        ks_stat_val_hist_std[i].append(np.std(run_ks_stat_val[i]))

                        ks_pval_train_hist[i].append(np.mean(run_ks_pval_train[i]))
                        ks_pval_train_hist_std[i].append(np.std(run_ks_pval_train[i]))
                        ks_pval_val_hist[i].append(np.mean(run_ks_pval_val[i]))
                        ks_pval_val_hist_std[i].append(np.std(run_ks_pval_val[i]))

                    w_val_hist_dphi.append(np.mean(run_w_val_dphi))
                    w_val_hist_dphi_std.append(np.std(run_w_val_dphi))
                    w_train_hist_dphi.append(np.mean(run_w_train_dphi))
                    w_train_hist_dphi_std.append(np.std(run_w_train_dphi))

                    ks_stat_val_hist_dphi.append(np.mean(run_ks_stat_val_dphi))
                    ks_stat_val_hist_dphi_std.append(np.std(run_ks_stat_val_dphi))
                    ks_stat_train_hist_dphi.append(np.mean(run_ks_stat_train_dphi))
                    ks_stat_train_hist_dphi_std.append(np.std(run_ks_stat_train_dphi))

                    ks_pval_val_hist_dphi.append(np.mean(run_ks_pval_val_dphi))
                    ks_pval_val_hist_dphi_std.append(np.std(run_ks_pval_val_dphi))
                    ks_pval_train_hist_dphi.append(np.mean(run_ks_pval_train_dphi))
                    ks_pval_train_hist_dphi_std.append(np.std(run_ks_pval_train_dphi))

                    eval_epochs.append(epoch)

                    if w_val_hist_dphi[-1] < best_eval_metric:
                        best_eval_metric = w_val_hist_dphi[-1]
                        patience_counter = 0
                        best_model_path = os.path.join(checkpoint_dir, f'epoch_{epoch}', f'model_epoch_{epoch}.pth')
                        print(best_model_path)
                        print(f"--- New best metric: {best_eval_metric:.6f} ---")
                    else:
                        patience_counter += 1
                        print(f"--- Patience : {patience_counter}/{custom_patience} ---")

                    if patience_counter >= custom_patience:
                        old_lr = optim4.param_groups[0]["lr"]
                        new_lr = old_lr * args.gamma

                        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)

                        # load model and optimizer states
                        vf4.load_state_dict(checkpoint['model_state_dict'])
                        optim4.load_state_dict(checkpoint['optimizer_state_dict'])

                        for param_group in optim4.param_groups:
                            param_group['lr'] = new_lr
                        
                        patience_counter = 0
                        print(f"--- Learning rate goes from {old_lr:.3e} to {new_lr:.3e} ---")

                # Logging metrics
                total_time_str = get_elapsed_time(global_start_time)
                elapsed = time.time() - global_start_time
                
                print(f'| Time: {total_time_str} | Speed: {elapsed * 1000 / epoch if epoch > 0 else 0:5.2f} ms/epoch')
                print(f'| Loss: {avg_loss:8.4f} | LR: {optim4.param_groups[0]["lr"]:.3e}')
                print_gpu_memory(epoch=epoch)

                # Create epoch directory
                epoch_dir = os.path.join(checkpoint_dir, f'epoch_{epoch}')
                os.makedirs(epoch_dir, exist_ok=True)
                
                checkpoint_path = os.path.join(epoch_dir, f'model_epoch_{epoch}.pth')
                
                # Save checkpoint
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': vf4.state_dict(),
                    'optimizer_state_dict': optim4.state_dict(),
                    'loss_history_train': losses_train,
                    'loss_history_val': losses_val,
                    'w_history_train': w_train_hist,
                    'w_history_val': w_val_hist,
                    'ks_stat_history_train': ks_stat_train_hist,
                    'ks_stat_history_val': ks_stat_val_hist,
                    'ks_pvalue_history_train': ks_pval_train_hist,
                    'ks_pvalue_history_val': ks_pval_val_hist,
                    'w_history_train_dphi': w_train_hist_dphi,
                    'w_history_val_dphi': w_val_hist_dphi,
                    'ks_stat_history_train_dphi': ks_stat_train_hist_dphi,
                    'ks_stat_history_val_dphi': ks_stat_val_hist_dphi,
                    'ks_pvalue_history_train_dphi': ks_pval_train_hist_dphi,
                    'ks_pvalue_history_val_dphi': ks_pval_val_hist_dphi,
                    'w_history_train_std': w_train_hist_std,
                    'w_history_val_std': w_val_hist_std,
                    'ks_stat_history_train_std': ks_stat_train_hist_std,
                    'ks_stat_history_val_std': ks_stat_val_hist_std,
                    'ks_pvalue_history_train_std': ks_pval_train_hist_std,
                    'ks_pvalue_history_val_std': ks_pval_val_hist_std,
                    'w_history_train_dphi_std': w_train_hist_dphi_std,
                    'w_history_val_dphi_std': w_val_hist_dphi_std,
                    'ks_stat_history_train_dphi_std': ks_stat_train_hist_dphi_std,
                    'ks_stat_history_val_dphi_std': ks_stat_val_hist_dphi_std,
                    'ks_pvalue_history_train_dphi_std': ks_pval_train_hist_dphi_std,
                    'ks_pvalue_history_val_dphi_std': ks_pval_val_hist_dphi_std,
                    'eval_epochs': eval_epochs,
                    'best_eval_metric': best_eval_metric,
                    'patience': patience_counter,
                    'lr': optim4.param_groups[0]["lr"]
                }
                    
                torch.save(save_dict, checkpoint_path)
                print(f"--- Checkpoint saved: {checkpoint_path} ---")
                print("-" * 100)

                # Generate evaluation plots
                for i in range(data_train.shape[1]):
                    save_evaluation_plots(
                        epoch,
                        sol,
                        T_eval.cpu(),
                        data_val,
                        checkpoint_dir,
                        dim_index=i,
                        variable_name=list_var_names[i]
                    )
                correlation_plot_2d(
                    epoch,
                    data_val,
                    sol,
                    checkpoint_dir,
                    variable_names=list_var_names[0:2],
                    index=(0, 1)
                )
                plot_dphi(
                    epoch,
                    dphi_data_val,
                    dphi_gen,
                    checkpoint_dir
                )
                
                # print(f"--- Evaluation plots saved for epoch {epoch} ---")

                vf4.train() # Return to training mode

                # 4. Updated Loss and metrics Plot Generation
                for i in range(data_train.shape[1]):
                    save_loss_and_metrics_plot(
                        epoch,
                        eval_epochs,
                        w_val_hist[i],
                        w_val_hist_std[i],
                        w_train_hist[i],
                        w_train_hist_std[i],
                        ks_stat_val_hist[i],
                        ks_stat_val_hist_std[i],
                        ks_stat_train_hist[i],
                        ks_stat_train_hist_std[i],
                        ks_pval_val_hist[i],
                        ks_pval_val_hist_std[i],
                        ks_pval_train_hist[i],
                        ks_pval_train_hist_std[i],
                        losses_train,
                        losses_val,
                        checkpoint_dir,
                        figname = 'loss_metrics_plots_' + list_var_names[i]
                        )
                save_loss_and_metrics_plot(
                    epoch,
                    eval_epochs,
                    w_val_hist_dphi,
                    w_val_hist_dphi_std,
                    w_train_hist_dphi,
                    w_train_hist_dphi_std,
                    ks_stat_val_hist_dphi,
                    ks_stat_val_hist_dphi_std,
                    ks_stat_train_hist_dphi,
                    ks_stat_train_hist_dphi_std,
                    ks_pval_val_hist_dphi,
                    ks_pval_val_hist_dphi_std,
                    ks_pval_train_hist_dphi,
                    ks_pval_train_hist_dphi_std,
                    losses_train,
                    losses_val,
                    checkpoint_dir,
                    figname = 'loss_metrics_plots_dphi'
                    )
                # print(f"--- Loss and metrics evolution plot saved for epoch {epoch} ---")

            # Remove references to batch tensors to free memory at the end of each epoch
            del batch, x_1, x_0, t, out, loss
            torch.cuda.empty_cache()
        print(f"TRAINING COMPLETED in {get_elapsed_time(global_start_time)}")
