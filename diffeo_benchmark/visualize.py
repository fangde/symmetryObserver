import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffeo_benchmark.diffeo.generator import DiffeoConfig, DiffeoGenerator
from diffeo_benchmark.evaluate import load_model

def visualize_2d_slice(
    model,
    true_diffeo,
    x_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    save_path: str = "results/diffeo_grid_comparison.png",
    grid_res: int = 21,
    extent: float = 2.0
):
    """
    Projects 8D diffeomorphism onto coordinates (x_1, x_2) while keeping (x_3...x_8) at 0.
    Plots:
    1. Reference Grid
    2. Ground Truth Diffeomorphic Warped Grid
    3. Model Predicted Warped Grid
    4. Pointwise Error Heatmap
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dim = 8
    
    xs = np.linspace(-extent, extent, grid_res)
    ys = np.linspace(-extent, extent, grid_res)
    X, Y = np.meshgrid(xs, ys)
    
    # Construct 8D grid points
    grid_pts_8d = np.zeros((grid_res, grid_res, dim), dtype=np.float32)
    grid_pts_8d[:, :, 0] = X
    grid_pts_8d[:, :, 1] = Y
    
    flat_pts = torch.from_numpy(grid_pts_8d.reshape(-1, dim)) # (N, 8)
    
    # Ground Truth Warp
    with torch.no_grad():
        true_warp = true_diffeo.forward(flat_pts).numpy().reshape(grid_res, grid_res, dim)
        
    # Model Predicted Warp
    flat_pts_batch = flat_pts.unsqueeze(0) # (1, N, 8)
    with torch.no_grad():
        model.eval()
        pred_warp_batch = model(x_ctx, y_ctx, flat_pts_batch)
        pred_warp = pred_warp_batch.squeeze(0).cpu().numpy().reshape(grid_res, grid_res, dim)
        
    # Error between true and predicted
    err_map = np.linalg.norm(pred_warp - true_warp, axis=-1)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Reference Grid
    ax = axes[0]
    for i in range(grid_res):
        ax.plot(X[i, :], Y[i, :], color="gray", alpha=0.6, lw=1)
        ax.plot(X[:, i], Y[:, i], color="gray", alpha=0.6, lw=1)
    ax.scatter(x_ctx[0, :, 0].cpu(), x_ctx[0, :, 1].cpu(), color="crimson", s=25, label=f"Context ($K={x_ctx.shape[1]}$)", zorder=5)
    ax.set_title("Input Slice ($x_1, x_2$)", fontsize=13, fontweight="bold")
    ax.set_xlim(-extent * 1.2, extent * 1.2)
    ax.set_ylim(-extent * 1.2, extent * 1.2)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.4)
    
    # 2. Ground Truth Warped Grid
    ax = axes[1]
    for i in range(grid_res):
        ax.plot(true_warp[i, :, 0], true_warp[i, :, 1], color="royalblue", lw=1.2)
        ax.plot(true_warp[:, i, 0], true_warp[:, i, 1], color="royalblue", lw=1.2)
    ax.scatter(y_ctx[0, :, 0].cpu(), y_ctx[0, :, 1].cpu(), color="darkred", s=25, label="Transformed Context", zorder=5)
    ax.set_title("Ground Truth $\Phi_z(X)$", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.4)
    
    # 3. Model Warped Grid
    ax = axes[2]
    for i in range(grid_res):
        ax.plot(pred_warp[i, :, 0], pred_warp[i, :, 1], color="darkorange", lw=1.2)
        ax.plot(pred_warp[:, i, 0], pred_warp[:, i, 1], color="darkorange", lw=1.2)
    ax.set_title(f"Model Inferred $\hat{{\Phi}}(X)$", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)
    
    # 4. Error Map
    ax = axes[3]
    c = ax.contourf(X, Y, err_map, levels=20, cmap="viridis")
    fig.colorbar(c, ax=ax, label="Coordinate Error $||\hat{Y} - Y||_2$")
    ax.set_title("Prediction Error Heatmap", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    from diffeo_benchmark.data.dataset import DiffeoEpisodeDataset
    device = torch.device("cpu")
    config = DiffeoConfig(dim=8, num_layers=4)
    dataset = DiffeoEpisodeDataset(num_episodes=5, config=config, num_context=32, num_query=64, seed=123)
    ep = dataset[0]
    phi = dataset.get_diffeo(0)
    model = load_model("transformer", "checkpoints/transformer_l4_k32.pt", dim=8, device=device)
    visualize_2d_slice(
        model=model,
        true_diffeo=phi,
        x_ctx=ep["x_ctx"].unsqueeze(0),
        y_ctx=ep["y_ctx"].unsqueeze(0),
        save_path="results/demo_slice_visualization.png"
    )
