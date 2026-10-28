import os
import argparse
import time
import json
import torch
import numpy as np

from diffeo_benchmark.diffeo.generator import DiffeoConfig
from diffeo_benchmark.data.dataset import DiffeoEpisodeDataset
from diffeo_benchmark.train import train, get_model
from diffeo_benchmark.evaluate import load_model, evaluate_single_setting
from diffeo_benchmark.visualize import visualize_2d_slice

def run_experiment_suite(
    train_episodes: int = 2500,
    val_episodes: int = 300,
    test_episodes: int = 200,
    epochs: int = 8,
    batch_size: int = 32,
    device_str: str = "auto",
    skip_training: bool = False
):
    if device_str == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    print(f"============================================================")
    print(f"   Diffeomorphic Transformation Learning Benchmark (DTLB)   ")
    print(f"============================================================")
    print(f"Device: {device} | Epochs: {epochs} | Train Episodes: {train_episodes}")
    
    models_to_run = ["transformer", "deepsets", "invertible_flow"]
    
    # Step 1: Train Models on Standard Setup (4 layers, K=32, Dim=8)
    if not skip_training:
        for model_name in models_to_run:
            ckpt_path = f"checkpoints/{model_name}_l4_k32.pt"
            print(f"\n[Training] Starting training for {model_name.upper()}...")
            cmd = (
                f"python3 -m diffeo_benchmark.train "
                f"--model {model_name} "
                f"--num_layers 4 "
                f"--num_context 32 "
                f"--num_train_episodes {train_episodes} "
                f"--num_val_episodes {val_episodes} "
                f"--epochs {epochs} "
                f"--batch_size {batch_size} "
                f"--device {device} "
                f"--save_dir checkpoints"
            )
            os.system(cmd)

    # Step 2: Comprehensive Evaluation Across the Matrix
    eval_device = torch.device("cpu") # Autograd Jacobians evaluate cleanly on CPU
    loaded_models = {}
    for model_name in models_to_run:
        ckpt_path = f"checkpoints/{model_name}_l4_k32.pt"
        loaded_models[model_name] = load_model(model_name, ckpt_path, dim=8, device=eval_device)

    results_table = []
    
    # -------------------------------------------------------------
    # Experiment 1: Core Comparison on Standard Benchmark (L=4, K=32)
    # -------------------------------------------------------------
    print("\n>>> Running Experiment 1: Model Comparison (L=4, K=32)")
    std_config = DiffeoConfig(dim=8, num_layers=4, s_std=0.3, a_std=0.4, w_std=0.5, b_std=0.5)
    std_test_ds = DiffeoEpisodeDataset(num_episodes=test_episodes, config=std_config, num_context=32, num_query=128, seed=9999)

    # Evaluate Oracle
    oracle_res = evaluate_single_setting(None, std_test_ds, device=eval_device, is_oracle=True, num_eval_episodes=min(50, test_episodes))
    results_table.append({
        "setting": "Standard (L=4, K=32)",
        "model": "Oracle (GT)",
        **oracle_res
    })

    for name, model in loaded_models.items():
        res = evaluate_single_setting(model, std_test_ds, device=eval_device, is_oracle=False, num_eval_episodes=min(50, test_episodes))
        results_table.append({
            "setting": "Standard (L=4, K=32)",
            "model": name,
            **res
        })

    # -------------------------------------------------------------
    # Experiment 2: Context Sample Complexity (K = 8, 32, 128)
    # -------------------------------------------------------------
    print("\n>>> Running Experiment 2: Context Sample Complexity (K = 8, 128)")
    for k_val in [8, 128]:
        k_ds = DiffeoEpisodeDataset(num_episodes=min(50, test_episodes), config=std_config, num_context=k_val, num_query=128, seed=8888 + k_val)
        for name in ["transformer", "deepsets"]:
            model = loaded_models[name]
            res = evaluate_single_setting(model, k_ds, device=eval_device, is_oracle=False, num_eval_episodes=30, eval_jacobian=False, eval_inverse=False)
            results_table.append({
                "setting": f"Context Sample K={k_val}",
                "model": name,
                **res
            })

    # -------------------------------------------------------------
    # Experiment 3: Out-of-Distribution & Stress Tests
    # -------------------------------------------------------------
    print("\n>>> Running Experiment 3: OOD & Robustness Tests")
    
    # 3a. Stronger non-linearity (OOD scale)
    ood_config = DiffeoConfig(dim=8, num_layers=4, s_std=0.6, a_std=0.8, w_std=0.8, b_std=0.8)
    ood_ds = DiffeoEpisodeDataset(num_episodes=min(50, test_episodes), config=ood_config, num_context=32, num_query=128, seed=7777)
    for name in ["transformer", "deepsets"]:
        model = loaded_models[name]
        res = evaluate_single_setting(model, ood_ds, device=eval_device, is_oracle=False, num_eval_episodes=30, eval_jacobian=False, eval_inverse=False)
        results_table.append({
            "setting": "OOD Strong Distortion",
            "model": name,
            **res
        })

    # 3b. Observation Noise (sigma = 0.05)
    noise_ds = DiffeoEpisodeDataset(num_episodes=min(50, test_episodes), config=std_config, num_context=32, num_query=128, noise_std=0.05, seed=6666)
    for name in ["transformer", "deepsets"]:
        model = loaded_models[name]
        res = evaluate_single_setting(model, noise_ds, device=eval_device, is_oracle=False, num_eval_episodes=30, eval_jacobian=False, eval_inverse=False)
        results_table.append({
            "setting": "Noisy Context (std=0.05)",
            "model": name,
            **res
        })

    # Save JSON results
    with open("results/benchmark_summary.json", "w") as f:
        json.dump(results_table, f, indent=2)

    # Generate Visualization Slice
    print("\n>>> Generating 2D Warped Grid Visualization...")
    sample_ep = std_test_ds[0]
    sample_phi = std_test_ds.get_diffeo(0)
    visualize_2d_slice(
        model=loaded_models["transformer"],
        true_diffeo=sample_phi,
        x_ctx=sample_ep["x_ctx"].unsqueeze(0),
        y_ctx=sample_ep["y_ctx"].unsqueeze(0),
        save_path="results/transformer_grid_visualization.png"
    )

    # Print Summary Markdown Table
    print("\n" + "="*80)
    print("                     DTLB BENCHMARK EXPERIMENT RESULTS                     ")
    print("="*80)
    print(f"| {'Setting':<25} | {'Model':<16} | {'Coord RMSE':<10} | {'Rel Err (%)':<11} | {'Jac Err (%)':<11} | {'det(J)>0 (%)':<12} |")
    print(f"|{'-'*27}|{'-'*18}|{'-'*12}|{'-'*13}|{'-'*13}|{'-'*14}|")
    for r in results_table:
        setting = r["setting"]
        model = r["model"]
        rmse = f"{r['coord_rmse']:.4f}"
        rel_err = f"{r['coord_rel_err']*100:.2f}%"
        jac_err = f"{r['jac_rel_err']*100:.2f}%" if r.get("jac_rel_err") is not None else "N/A"
        det_pos = f"{r['positive_det_rate']*100:.1f}%" if r.get("positive_det_rate") is not None else "N/A"
        print(f"| {setting:<25} | {model:<16} | {rmse:<10} | {rel_err:<11} | {jac_err:<11} | {det_pos:<12} |")
    print("="*80)
    print("Summary saved to results/benchmark_summary.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_episodes", type=int, default=3000)
    parser.add_argument("--val_episodes", type=int, default=300)
    parser.add_argument("--test_episodes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip_training", action="store_true")
    args = parser.parse_args()

    run_experiment_suite(
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        test_episodes=args.test_episodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device_str=args.device,
        skip_training=args.skip_training
    )
