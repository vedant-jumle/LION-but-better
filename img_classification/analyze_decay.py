"""
Inspect learned decay lambda values from a trained LION-D checkpoint.
Usage: python analyze_decay.py --checkpoint /scratch/vvjumle/checkpoints/lion_tiny_learned/checkpoint.pth
"""
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', default='decay_analysis.png')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt['model'] if 'model' in ckpt else ckpt

    decay_data = {}
    for k, v in state.items():
        if k.endswith('.a_i') and 'attn' in k:
            layer = k.replace('.a_i', '')
            lambda_vals = torch.sigmoid(v).numpy()
            decay_data[layer] = lambda_vals

    if not decay_data:
        print("No a_i parameters found in checkpoint.")
        return

    layers = sorted(decay_data.keys())
    n_layers = len(layers)
    n_heads = len(decay_data[layers[0]])

    print(f"\nFound {n_layers} layers, {n_heads} heads each\n")
    print(f"{'Layer':<40} {'Min':>6} {'Mean':>6} {'Max':>6}  Values")
    print("-" * 80)
    for layer in layers:
        vals = decay_data[layer]
        print(f"{layer:<40} {vals.min():>6.3f} {vals.mean():>6.3f} {vals.max():>6.3f}  {np.round(vals, 3).tolist()}")

    matrix = np.array([decay_data[l] for l in layers])

    fig, ax = plt.subplots(figsize=(max(6, n_heads * 1.5), max(4, n_layers * 0.5 + 1)))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f'H{i}' for i in range(n_heads)])
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f'Block {i}' for i in range(n_layers)], fontsize=8)
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title(r'Learned $\lambda$ = sigmoid($a_i$) per head per layer' + '\n'
                 r'(0 = no decay, 1 = full memory)')
    plt.colorbar(im, ax=ax, label=r'$\lambda = \sigma(a_i)$')

    for i in range(n_layers):
        for j in range(n_heads):
            ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=8,
                    color='black' if 0.3 < matrix[i,j] < 0.7 else 'white')

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"\nHeatmap saved to {args.out}")
    print(f"\nOverall: mean λ = {matrix.mean():.3f}, std = {matrix.std():.3f}")
    print(f"Range: [{matrix.min():.3f}, {matrix.max():.3f}]")
    if matrix.std() < 0.05:
        print("→ Low variance: heads converge to similar λ — fixed decay may suffice")
    else:
        print("→ High variance: heads specialize to different λ — learned decay is beneficial")


if __name__ == '__main__':
    main()
