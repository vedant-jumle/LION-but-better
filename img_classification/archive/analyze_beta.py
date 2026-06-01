"""
Inspect learned beta values from a trained LION-D checkpoint.
Usage: python analyze_beta.py --checkpoint /scratch/vvjumle/checkpoints/lion_tiny_decay_beta/checkpoint.pth
"""
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', default='beta_analysis.png')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt['model'] if 'model' in ckpt else ckpt

    # extract beta values per layer
    beta_data = {}
    for k, v in state.items():
        if 'beta' in k:
            layer = k.replace('.beta', '')
            beta_vals = torch.sigmoid(v).numpy()
            beta_data[layer] = beta_vals

    if not beta_data:
        print("No beta parameters found in checkpoint.")
        return

    layers = sorted(beta_data.keys())
    n_layers = len(layers)
    n_heads = len(beta_data[layers[0]])

    print(f"\nFound {n_layers} layers, {n_heads} heads each\n")
    print(f"{'Layer':<40} {'Min':>6} {'Mean':>6} {'Max':>6}  Values")
    print("-" * 80)
    for layer in layers:
        vals = beta_data[layer]
        print(f"{layer:<40} {vals.min():>6.3f} {vals.mean():>6.3f} {vals.max():>6.3f}  {np.round(vals, 3).tolist()}")

    # heatmap: layers x heads
    matrix = np.array([beta_data[l] for l in layers])

    fig, ax = plt.subplots(figsize=(max(6, n_heads), max(4, n_layers * 0.4 + 1)))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f'H{i}' for i in range(n_heads)])
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([l.split('.')[-3] if '.' in l else l for l in layers], fontsize=8)
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title('Learned β values (sigmoid) per head per layer\n0=self-loop suppressed, 1=self-loop amplified')
    plt.colorbar(im, ax=ax, label='β = sigmoid(w)')

    # annotate cells
    for i in range(n_layers):
        for j in range(n_heads):
            ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=7,
                    color='black' if 0.3 < matrix[i,j] < 0.7 else 'white')

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"\nHeatmap saved to {args.out}")


if __name__ == '__main__':
    main()
