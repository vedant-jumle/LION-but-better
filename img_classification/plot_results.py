"""
Plot training curves for multiple LION-D variants.
Usage:
  python plot_results.py \
    --logs label1:/path/to/log1.txt label2:/path/to/log2.txt ... \
    --out results.png

Example:
  python plot_results.py \
    --logs "Learned:/scratch/vvjumle/checkpoints/lion_tiny_learned/log.txt" \
           "Fixed 0.5:/scratch/vvjumle/checkpoints/lion_tiny_fixed05/log.txt" \
           "Fixed 0.8:/scratch/vvjumle/checkpoints/lion_tiny_fixed08/log.txt" \
           "Fixed 0.9:/scratch/vvjumle/checkpoints/lion_tiny_fixed09/log.txt" \
    --out results.png
"""
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def load_log(path):
    epochs, train_loss, val_acc1, val_acc5 = [], [], [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            epochs.append(d['epoch'])
            train_loss.append(d.get('train_loss', float('nan')))
            val_acc1.append(d.get('test_acc1', float('nan')))
            val_acc5.append(d.get('test_acc5', float('nan')))
    return epochs, train_loss, val_acc1, val_acc5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', nargs='+', required=True,
                        help='label:path pairs, e.g. "Learned:/path/log.txt"')
    parser.add_argument('--out', default='results.png')
    args = parser.parse_args()

    runs = []
    for entry in args.logs:
        label, path = entry.split(':', 1)
        epochs, loss, acc1, acc5 = load_log(path)
        runs.append((label, epochs, loss, acc1, acc5))

    colors = cm.tab10(np.linspace(0, 0.9, len(runs)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('LION-D Tiny: Learned vs Fixed Decay λ', fontsize=13)

    for (label, epochs, loss, acc1, acc5), color in zip(runs, colors):
        axes[0].plot(epochs, loss, label=label, color=color)
        axes[1].plot(epochs, acc1, label=label, color=color)
        axes[2].plot(epochs, acc5, label=label, color=color)

    for ax, title, ylabel in zip(axes,
                                  ['Training Loss', 'Validation Top-1', 'Validation Top-5'],
                                  ['Loss', 'Top-1 Accuracy (%)', 'Top-5 Accuracy (%)']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.out}\n")

    print(f"{'Variant':<15} {'Best Top-1':>10} {'@ Epoch':>8} {'Best Top-5':>10}")
    print("-" * 48)
    for label, epochs, loss, acc1, acc5 in runs:
        if acc1:
            best_idx = acc1.index(max(acc1))
            print(f"{label:<15} {max(acc1):>10.2f}% {epochs[best_idx]:>8} {max(acc5):>10.2f}%")


if __name__ == '__main__':
    main()
