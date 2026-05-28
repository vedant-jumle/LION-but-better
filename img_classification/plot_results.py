"""
Plot training curves comparing baseline vs beta model.
Usage: python plot_results.py \
    --baseline /scratch/vvjumle/checkpoints/lion_tiny_decay/log.txt \
    --beta     /scratch/vvjumle/checkpoints/lion_tiny_decay_beta/log.txt \
    --out      results.png
"""
import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--beta', required=True)
    parser.add_argument('--out', default='results.png')
    args = parser.parse_args()

    b_epochs, b_loss, b_acc1, b_acc5 = load_log(args.baseline)
    e_epochs, e_loss, e_acc1, e_acc5 = load_log(args.beta)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('LION-D Tiny: Baseline vs Learned β', fontsize=13)

    # Train loss
    axes[0].plot(b_epochs, b_loss, label='Baseline', color='steelblue')
    axes[0].plot(e_epochs, e_loss, label='Learned β', color='darkorange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Val Top-1
    axes[1].plot(b_epochs, b_acc1, label='Baseline', color='steelblue')
    axes[1].plot(e_epochs, e_acc1, label='Learned β', color='darkorange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Top-1 Accuracy (%)')
    axes[1].set_title('Validation Top-1 Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Val Top-5
    axes[2].plot(b_epochs, b_acc5, label='Baseline', color='steelblue')
    axes[2].plot(e_epochs, e_acc5, label='Learned β', color='darkorange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Top-5 Accuracy (%)')
    axes[2].set_title('Validation Top-5 Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.out}")

    # Print summary
    if b_acc1:
        print(f"\nBaseline   — best Top-1: {max(b_acc1):.2f}% @ epoch {b_epochs[b_acc1.index(max(b_acc1))]}")
    if e_acc1:
        print(f"Learned β  — best Top-1: {max(e_acc1):.2f}% @ epoch {e_epochs[e_acc1.index(max(e_acc1))]}")


if __name__ == '__main__':
    main()
