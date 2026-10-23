"""
Sweep a directory of LBDN `Toy` checkpoints and produce two plots over the test set:

  1. margin_curve.png: for every margin threshold t between a run's min and max
     empirical margin (top-1 logit - runner-up logit, as in margin_sweep.py), the
     percentage of test samples whose margin exceeds t.
  2. activation_stats.png: for every layer of a run, the mean (top) and variance
     (bottom) of that layer's output activations over the test set.

One line per checkpoint, colored by layer type. Plain (unconstrained) networks are
excluded. Use --width / --depth to narrow the
sweep to a handful of runs (e.g. --width 128 --depth 8 gives one line per layer type).
"""

import os
import argparse

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from model import getModel
from dataset import getDataLoader
from margin_sweep import LAYER_COLORS, build_config, load_checkpoint, compute_margins

INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style_axes(ax):
    ax.figure.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def layer_legend(ax, layers):
    handles = [Line2D([0], [0], color=LAYER_COLORS.get(l, MUTED), linewidth=2, label=l)
               for l in layers]
    ax.legend(handles=handles, title="Layer", frameon=False)


def hooked_layers(model):
    """(name, module) for every layer whose output activations we record: each child of
    model.model except Dropout/BatchNorm. For Plain nets the hidden Linear is skipped
    (its ReLU is recorded instead, so we see post-activation values like the other types)."""
    children = list(model.model.children())
    layers = []
    for i, m in enumerate(children):
        if isinstance(m, (nn.Dropout, nn.BatchNorm1d)):
            continue
        if isinstance(m, nn.Linear) and i + 1 < len(children) and isinstance(children[i + 1], nn.ReLU):
            continue
        layers.append(m)
    names = [f"L{i + 1}" for i in range(len(layers) - 1)] + ["logits"]
    return list(zip(names, layers))


@torch.no_grad()
def compute_margins_and_activations(model, testLoader, device):
    """One pass over the test set: margins, correctness, and per-layer (mean, variance)."""
    sums = {}

    def make_hook(name):
        def hook(_m, _in, out):
            out = out.detach().double()
            s = sums.setdefault(name, [0.0, 0.0, 0])
            s[0] += out.sum().item()
            s[1] += out.pow(2).sum().item()
            s[2] += out.numel()
        return hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in hooked_layers(model)]
    margins, correct = compute_margins(model, testLoader, device)
    for h in handles:
        h.remove()

    stats = {}
    for name, (total, sumsq, count) in sums.items():
        mean = total / count
        stats[name] = (mean, max(sumsq / count - mean ** 2, 0.0))
    return margins, correct, stats


def plot_margin_curves(runs, out_path, num_thresholds, title):
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    for r in runs:
        m = r["margins"].numpy()
        thresholds = np.linspace(m.min(), m.max(), num_thresholds)
        pct_over = 100.0 * (m[None, :] > thresholds[:, None]).mean(axis=1)
        ax.plot(thresholds, pct_over, color=LAYER_COLORS.get(r["layer"], MUTED),
                linewidth=1.6, alpha=0.85)
    ax.set_xlabel("Margin threshold t  (top-1 logit - runner-up logit)")
    ax.set_ylabel("Test samples with margin > t (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    style_axes(ax)
    layer_legend(ax, list(dict.fromkeys(r["layer"] for r in runs)))
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_activation_stats(runs, out_path, title):
    fig, (ax_mean, ax_var) = plt.subplots(2, 1, figsize=(7.5, 7), dpi=150, sharex=True)
    for r in runs:
        color = LAYER_COLORS.get(r["layer"], MUTED)
        names = list(r["act_stats"])
        x = np.arange(len(names))
        for ax, idx in ((ax_mean, 0), (ax_var, 1)):
            ax.plot(x, [r["act_stats"][n][idx] for n in names], color=color, linewidth=1.6,
                    alpha=0.85, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
    # x tick labels come from the deepest run; shallower runs share the leading positions
    longest = max((list(r["act_stats"]) for r in runs), key=len)
    ax_var.set_xticks(np.arange(len(longest)))
    ax_var.set_xticklabels(longest)
    ax_var.set_xlabel("Layer")
    ax_mean.set_ylabel("Mean activation")
    ax_var.set_ylabel("Activation variance")
    ax_mean.set_title(title)
    for ax in (ax_mean, ax_var):
        style_axes(ax)
    layer_legend(ax_mean, list(dict.fromkeys(r["layer"] for r in runs)))
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str,
        default="/project/6105016/shayuxin/verifying-robust-networks/LBDN/saved_models_seed123/mnist")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--loss", type=str, default="xent", choices=["xent", "multimargin"])
    parser.add_argument("--width", type=int, default=None, help="only runs with this width")
    parser.add_argument("--depth", type=int, default=None, help="only runs with this depth")
    parser.add_argument("--max_epoch", type=int, default=19,
                        help="skip checkpoints advanced past this (0-indexed) epoch by --resume")
    parser.add_argument("--num_thresholds", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="where to write the plots (default: <root>)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir or args.root
    os.makedirs(out_dir, exist_ok=True)

    testLoader = None
    runs = []
    for run_dir_name in sorted(os.listdir(args.root)):
        ckpt_path = os.path.join(args.root, run_dir_name, "model.ckpt")
        if not os.path.isfile(ckpt_path):
            continue
        config = build_config(run_dir_name, args.dataset, args.seed)
        if config is None or config.loss != args.loss or config.layer == "Plain":
            continue
        if args.width is not None and config.width != args.width:
            continue
        if args.depth is not None and config.depth != args.depth:
            continue

        if testLoader is None:
            _, testLoader = getDataLoader(config)
        model = getModel(config).to(device)
        ckpt_epoch = load_checkpoint(model, ckpt_path, device)
        if ckpt_epoch is not None and ckpt_epoch > args.max_epoch:
            print(f"skip (epoch {ckpt_epoch} > max_epoch {args.max_epoch}): {run_dir_name}")
            continue

        margins, correct, act_stats = compute_margins_and_activations(model, testLoader, device)
        runs.append(dict(run_dir=run_dir_name, layer=config.layer, margins=margins,
                         act_stats=act_stats))
        print(f"{run_dir_name}: acc={100.0 * correct.float().mean().item():.2f}%  "
              f"margin in [{margins.min():.4f}, {margins.max():.4f}]")

    if not runs:
        print("no checkpoints matched")
        return

    filt = ", ".join(f"{k}={v}" for k, v in (("width", args.width), ("depth", args.depth)) if v)
    desc = f"{args.dataset} test, {args.loss} loss" + (f", {filt}" if filt else "")
    margin_path = os.path.join(out_dir, "margin_curve.png")
    act_path = os.path.join(out_dir, "activation_stats.png")
    plot_margin_curves(runs, margin_path, args.num_thresholds,
                       f"Test samples over margin ({desc})")
    plot_activation_stats(runs, act_path, f"Activation mean and variance per layer ({desc})")
    print(f"wrote {margin_path}\nwrote {act_path}")


if __name__ == "__main__":
    main()
