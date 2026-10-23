"""
Plot the per-neuron splitting n from beta_neuron_analysis.py's n_per_layer.csv, for
xent-trained LBDN `Toy` models at one fixed (p, eps):

  1. n_per_neuron_vs_width.png: total n over all hidden layers divided by the
     number of hidden neurons, vs width; one panel per depth, one line per
     Lipschitz layer type.
  2. n_per_neuron_by_layer.png: n_sum / d for each hidden layer (L1 ... L_{depth+1})
     at one width; one panel per depth, one line per layer type.
  3. n_per_neuron_over_training.png: total n / neurons vs epoch, for the runs that
     saved model_epoch*.ckpt snapshots.

Most runs stop at epoch 20, but a few were resumed to epoch 81. For figures 1 and 2,
those runs use their snapshot whose epoch is closest to --epoch instead of model.ckpt.
The width-86 run and runs without a width/depth/loss suffix (legacy) are excluded.
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from margin_sweep import LAYER_COLORS
from margin_plots import INK, MUTED, SURFACE, style_axes, layer_legend

LAYER_ORDER = ["Plain", "Sandwich", "Orthogon", "Aol"]
EXCLUDE_WIDTHS = {86}


def load_rows(csv_path, loss, pct, eps):
    """Rows of n_per_layer.csv for one loss and (pct, eps), with numeric fields parsed."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if not r["run_dir"].endswith(f"-{r['loss']}"):
                continue  # legacy run name without width/depth/loss suffix
            if r["loss"] != loss or int(r["pct"]) != pct or not np.isclose(float(r["eps"]), eps):
                continue
            if int(r["width"]) in EXCLUDE_WIDTHS or r["epoch"] == "None":
                continue
            r.update(width=int(r["width"]), depth=int(r["depth"]), epoch=int(r["epoch"]),
                     d=int(r["d"]), n_sum=float(r["n_sum"]),
                     layer_idx=int(r["net_layer"].lstrip("L")))
            rows.append(r)
    return rows


def group_checkpoints(rows):
    """{(run_dir, ckpt): dict(layer, width, depth, epoch, layers={layer_idx: (n_sum, d)})}"""
    ckpts = {}
    for r in rows:
        c = ckpts.setdefault((r["run_dir"], r["ckpt"]), dict(
            run_dir=r["run_dir"], layer=r["layer"], width=r["width"], depth=r["depth"],
            epoch=r["epoch"], layers={}))
        c["layers"][r["layer_idx"]] = (r["n_sum"], r["d"])
    for c in ckpts.values():
        n, d = zip(*c["layers"].values())
        c["n_per_neuron"] = sum(n) / sum(d)
    return ckpts


def pick_one_per_run(ckpts, epoch):
    """For every run, the checkpoint whose epoch is closest to `epoch` (ties -> model.ckpt)."""
    best = {}
    for (run, name), c in ckpts.items():
        key = (abs(c["epoch"] - epoch), name != "model")
        if run not in best or key < best[run][0]:
            best[run] = (key, c)
    return [c for _, c in best.values()]


def layers_present(runs):
    present = {r["layer"] for r in runs}
    return [l for l in LAYER_ORDER if l in present]


def plot_line(ax, x, y, layer):
    ax.plot(x, y, color=LAYER_COLORS.get(layer, MUTED), linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)


def plot_vs_width(runs, out_path, title, subtitle):
    depths = sorted({r["depth"] for r in runs})
    widths = sorted({r["width"] for r in runs})
    fig, axes = plt.subplots(1, len(depths), figsize=(4 * len(depths), 4.2), dpi=150,
                             sharey=True, squeeze=False)
    for ax, depth in zip(axes[0], depths):
        for layer in layers_present(runs):
            pts = sorted((r["width"], r["n_per_neuron"]) for r in runs
                         if r["layer"] == layer and r["depth"] == depth)
            if pts:
                plot_line(ax, *zip(*pts), layer)
        ax.set_xscale("log", base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels([f"{w // widths[0]}×" for w in widths])
        ax.minorticks_off()
        ax.set_xlabel("multiples of original model size")
        ax.set_title(f"depth={depth}")
        style_axes(ax)
    axes[0][0].set_yscale("log")
    axes[0][0].set_ylabel("Total n / number of hidden neurons")
    method_legend(axes[0][-1], layers_present(runs))
    finish_figure(fig, out_path, title, subtitle)


def method_legend(ax, layers, loc="best"):
    layer_legend(ax, layers)
    ax.get_legend().set_title("Lipschitz method")
    ax.get_legend().set_loc(loc)


def finish_figure(fig, out_path, title, subtitle):
    """Title on top, the (loss, p, eps) setting in muted text beneath it."""
    fig.suptitle(title, y=0.98, color=INK)
    fig.text(0.5, 0.92, subtitle, ha="center", va="top", color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_by_layer(runs, width, out_path, title, subtitle):
    runs = [r for r in runs if r["width"] == width]
    depths = sorted({r["depth"] for r in runs})
    fig, axes = plt.subplots(1, len(depths), figsize=(4 * len(depths), 4.2), dpi=150,
                             sharey=True, squeeze=False)
    for ax, depth in zip(axes[0], depths):
        for layer in layers_present(runs):
            for r in runs:
                if r["layer"] == layer and r["depth"] == depth:
                    idx = sorted(r["layers"])
                    plot_line(ax, idx, [r["layers"][i][0] / r["layers"][i][1] for i in idx], layer)
        ax.set_xticks(range(1, depth + 2))
        ax.set_xticklabels([f"L{i}" for i in range(1, depth + 2)])
        ax.set_xlabel("hidden layer")
        ax.set_title(f"depth={depth}")
        style_axes(ax)
    axes[0][0].set_yscale("log")
    axes[0][0].set_ylabel("n_sum / d  (per layer)")
    # the depth=2 panel has room above its lines; deeper Plain nets fill the others
    method_legend(axes[0][0], layers_present(runs), loc="upper right")
    finish_figure(fig, out_path, title, subtitle)


def plot_over_training(ckpts, out_path, title, subtitle):
    by_run = defaultdict(dict)
    for (run, name), c in ckpts.items():
        by_run[run][c["epoch"]] = c  # model.ckpt duplicates its last snapshot's epoch
    runs = {run: eps for run, eps in by_run.items() if len(eps) > 1}
    if not runs:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    for run, eps in sorted(runs.items()):
        e = sorted(eps)
        plot_line(ax, e, [eps[i]["n_per_neuron"] for i in e], eps[e[0]]["layer"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("Total n / number of hidden neurons")
    style_axes(ax)
    method_legend(ax, layers_present([next(iter(eps.values())) for eps in runs.values()]))
    finish_figure(fig, out_path, title, subtitle)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis_dir", type=str,
        default="/project/6105016/shayuxin/verifying-robust-networks/LBDN/saved_models_seed123/mnist/beta_n_analysis")
    parser.add_argument("--loss", type=str, default="xent", choices=["xent", "multimargin"])
    parser.add_argument("--pct", type=int, default=80)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.03, 0.05],
                        help="one set of plots per eps")
    parser.add_argument("--epoch", type=int, default=20,
                        help="for runs with several checkpoints, use the one closest to this epoch")
    parser.add_argument("--width", type=int, default=128, help="width for the per-layer plot")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="where to write the plots (default: <analysis_dir>/plots)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.analysis_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    for eps in args.eps:
        rows = load_rows(os.path.join(args.analysis_dir, "n_per_layer.csv"), args.loss, args.pct, eps)
        if not rows:
            print(f"no rows matched eps={eps:g}")
            continue
        ckpts = group_checkpoints(rows)
        runs = pick_one_per_run(ckpts, args.epoch)
        print(f"eps={eps:g}")
        for r in sorted(runs, key=lambda r: (r["layer"], r["depth"], r["width"])):
            print(f"  {r['run_dir']} (epoch {r['epoch']}): n/neuron = {r['n_per_neuron']:.2f}")

        desc = f"{args.loss} loss, p={args.pct}, ε={eps:g}"
        tag = f"eps{eps:g}"
        plot_vs_width(runs, os.path.join(out_dir, f"n_per_neuron_vs_width_{tag}.png"),
                      "Network splitting dependent on model width", desc)
        plot_by_layer(runs, args.width, os.path.join(out_dir, f"n_per_neuron_by_layer_{tag}.png"),
                      "Network splitting, per layer", f"width {args.width}, {desc}")
        plot_over_training(ckpts, os.path.join(out_dir, f"n_per_neuron_over_training_{tag}.png"),
                           "Network splitting over training", f"width 128, depth 2, {desc}")
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
