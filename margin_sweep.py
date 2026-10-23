"""
Sweep a directory of LBDN `Toy` checkpoints and compute a certified robust test
accuracy from the empirical margin (top-1 logit - runner-up logit): a sample is
counted robust at perturbation size eps if its margin exceeds the error bound

    delta = sum_{l=0}^{L} beta_l * sqrt(k * d_l)

collapsed under constant beta_l=eps, d_l=width (same simplification as
plot_error_bound.py's bound()), with L taken as the network's depth. Reports
robust accuracy (at eps in {36,72,108}/255) alongside baseline (clean) test
accuracy for every checkpoint, and plots one against the other.
"""

import os
import re
import types
import argparse

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from model import getModel
from dataset import getDataLoader

# validated categorical palette, fixed order (dataviz skill reference palette)
LAYER_COLORS = {
    "Plain": "#2a78d6",
    "Sandwich": "#eb6834",
    "Orthogon": "#1baf7a",
    "Aol": "#eda100",
    "SLL": "#e87ba4",
}

# error_bound(L, d, eps, k) = L * eps * sqrt(k*d): certified perturbation budget
# delta = sum_{l=0}^{L} beta_l * sqrt(k*d_l), collapsed under constant beta_l=eps,
# d_l=width (same simplification as plot_error_bound.py's bound()). L is taken
# as the network's depth (number of hidden layers).
EPS_LIST = [36.0/255, 72.0/255, 108.0/255]


def error_bound(depth, width, eps, k):
    return depth * eps * np.sqrt(k * width)


DIR_RE = re.compile(
    r"^Toy-(?P<layer>\w+?)-(?P<scale>\w+)-gamma(?P<gamma>[\d.]+)"
    r"(?:-w(?P<width>\d+)-d(?P<depth>\d+))?"
    r"(?:-(?P<loss>xent|multimargin))?$"
)


def build_config(run_dir_name, dataset, seed):
    m = DIR_RE.match(run_dir_name)
    if m is None:
        return None

    layer = m.group("layer")
    # dirs predating configurable width/depth used the original hardcoded sizes
    width = int(m.group("width")) if m.group("width") else (86 if layer == "Sandwich" else 128)
    depth = int(m.group("depth")) if m.group("depth") else 8
    # dirs predating configurable loss were always trained with xent
    loss = m.group("loss") or "xent"

    config = types.SimpleNamespace(
        model="Toy",
        dataset=dataset,
        seed=seed,
        layer=layer,
        scale=m.group("scale"),
        gamma=float(m.group("gamma")),
        width=width,
        depth=depth,
        loss=loss,
        dropout=0.0,
        LLN=False,
        normalized=False,
        train_batch_size=256,
        test_batch_size=256,
        num_workers=0,
        in_channels=1,
        img_size=28,
        num_classes=10,
    )
    config.in_channels = config.in_channels * config.img_size * config.img_size
    return config


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    is_new_format = isinstance(ckpt, dict) and "model" in ckpt
    state_dict = ckpt["model"] if is_new_format else ckpt
    model.load_state_dict(state_dict)
    return ckpt["epoch"] if is_new_format else None


@torch.no_grad()
def compute_margins(model, testLoader, device):
    model.eval()
    margins, correct = [], []
    for batch in testLoader:
        x, y = batch[0].to(device), batch[1].to(device)
        x = x.view(x.shape[0], -1)
        logits = model(x)

        sorted_logits, _ = torch.sort(logits, dim=1)
        margin = sorted_logits[:, -1] - sorted_logits[:, -2]  # top1 - runner-up gap
        margins.append(margin.cpu())
        correct.append((logits.argmax(dim=1) == y).cpu())

    return torch.cat(margins), torch.cat(correct)


EPS_MARKERS = {EPS_LIST[0]: "o", EPS_LIST[1]: "^", EPS_LIST[2]: "s"}
EPS_LABELS = {EPS_LIST[0]: "36/255", EPS_LIST[1]: "72/255", EPS_LIST[2]: "108/255"}
EPS_COLS = {EPS_LIST[0]: "robust_accuracy_36", EPS_LIST[1]: "robust_accuracy_72",
            EPS_LIST[2]: "robust_accuracy_108"}


def plot_accuracy_comparison(results, out_path, loss_label):
    """One point per (checkpoint, eps): x = baseline test accuracy, y = robust
    test accuracy at that eps. Color = layer, marker shape = eps. The y=x line
    marks the theoretical ceiling (robust accuracy can never exceed baseline)."""
    if not results:
        return

    all_vals = [r["baseline_accuracy"] for r in results]
    for eps in EPS_LIST:
        all_vals += [r[EPS_COLS[eps]] for r in results]
    lo, hi = min(all_vals), max(all_vals)
    pad = max(2.0, 0.05 * (hi - lo))
    lims = [max(0.0, lo - pad), min(100.0, hi + pad)]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.plot(lims, lims, color="#c3c2b7", linewidth=1.2, linestyle="--", zorder=1)

    seen_layers = []
    for r in results:
        color = LAYER_COLORS.get(r["layer"], "#898781")
        if r["layer"] not in seen_layers:
            seen_layers.append(r["layer"])
        for eps in EPS_LIST:
            ax.scatter(r["baseline_accuracy"], r[EPS_COLS[eps]],
                       color=color, marker=EPS_MARKERS[eps], s=45,
                       alpha=0.8, linewidths=0, zorder=3)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Baseline test accuracy (%)", color="#0b0b0b")
    ax.set_ylabel("Robust test accuracy (%)", color="#0b0b0b")
    ax.set_title(f"Robust vs. baseline test accuracy across the Toy grid ({loss_label} loss)",
                 color="#0b0b0b")
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")

    layer_handles = [Line2D([0], [0], marker="o", linestyle="none",
                             markerfacecolor=LAYER_COLORS.get(l, "#898781"),
                             markeredgewidth=0, markersize=8, label=l)
                     for l in seen_layers]
    eps_handles = [Line2D([0], [0], marker=EPS_MARKERS[eps], linestyle="none",
                           markerfacecolor="#52514e", markeredgewidth=0,
                           markersize=8, label=EPS_LABELS[eps])
                   for eps in EPS_LIST]
    legend1 = ax.legend(handles=layer_handles, title="Layer", loc="upper left", frameon=False)
    ax.add_artist(legend1)
    ax.legend(handles=eps_handles, title="eps", loc="lower right", frameon=False)

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
    parser.add_argument("--out_csv", type=str, default=None,
                        help="summary CSV path (default: <root>/margin_sweep_summary.csv)")
    parser.add_argument("--log_file", type=str, default=None,
                        help="run log path (default: <root>/margin_sweep_log.txt)")
    parser.add_argument("--loss", type=str, default="xent", choices=["xent", "multimargin"],
                        help="only evaluate checkpoints trained with this loss")
    parser.add_argument("--plot_path", type=str, default=None,
                        help="accuracy comparison plot path (default: <root>/accuracy_comparison.png)")
    parser.add_argument("--k", type=float, default=0.05,
                        help="fixed k in the error bound delta = depth * eps * sqrt(k * width)")
    parser.add_argument("--max_epoch", type=int, default=19,
                        help="only evaluate checkpoints whose saved epoch (0-indexed) is at most this "
                             "-- keeps the experiment to the original 20-epoch checkpoints even if some "
                             "have since been advanced further by --resume")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_csv = args.out_csv or os.path.join(args.root, "margin_sweep_summary.csv")
    log_file = args.log_file or os.path.join(args.root, "margin_sweep_log.txt")
    plot_path = args.plot_path or os.path.join(args.root, "accuracy_comparison.png")
    log_fh = open(log_file, "a")

    def log(msg):
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    testLoader = None
    results = []

    for run_dir_name in sorted(os.listdir(args.root)):
        ckpt_path = os.path.join(args.root, run_dir_name, "model.ckpt")
        if not os.path.isfile(ckpt_path):
            continue

        config = build_config(run_dir_name, args.dataset, args.seed)
        if config is None:
            log(f"skip (unrecognized dir): {run_dir_name}")
            continue
        if config.loss != args.loss:
            continue

        if testLoader is None:
            _, testLoader = getDataLoader(config)

        model = getModel(config).to(device)
        ckpt_epoch = load_checkpoint(model, ckpt_path, device)
        if ckpt_epoch is not None and ckpt_epoch > args.max_epoch:
            log(f"skip (epoch {ckpt_epoch} > max_epoch {args.max_epoch}, already advanced by --resume): "
                f"{run_dir_name}")
            continue

        margins, correct = compute_margins(model, testLoader, device)

        baseline_acc = 100.0 * correct.float().mean().item()
        robust_acc = {}
        for eps in EPS_LIST:
            delta = error_bound(config.depth, config.width, eps, args.k)
            robust_mask = correct & (margins > delta)
            robust_acc[eps] = 100.0 * robust_mask.float().mean().item()

        results.append(dict(
            run_dir=run_dir_name, layer=config.layer, loss=config.loss,
            width=config.width, depth=config.depth, epoch=ckpt_epoch,
            baseline_accuracy=baseline_acc,
            robust_accuracy_36=robust_acc[EPS_LIST[0]],
            robust_accuracy_72=robust_acc[EPS_LIST[1]],
            robust_accuracy_108=robust_acc[EPS_LIST[2]],
        ))
        log(f"{run_dir_name}: baseline_acc={baseline_acc:.2f}%  "
            f"robust_acc[36/255]={robust_acc[EPS_LIST[0]]:.2f}%  "
            f"robust_acc[72/255]={robust_acc[EPS_LIST[1]]:.2f}%  "
            f"robust_acc[108/255]={robust_acc[EPS_LIST[2]]:.2f}%")

    if results:
        cols = ["run_dir", "layer", "loss", "width", "depth", "epoch",
                "baseline_accuracy", "robust_accuracy_36", "robust_accuracy_72", "robust_accuracy_108"]
        with open(out_csv, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in results:
                f.write(",".join(str(r[c]) for c in cols) + "\n")
        log(f"\nwrote {len(results)} rows to {out_csv}")

        plot_accuracy_comparison(results, plot_path, args.loss)
        log(f"wrote accuracy comparison plot to {plot_path}")
    else:
        log("no checkpoints found")

    log_fh.close()


if __name__ == "__main__":
    main()
