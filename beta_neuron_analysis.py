"""
For every LBDN `Toy` checkpoint (model.ckpt and every model_epoch*.ckpt snapshot)
under --root, one pass over the test set computes:

  Part 1 (margins -> beta)
    1. The empirical margin (top-1 logit - runner-up logit, as in margin_sweep.py)
       of every test sample, and the margin threshold M_p such that p% of samples
       have margin > M_p, for p in PCTS.
    2. The largest beta satisfying
           M / sqrt(2) >= sum_{l=0}^{L} beta * sqrt(eps * d_l)
       i.e. beta = M / (sqrt(2) * sum_l sqrt(eps * d_l)), for eps in EPS_LIST.
       The sum runs over the L+1 hidden layers of the Toy model (1 input layer
       + `depth` hidden layers, all of width d_l = width); the logit layer is
       not included.

  Part 2 (activations -> n)
    3. sigma_i, the std of neuron i's output over the test set, for every hidden
       neuron (post-ReLU for Plain nets, as in margin_plots.py), and
           n_i = floor(sigma_i * sqrt(2) / beta * N_CONST)
       computed per neuron; neurons with n_i < 1 floor to 0 and so do not
       contribute to the layer sum.
    4. n per neuron and summed over each layer, for every (p, eps) beta.

Outputs (in --out_dir, default <root>/beta_n_analysis):
  beta.csv          one row per (checkpoint, p, eps): M, sum_d, beta
  n_per_layer.csv   one row per (checkpoint, p, eps, layer): sum / mean / min / max of n,
                    and n_active = number of neurons with n >= 1
  per_neuron/<run_dir>__<ckpt>.npz
                    pcts, eps, M [P], beta [P, E], and for each hidden layer
                    sigma_<layer> [d] and n_<layer> [P, E, d]
"""

import os
import re
import glob
import argparse

import torch
import numpy as np

from model import getModel
from dataset import getDataLoader
from margin_sweep import build_config, load_checkpoint, compute_margins
from margin_plots import hooked_layers

PCTS = [99, 95, 90, 85, 80, 75]
EPS_LIST = [0.01, 0.02, 0.03, 0.04, 0.05]
N_CONST = 1.819


def margin_thresholds(margins, pcts):
    """M_p such that p% of samples have margin > M_p, i.e. the (100 - p)th percentile."""
    m = margins.double().numpy()
    return np.array([np.percentile(m, 100 - p) for p in pcts])


def solve_beta(M, eps, layer_widths):
    """Largest beta with M / sqrt(2) >= sum_l beta * sqrt(eps * d_l)."""
    return M / (np.sqrt(2) * np.sum(np.sqrt(eps * np.asarray(layer_widths))))


def neuron_n(sigma, beta):
    """Per-neuron n, floored; neurons with n < 1 become 0 and drop out of any sum."""
    return np.floor(sigma * np.sqrt(2) / beta * N_CONST)


@torch.no_grad()
def compute_margins_and_neuron_std(model, testLoader, device):
    """One pass over the test set: margins, correctness, and per-neuron std of every
    hidden layer's output (the logit layer is dropped)."""
    sums = {}

    def make_hook(name):
        def hook(_m, _in, out):
            out = out.detach().double()
            s = sums.setdefault(name, [0.0, 0.0, 0])
            s[0] = s[0] + out.sum(dim=0)
            s[1] = s[1] + out.pow(2).sum(dim=0)
            s[2] += out.shape[0]
        return hook

    hidden = hooked_layers(model)[:-1]
    handles = [m.register_forward_hook(make_hook(n)) for n, m in hidden]
    margins, correct = compute_margins(model, testLoader, device)
    for h in handles:
        h.remove()

    sigmas = {}
    for name, _ in hidden:
        total, sumsq, count = sums[name]
        mean = total / count
        sigmas[name] = (sumsq / count - mean ** 2).clamp(min=0.0).sqrt().cpu().numpy()
    return margins, correct, sigmas


def ckpt_epoch_from_name(ckpt_path):
    m = re.search(r"model_epoch(\d+)\.ckpt$", ckpt_path)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str,
        default="/project/6105016/shayuxin/verifying-robust-networks/LBDN/saved_models_seed123/mnist")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="where to write results (default: <root>/beta_n_analysis)")
    parser.add_argument("--filter", type=str, default=None,
                        help="only run dirs whose name contains this substring")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir or os.path.join(args.root, "beta_n_analysis")
    os.makedirs(os.path.join(out_dir, "per_neuron"), exist_ok=True)

    beta_rows, layer_rows = [], []
    testLoader = None

    for run_dir_name in sorted(os.listdir(args.root)):
        run_dir = os.path.join(args.root, run_dir_name)
        if not os.path.isdir(run_dir):
            continue
        if args.filter and args.filter not in run_dir_name:
            continue
        config = build_config(run_dir_name, args.dataset, args.seed)
        if config is None:
            continue

        ckpts = sorted(glob.glob(os.path.join(run_dir, "model*.ckpt")))
        for ckpt_path in ckpts:
            ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            if testLoader is None:
                _, testLoader = getDataLoader(config)
            model = getModel(config).to(device)
            try:
                ckpt_epoch = load_checkpoint(model, ckpt_path, device)
            except RuntimeError as e:
                print(f"skip (failed to load {ckpt_name}): {run_dir_name}: {str(e).splitlines()[0]}")
                continue
            # snapshots name the 1-indexed epoch; model.ckpt stores it 0-indexed (if at all)
            epoch = ckpt_epoch_from_name(ckpt_path)
            if epoch is None and ckpt_epoch is not None:
                epoch = ckpt_epoch + 1

            margins, correct, sigmas = compute_margins_and_neuron_std(model, testLoader, device)
            acc = 100.0 * correct.float().mean().item()
            layer_widths = [len(s) for s in sigmas.values()]
            sum_d = int(np.sum(layer_widths))

            M = margin_thresholds(margins, PCTS)                          # [P]
            beta = np.array([[solve_beta(M[i], eps, layer_widths) for eps in EPS_LIST]
                             for i in range(len(PCTS))])                  # [P, E]

            meta = dict(run_dir=run_dir_name, ckpt=ckpt_name, layer=config.layer, loss=config.loss,
                        width=config.width, depth=config.depth, epoch=epoch, accuracy=acc)
            npz = dict(pcts=np.array(PCTS), eps=np.array(EPS_LIST), M=M, beta=beta)
            for i, p in enumerate(PCTS):
                for j, eps in enumerate(EPS_LIST):
                    beta_rows.append(dict(meta, pct=p, M=M[i], eps=eps, sum_d=sum_d, beta=beta[i, j]))

            for name, sigma in sigmas.items():
                n = neuron_n(sigma[None, None, :], beta[:, :, None])         # [P, E, d]
                npz[f"sigma_{name}"] = sigma
                npz[f"n_{name}"] = n
                for i, p in enumerate(PCTS):
                    for j, eps in enumerate(EPS_LIST):
                        n_ij = n[i, j]
                        layer_rows.append(dict(meta, pct=p, eps=eps, beta=beta[i, j], net_layer=name,
                                               d=len(sigma), sigma_sum=sigma.sum(),
                                               n_active=int((n_ij >= 1).sum()),
                                               n_sum=n_ij.sum(), n_mean=n_ij.mean(),
                                               n_min=n_ij.min(), n_max=n_ij.max()))
            np.savez(os.path.join(out_dir, "per_neuron", f"{run_dir_name}__{ckpt_name}.npz"), **npz)

            print(f"{run_dir_name}/{ckpt_name}: acc={acc:.2f}%  "
                  + "  ".join(f"M{p}={M[i]:.4f}" for i, p in enumerate(PCTS)))

    if not beta_rows:
        print("no checkpoints found")
        return

    for fname, rows in (("beta.csv", beta_rows), ("n_per_layer.csv", layer_rows)):
        path = os.path.join(out_dir, fname)
        cols = list(rows[0].keys())
        with open(path, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(str(r[c]) for c in cols) + "\n")
        print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
