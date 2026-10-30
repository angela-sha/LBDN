"""
Flatten the per_neuron/*.npz files written by beta_neuron_analysis.py into one CSV per
checkpoint, one row per hidden neuron, so n can be indexed straight into the network.

Columns:
  net_layer      L1, L2, ... as in the npz (hooked_layers order, logits excluded)
  module         key prefix in model.state_dict() of the hooked module, e.g. "model.2";
                 sigma/n were measured on this module's output
  weight_module  key prefix of the module whose weight row `neuron` produces this unit
                 (the Linear before the ReLU for Plain nets, otherwise == module)
  neuron         output index i within the layer; n_<layer>[..., i] and sigma_<layer>[i]
  sigma          std of the neuron's output over the test set
  n_p<p>_eps<eps> one column per (p, eps) pair; beta for that pair is in beta.csv /
                 npz["beta"]

Usage:
  python per_neuron_to_csv.py [--in_dir <root>/beta_n_analysis/per_neuron] [--out_dir ...]
Also writes all.csv (every checkpoint stacked, with run_dir/ckpt columns) in --out_dir.
"""

import os
import glob
import argparse

import numpy as np
import torch.nn as nn

from model import getModel
from margin_sweep import build_config
from margin_plots import hooked_layers


def module_keys(config):
    """{net_layer: (module, weight_module)} for the hidden layers of a Toy model."""
    model = getModel(config)
    children = list(model.model.children())
    keys = {}
    for name, m in hooked_layers(model)[:-1]:
        k = next(i for i, c in enumerate(children) if c is m)
        w = k - 1 if isinstance(m, nn.ReLU) else k
        keys[name] = (f"model.{k}", f"model.{w}")
    return keys


def npz_rows(path, keys):
    d = np.load(path)
    pcts, eps = d["pcts"], d["eps"]
    n_cols = [f"n_p{p}_eps{e:g}" for p in pcts for e in eps]
    rows = []
    for name, (module, weight_module) in keys.items():
        sigma = d[f"sigma_{name}"]
        n = d[f"n_{name}"].reshape(len(pcts) * len(eps), -1)        # [P*E, d], p-major
        for i in range(len(sigma)):
            row = dict(net_layer=name, module=module, weight_module=weight_module,
                       neuron=i, sigma=sigma[i])
            row.update({c: int(n[j, i]) for j, c in enumerate(n_cols)})
            rows.append(row)
    return rows


def write_csv(path, rows):
    cols = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", type=str,
        default="/project/6105016/shayuxin/verifying-robust-networks/LBDN/saved_models_seed123/mnist/beta_n_analysis/per_neuron")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="where to write CSVs (default: <in_dir>/../per_neuron_csv)")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.in_dir.rstrip("/")), "per_neuron_csv")
    os.makedirs(out_dir, exist_ok=True)

    all_rows, keys_cache = [], {}
    for path in sorted(glob.glob(os.path.join(args.in_dir, "*.npz"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        run_dir_name, ckpt_name = stem.split("__")
        if run_dir_name not in keys_cache:
            config = build_config(run_dir_name, args.dataset, args.seed)
            if config is None:
                print(f"skip (unparseable run dir): {stem}")
                continue
            keys_cache[run_dir_name] = module_keys(config)

        rows = npz_rows(path, keys_cache[run_dir_name])
        write_csv(os.path.join(out_dir, f"{stem}.csv"), rows)
        all_rows += [dict(run_dir=run_dir_name, ckpt=ckpt_name, **r) for r in rows]
        print(f"{stem}: {len(rows)} neurons")

    if all_rows:
        write_csv(os.path.join(out_dir, "all.csv"), all_rows)
        print(f"wrote {len(all_rows)} rows to {os.path.join(out_dir, 'all.csv')}")


if __name__ == "__main__":
    main()
