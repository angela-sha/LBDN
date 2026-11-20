#!/usr/bin/env python3

import argparse
import csv
import torch
import numpy as np
from torch import nn
from pathlib import Path

def split(config):
    ckpt_path = f"{config.train_dir}/model.ckpt"
    split_data = np.load(config.split_data)
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
    else: # backwards compatibility
        model.load_state_dict(ckpt)
    split_inner(model, ckpt)

def split_toy(config):
    pass

def split_inner(model, splits):
    with torch.no_grad():
        layer_counter = 1
        
        for name, module in model.named_modules():
            npz_key = f"n_L{layer_counter}"
            if npz_key not in splits:
                layer_counter += 1
                continue

            repeats = torch.tensor(splits[npz_key], dtype=torch.long)

            raw_weight = module.weight

            num_rows = raw_weight.shape[0]
            if len(repeats) != num_rows:
                raise ValueError

            duplicate_indices = torch.repeat_interleave(torch.arange(num_rows), repeats)
            raw_weight = raw_weight[duplicate_indices, :]

            module.weight = nn.Parameter(raw_weight)

            if module.bias is not None:
                module.bias = nn.Parameter(module.bias[duplicate_indices])

            module.out_features = module.weight.shape[0]
            layer_counter += 1
    return model

def load_splits_csv(file, split_amt="n_p99_eps0.01"):
    with open(file, mode='r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        split_index = header.index(split_amt)
        layers = {}
        for row in csv_reader:
            net_layer = row[0]
            neuron = int(row[3])
            split = float(row[split_amt])
            layers.setdefault(net_layer, [])
            set_extend(layers[net_layer], neuron, split)


def load_splits(file, split_amt="n_p99_eps0.01"):
    with np.load(file) as data:
        return data


def set_extend(list, i, item):
    additional_items = i + 1 - len(list)
    list.extend([0] * additional_items)
    list[i] = item


def load_model(dataset, model, model_file):
    checkpoint = torch.load(model_file, weights_only=False)
    print(checkpoint["model"].keys())
    network = Network(checkpoint["model"], activation_fn=nn.ReLU)
    nonalpha_dict = {k: v for k, v in checkpoint["model"].items() if not k.endswith(".alpha")}
    network.load_state_dict(nonalpha_dict)
    return nonalpha_dict


def main():
    dataset = args.dataset
    model_name = args.arch
    model_dir = args.model_dir
    splits_dir = args.splits_dir

    model = load_model(dataset, model_name, model_dir)
    splits = load_splits(splits_dir / f"{model_name}__model.npz")

    # split(model, splits)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch splitting')
    parser.add_argument('dataset', type=str)
    parser.add_argument('arch', type=str)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('splits_dir', type=Path)
    parser.add_argument('--gpu', default=None, type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()
    main()
