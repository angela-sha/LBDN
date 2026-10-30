import copy
import torch
import torch.nn as nn
from layer import OrthogonFc, OrthogonLin, AolFc, AolLin, SandwichFc, SandwichLin, cayley
from model import getModel
from dataset import getDataLoader
from margin_sweep import build_config, load_checkpoint

"""
Fuse orthogonal linear layer affine map into weight layer.
"""

@torch.no_grad()
def fuse_orthogonal(m):
    W = m.weight.double()
    Q = cayley(m.alpha.double() * W / W.norm())       # compute in fp64, then cast
    lin = nn.Linear(m.in_features, m.out_features, bias=m.bias is not None,
                    device=m.weight.device, dtype=m.weight.dtype)
    lin.weight.copy_((m.scale * Q).to(m.weight.dtype))
    if m.bias is not None:
        lin.bias.copy_(m.bias)
    return nn.Sequential(lin, nn.ReLU()) if isinstance(m, OrthogonFc) else lin

def fuse_model(model):
    for name, child in model.named_children():
        if isinstance(child, (OrthogonLin, OrthogonFc)):
            setattr(model, name, fuse_orthogonal(child))
        else:
            fuse_model(child)
    return model

"""
Fuse AOL scale map into weight layer

"""
@torch.no_grad()
def fuse_aol(m):
    W = m.weights.double()
    t = 1 / torch.sqrt(torch.abs(W.T @ W).sum(1))      # same formula as forward (epsilon is unused there too)
    W_eff = m.scale * W * t[None, :]                    # scale column j by s * t_j
    cout, cin = W.shape
    lin = nn.Linear(cin, cout, bias=True,
                    device=m.weights.device, dtype=m.weights.dtype)
    lin.weight.copy_(W_eff.to(m.weights.dtype))
    lin.bias.copy_(m.bias)
    return nn.Sequential(lin, nn.ReLU()) if isinstance(m, AolFc) else lin

def fuse_aol_model(model):
    for name, child in model.named_children():
        if isinstance(child, (AolLin, AolFc)):
            setattr(model, name, fuse_aol(child))
        else:
            fuse_aol_model(child)
    return model

"""
Fuse Sandwich map into one affine map before and one affine map after the ReLU.

"""
@torch.no_grad()
def _AB(m):
    W = m.weight.double()
    Q = cayley(m.alpha.double() * W / W.norm())
    fout = W.shape[0]
    return Q[:, :fout], Q[:, fout:]          # A, B

def _lin(W, b, dtype, device):
    lin = nn.Linear(W.shape[1], W.shape[0], bias=b is not None, dtype=dtype, device=device)
    lin.weight.copy_(W.to(dtype))
    if b is not None:
        lin.bias.copy_(b.to(dtype))
    return lin

@torch.no_grad()
def fuse_sandwich(m):
    A, B = _AB(m)
    dt, dev, s = m.weight.dtype, m.weight.device, m.scale
    if isinstance(m, SandwichLin):
        W = 2 * s * A.T @ B if m.AB else s * B
        return _lin(W, m.bias, dt, dev)
    # SandwichFc
    b1 = None if m.bias is None else torch.exp(m.psi.double()) * m.bias.double()
    return nn.Sequential(
        _lin(2**0.5 * s * B, b1, dt, dev),
        nn.ReLU(),
        _lin(2**0.5 * A.T, None, dt, dev),
    )

def fuse_sandwich_model(model):
    for name, child in model.named_children():
        if isinstance(child, (SandwichLin, SandwichFc)):
            setattr(model, name, fuse_sandwich(child))
        else:
            fuse_sandwich_model(child)
    return model

# layer type -> (model fuser, layer classes that should be gone after fusing)
FUSERS = {
    'Orthogon': (fuse_model, (OrthogonLin, OrthogonFc)),
    'Aol': (fuse_aol_model, (AolLin, AolFc)),
    'Sandwich': (fuse_sandwich_model, (SandwichLin, SandwichFc)),
}

@torch.no_grad()
def test_fuse(run_dir="saved_models_seed123/mnist/Toy-Orthogon-small-gamma1.0-w128-d2-xent",
              n_samples=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = build_config(run_dir.rstrip("/").split("/")[-1], "mnist", 123)
    config.test_batch_size = n_samples
    model = getModel(config).to(device)
    load_checkpoint(model, f"{run_dir}/model.ckpt", device)
    model.eval()

    fuse, layer_types = FUSERS[config.layer]
    print(f"== {run_dir}")
    fused = fuse(copy.deepcopy(model)).eval()
    assert not any(isinstance(m, layer_types) for m in fused.modules())

    _, testLoader = getDataLoader(config)
    x, _ = next(iter(testLoader))
    x = x.to(device).view(x.shape[0], -1)

    # per-layer activations: each model is fed its own previous output
    h_orig, h_fused = x, x
    for k, (m_o, m_f) in enumerate(zip(model.model, fused.model)):
        h_orig, h_fused = m_o(h_orig), m_f(h_fused)
        err = (h_orig - h_fused).abs().max().item()
        print(f"layer {k} ({type(m_o).__name__}): max abs diff {err:.3e}")
        assert torch.allclose(h_orig, h_fused, atol=1e-5, rtol=1e-4), f"layer {k}: {err:.3e}"

    y_orig, y_fused = model(x), fused(x)

    max_err = (y_orig - y_fused).abs().max().item()
    print(f"max abs diff: {max_err:.3e}")
    assert torch.allclose(y_orig, y_fused, atol=1e-5, rtol=1e-4), max_err
    assert torch.equal(y_orig.argmax(1), y_fused.argmax(1))
    print("fused model matches original")

if __name__ == "__main__":
    for layer in FUSERS:
        test_fuse(f"saved_models_seed123/mnist/Toy-{layer}-small-gamma1.0-w128-d2-xent")
