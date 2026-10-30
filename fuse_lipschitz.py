import torch 
from layer import OrthogonFc, OrthogonLin

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