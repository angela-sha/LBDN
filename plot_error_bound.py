"""
Plot the layerwise error bound

    ||e_L||  <=  sum_{l=1}^{L} beta_l * sqrt(k * d_l)

Assuming beta_l = beta and d_l = d are constant across layers, the sum
collapses to

    ||e_L||  <=  L * beta * sqrt(k * d)

x-axis : k
curves : beta
panels : d = 256 and d = 512
figures: one per value of L

All figures share the same y-limits so the growth with L is visible when
the images are viewed side by side.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Params
# ----------------------------------------------------------------------
L_VALUES = [2, 4, 8]               # one figure per L
K = np.linspace(0.01, 0.15, 400)   # x-axis
BETAS = np.linspace(0.0, 0.05, 6)  # one curve per beta
DIMS = [256, 512]                  # one panel per d


def bound(k, beta, d, n_layers):
    """Upper bound on ||e_L|| for constant beta and d."""
    return n_layers * beta * np.sqrt(k * d)


# Common y-limit: worst case is max L, max beta, max d, max k
Y_MAX = bound(K[-1], BETAS[-1], max(DIMS), max(L_VALUES)) * 1.05

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
cmap = plt.get_cmap("viridis")

for L in L_VALUES:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, d in zip(axes, DIMS):
        for i, beta in enumerate(BETAS):
            ax.plot(
                K,
                bound(K, beta, d, L),
                color=cmap(i / (len(BETAS) - 1)),
                lw=2,
                label=rf"$\beta_\ell = {beta:.3f}$",
            )
        ax.set_title(rf"$d_\ell = {d}$")
        ax.set_xlabel(r"$k$")
        ax.set_xlim(K[0], K[-1])
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(r"$\|e_L\|$ (upper bound)")
    axes[0].set_ylim(0, Y_MAX)
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle(
        rf"$\|e_L\| \leq \sum_{{\ell=1}}^{{L}} \beta_\ell \sqrt{{k\,d_\ell}}$   "
        rf"($L={L}$)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(f"error_bound_L{L}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)