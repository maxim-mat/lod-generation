"""Building blocks ported from MiDi.

MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation
(Vignac et al., arXiv:2302.09048). Reference implementation:
https://github.com/cvignac/MiDi -- midi/models/layers.py, midi/utils.py

The positional layers here are equivariant to rotations but *not* to
translations: they consume ||pos_i|| as a feature, which is only meaningful
once the point cloud is centred. Translation is instead quotiented out by
keeping every position tensor on the zero centre-of-mass subspace.
"""
import torch
import torch.nn as nn
from torch.nn import init


# Guards the sqrt below: d/dv sqrt(v) is unbounded as v -> 0, and a constant
# feature across nodes has exactly zero variance.
_VAR_EPS = 1e-6


def remove_mean_with_mask(x, node_mask, xy_only=False):
    """Project onto the zero centre-of-mass subspace, ignoring padded nodes.

    Args:
        x (Tensor): [B, N, D]
        node_mask (Tensor): [B, N], 1 for active nodes.
        xy_only (bool): se2 mode — remove the mean of the first two components
            only; the last (z) passes through untouched.
    """
    mask = node_mask.unsqueeze(-1).to(x.dtype)
    num_nodes = mask.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (x * mask).sum(dim=1, keepdim=True) / num_nodes
    if xy_only:
        mean = torch.cat((mean[..., :2], torch.zeros_like(mean[..., 2:])), dim=-1)
    return (x - mean) * mask


def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


class SE3Norm(nn.Module):
    """Normalise positions by their mean norm, with a single learnable scale.

    Rotation-equivariant by construction: `pos` is only ever scaled by a
    rotation-invariant quantity.
    """

    def __init__(self, eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.normalized_shape = (1,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)

    def forward(self, pos, x_mask):
        """pos: [B, N, 3]; x_mask: [B, N, 1]"""
        norm = torch.norm(pos, dim=-1, keepdim=True)                     # [B, N, 1]
        mean_norm = torch.sum(norm, dim=1, keepdim=True) / torch.sum(x_mask, dim=1, keepdim=True)
        return self.weight * pos / (mean_norm + self.eps)

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}".format(**self.__dict__)


class PositionsMLP(nn.Module):
    """Rescale each position by an MLP of its norm, then re-centre."""

    def __init__(self, hidden_dim, eps=1e-5, xy_only=False):
        super().__init__()
        self.eps = eps
        self.xy_only = xy_only
        self.mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, pos, node_mask):
        norm = torch.norm(pos, dim=-1, keepdim=True)                     # [B, N, 1]
        new_norm = self.mlp(norm)                                        # [B, N, 1]
        new_pos = pos * new_norm / (norm + self.eps)
        new_pos = new_pos * node_mask.unsqueeze(-1)
        return remove_mean_with_mask(new_pos, node_mask, xy_only=self.xy_only)


class Xtoy(nn.Module):
    """Map node features to global features.

    Deviates from MiDi: upstream pools a *variance* alongside mean/min/max and
    feeds all four to one Linear. The variance is degree 2 in the activations
    while its siblings are degree 1, which makes the global-feature branch
    overflow fp32 once activations leave the O(1) range molecules keep them in.
    Taking the square root restores a common scale; a standard deviation carries
    the same information as a variance.
    """

    def __init__(self, dx, dy):
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, x_mask):
        x_mask = x_mask.expand(-1, -1, X.shape[-1])
        float_imask = 1 - x_mask.float()
        m = X.sum(dim=1) / torch.sum(x_mask, dim=1)
        mi = (X + 1e5 * float_imask).min(dim=1)[0]
        ma = (X - 1e5 * float_imask).max(dim=1)[0]
        var = torch.sum(((X - m[:, None, :]) ** 2) * x_mask, dim=1) / torch.sum(x_mask, dim=1)
        std = torch.sqrt(var + _VAR_EPS)
        return self.lin(torch.hstack((m, mi, ma, std)))


class Etoy(nn.Module):
    """Map edge features to global features.

    Pools a standard deviation rather than MiDi's variance; see `Xtoy`.
    """

    def __init__(self, d, dy):
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E, e_mask1, e_mask2):
        mask = (e_mask1 * e_mask2).expand(-1, -1, -1, E.shape[-1])
        float_imask = 1 - mask.float()
        divide = torch.sum(mask, dim=(1, 2))
        m = E.sum(dim=(1, 2)) / divide
        mi = (E + 1e5 * float_imask).min(dim=2)[0].min(dim=1)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0].max(dim=1)[0]
        var = torch.sum(((E - m[:, None, None, :]) ** 2) * mask, dim=(1, 2)) / divide
        std = torch.sqrt(var + _VAR_EPS)
        return self.lin(torch.hstack((m, mi, ma, std)))


class EtoX(nn.Module):
    """Aggregate edge features onto their incident nodes.

    Pools a standard deviation rather than MiDi's variance; see `Xtoy`.
    """

    def __init__(self, de, dx):
        super().__init__()
        self.lin = nn.Linear(4 * de, dx)

    def forward(self, E, e_mask2):
        bs, n, _, de = E.shape
        e_mask2 = e_mask2.expand(-1, n, -1, de)
        float_imask = 1 - e_mask2.float()
        m = E.sum(dim=2) / torch.sum(e_mask2, dim=2)
        mi = (E + 1e5 * float_imask).min(dim=2)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0]
        var = torch.sum(((E - m[:, :, None, :]) ** 2) * e_mask2, dim=2) / torch.sum(e_mask2, dim=2)
        std = torch.sqrt(var + _VAR_EPS)
        return self.lin(torch.cat((m, mi, ma, std), dim=2))
