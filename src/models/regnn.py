"""Relaxed-equivariant graph transformer (rEGNN), ported from MiDi.

MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation
(Vignac et al., arXiv:2302.09048). Reference implementation:
https://github.com/cvignac/MiDi -- midi/models/transformer_model.py

The network jointly denoises node categories (X), coordinates (pos) and a dense
adjacency matrix (E), conditioned on a global feature vector (y) that carries
the diffusion timestep. Positions are updated by an EGNN-style velocity
(a weighted sum of relative displacements) normalised by SE3Norm, which keeps
the update rotation-equivariant and scale-stable.

Three symmetry groups are supported via `equivariance`:

  "o3"  Faithful MiDi. Every positional feature is rotation-invariant, so the
        learned density satisfies p(G) = p(R.G) for any rotation/reflection R.
        Correct for molecules, which have no preferred orientation.

  "so2" Equivariant to yaw about the vertical axis only. Buildings have a canonical
        vertical: walls are vertical, roofs and ground are horizontal, and
        `post_process` reads surface semantics off the sign of `normal[2]`. Under
        "o3" a generated building comes out in an arbitrary pose, and the pose is
        not recoverable afterwards for the common case -- an LOD1 building is an
        extruded footprint, and a rectangular footprint's 1-skeleton is a box, which
        has no intrinsic "up". This mode extends MiDi's own canonical-pose argument
        (sec. 4.3: relaxing translation invariance is valid because zero-CoM defines
        a canonical pose for the translation group) from translations to the gravity
        subgroup.

  "se2" so2's feature set, but the zero-CoM projections remove the xy mean
        only: z is absolute (standardised upstream by the train-split `z_shift`).
        The quotiented group is exactly SE(2) — yaw plus horizontal translation —
        so ground height and building elevation become model-visible.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm, Linear

from src.models.layers import (
    EtoX,
    Etoy,
    PositionsMLP,
    SE3Norm,
    Xtoy,
    masked_softmax,
    remove_mean_with_mask,
)


def mask_graph(X, E, pos, node_mask, xy_only=False):
    """Apply node/edge masks, drop self-loops, and re-centre positions."""
    x_mask = node_mask.unsqueeze(-1)                 # [B, N, 1]
    e_mask1 = x_mask.unsqueeze(2)                    # [B, N, 1, 1]
    e_mask2 = x_mask.unsqueeze(1)                    # [B, 1, N, 1]
    n = node_mask.shape[1]
    diag_mask = ~torch.eye(n, dtype=torch.bool, device=node_mask.device)
    diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]

    X = X * x_mask
    E = E * e_mask1 * e_mask2 * diag_mask
    pos = remove_mean_with_mask(pos * x_mask, node_mask, xy_only=xy_only)
    return X, E, pos


class NodeEdgeBlock(nn.Module):
    """Self-attention over nodes that also updates edges, positions and globals."""

    def __init__(self, dx, de, dy, n_head, equivariance="so2"):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- n_head: {n_head}"
        if equivariance not in ("o3", "so2", "se2"):
            raise ValueError(
                f"Unknown equivariance: {equivariance!r}. Must be 'o3', 'so2' or 'se2'."
            )
        self.dx, self.de, self.dy = dx, de, dy
        self.df = int(dx / n_head)
        self.n_head = n_head
        self.equivariance = equivariance

        self.in_E = Linear(de, de)

        # FiLM X to E
        self.x_e_mul1 = Linear(dx, de)
        self.x_e_mul2 = Linear(dx, de)

        # Geometry encoding. "so2"/"se2" add the height z_i and the vertical/
        # horizontal split of each displacement: yaw-invariant, not O(3)-invariant.
        node_geom_dim = 2 if equivariance in ("so2", "se2") else 1
        pair_geom_dim = 4 if equivariance in ("so2", "se2") else 2
        self.lin_dist1 = Linear(pair_geom_dim, de)
        self.lin_norm_pos1 = Linear(node_geom_dim, de)
        self.lin_norm_pos2 = Linear(node_geom_dim, de)
        self.dist_add_e = Linear(de, de)
        self.dist_mul_e = Linear(de, de)

        # Attention
        self.k = Linear(dx, dx)
        self.q = Linear(dx, dx)
        self.v = Linear(dx, dx)
        self.a = Linear(dx, n_head, bias=False)
        self.out = Linear(dx * n_head, dx)

        self.e_att_mul = Linear(de, n_head)
        self.pos_att_mul = Linear(de, n_head)
        self.e_x_mul = EtoX(de, dx)
        self.pos_x_mul = EtoX(de, dx)

        # FiLM y to E / y to X
        self.y_e_mul = Linear(dy, de)
        self.y_e_add = Linear(dy, de)
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Global features
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)
        self.dist_y = Etoy(de, dy)

        # Positions
        self.e_pos1 = Linear(de, de, bias=False)
        self.e_pos2 = Linear(de, 1, bias=False)

        self.x_out = Linear(dx, dx)
        self.e_out = Linear(de, de)
        self.y_out = nn.Sequential(Linear(dy, dy), nn.ReLU(), Linear(dy, dy))

    def forward(self, X, E, y, pos, node_mask):
        """X: [B,N,dx]  E: [B,N,N,de]  y: [B,dy]  pos: [B,N,3]  node_mask: [B,N]"""
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)

        # 0. Rotation-invariant geometry: pairwise distances, norms and cosines.
        pos = pos * x_mask
        norm_pos = torch.norm(pos, dim=-1, keepdim=True)                 # [B, N, 1]
        normalized_pos = pos / (norm_pos + 1e-7)

        pairwise_dist = torch.cdist(pos, pos).unsqueeze(-1).float()
        cosines = torch.sum(normalized_pos.unsqueeze(1) * normalized_pos.unsqueeze(2), dim=-1, keepdim=True)
        pos_info = torch.cat((pairwise_dist, cosines), dim=-1)
        node_info = norm_pos

        if self.equivariance in ("so2", "se2"):
            # Yaw-invariant, but not O(3)-invariant: the height of a node and the
            # vertical/horizontal split of each displacement. |dz| rather than dz
            # keeps the pair features symmetric in (i, j), as E must be.
            z = pos[..., 2:3]                                            # [B, N, 1]
            dz = (z.unsqueeze(1) - z.unsqueeze(2)).abs()                 # [B, N, N, 1]
            horiz_dist = torch.cdist(pos[..., :2], pos[..., :2]).unsqueeze(-1).float()
            pos_info = torch.cat((pos_info, dz, horiz_dist), dim=-1)
            node_info = torch.cat((norm_pos, z), dim=-1)

        norm1 = self.lin_norm_pos1(node_info)
        norm2 = self.lin_norm_pos2(node_info)
        dist1 = F.relu(self.lin_dist1(pos_info) + norm1.unsqueeze(2) + norm2.unsqueeze(1)) * e_mask1 * e_mask2

        # 1. Edges: FiLM in the node features, the geometry, then the globals.
        Y = self.in_E(E)

        x_e_mul1 = self.x_e_mul1(X) * x_mask
        x_e_mul2 = self.x_e_mul2(X) * x_mask
        Y = Y * x_e_mul1.unsqueeze(1) * x_e_mul2.unsqueeze(2) * e_mask1 * e_mask2

        dist_add = self.dist_add_e(dist1)
        dist_mul = self.dist_mul_e(dist1)
        Y = (Y + dist_add + Y * dist_mul) * e_mask1 * e_mask2

        y_e_add = self.y_e_add(y).unsqueeze(1).unsqueeze(1)
        y_e_mul = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        E = (Y + y_e_add + Y * y_e_mul) * e_mask1 * e_mask2
        Eout = self.e_out(E) * e_mask1 * e_mask2

        # 2. Nodes: attention modulated by edge and positional features.
        Q = (self.q(X) * x_mask).unsqueeze(2)                            # [B, N, 1, dx]
        K = (self.k(X) * x_mask).unsqueeze(1)                            # [B, 1, N, dx]
        prod = Q * K / math.sqrt(Y.size(-1))
        a = self.a(prod) * e_mask1 * e_mask2                             # [B, N, N, n_head]

        a = a + self.e_att_mul(E) * a
        a = a + self.pos_att_mul(dist1) * a
        a = a * e_mask1 * e_mask2

        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        alpha = masked_softmax(a, softmax_mask, dim=2).unsqueeze(-1)     # [B, N, N, n_head, 1]
        V = (self.v(X) * x_mask).unsqueeze(1).unsqueeze(3)               # [B, 1, N, 1, dx]
        weighted_V = (alpha * V).sum(dim=2).flatten(start_dim=2)         # [B, N, n_head * dx]
        weighted_V = self.out(weighted_V) * x_mask

        weighted_V = weighted_V + self.e_x_mul(E, e_mask2) * weighted_V
        weighted_V = weighted_V + self.pos_x_mul(dist1, e_mask2) * weighted_V

        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = weighted_V * (yx2 + 1) + yx1
        Xout = self.x_out(newX) * x_mask

        # 3. Globals.
        new_y = (self.y_y(y) + self.x_y(newX, x_mask) + self.e_y(Y, e_mask1, e_mask2)
                 + self.dist_y(dist1, e_mask1, e_mask2))
        yout = self.y_out(new_y)

        # 4. Positions: EGNN velocity, sum_j m_ij (pos_i - pos_j).
        pos1 = pos.unsqueeze(1).expand(-1, n, -1, -1)
        pos2 = pos.unsqueeze(2).expand(-1, -1, n, -1)
        delta_pos = pos2 - pos1

        messages = self.e_pos2(F.relu(self.e_pos1(Y)))                   # [B, N, N, 1]
        vel = (messages * delta_pos).sum(dim=2) * x_mask
        vel = remove_mean_with_mask(vel, node_mask, xy_only=self.equivariance == "se2")

        return Xout, Eout, yout, vel


class XEyTransformerLayer(nn.Module):
    """Pre-attention block + feed-forward, with SE3Norm on the position update."""

    def __init__(self, dx, de, dy, n_head, dim_ffX=256, dim_ffE=64, dim_ffy=256,
                 dropout=0.1, layer_norm_eps=1e-5, equivariance="so2"):
        super().__init__()
        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, equivariance=equivariance)

        self.linX1 = Linear(dx, dim_ffX)
        self.linX2 = Linear(dim_ffX, dx)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.norm_pos1 = SE3Norm(eps=layer_norm_eps)

        self.linE1 = Linear(de, dim_ffE)
        self.linE2 = Linear(dim_ffE, de)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy)
        self.lin_y2 = Linear(dim_ffy, dy)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X, E, y, pos, node_mask):
        x_mask = node_mask.unsqueeze(-1)
        newX, newE, new_y, vel = self.self_attn(X, E, y, pos, node_mask=node_mask)

        X = self.normX1(X + self.dropoutX1(newX))
        new_pos = self.norm_pos1(vel, x_mask) + pos
        E = self.normE1(E + self.dropoutE1(newE))
        y = self.norm_y1(y + self.dropout_y1(new_y))

        ff_outputX = self.dropoutX3(self.linX2(self.dropoutX2(self.activation(self.linX1(X)))))
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.dropoutE3(self.linE2(self.dropoutE2(self.activation(self.linE1(E)))))
        E = self.normE2(E + ff_outputE)
        E = 0.5 * (E + torch.transpose(E, 1, 2))

        ff_output_y = self.dropout_y3(self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y)))))
        y = self.norm_y2(y + ff_output_y)

        X, E, new_pos = mask_graph(
            X, E, new_pos, node_mask, xy_only=self.self_attn.equivariance == "se2"
        )
        return X, E, y, new_pos


class rEGNNTransformer(nn.Module):
    """Denoising network predicting the clean (pos, E, X) from a noisy graph."""

    def __init__(self, num_node_classes=2, num_edge_classes=2, hidden_dim=64,
                 edge_dim=32, global_dim=32, n_head=8, num_layers=4, pos_mlp_dim=16,
                 dropout=0.1, equivariance="so2"):
        """
        Args:
            num_node_classes (int): Node categories (Active / Virtual).
            num_edge_classes (int): Edge categories (no-edge / edge).
            hidden_dim (int): Node channel width (dx). Must be divisible by n_head.
            edge_dim (int): Edge channel width (de). Edge tensors are [B,N,N,de],
                so this dominates memory -- keep it small.
            global_dim (int): Global feature width (dy).
            n_head (int): Attention heads.
            num_layers (int): Number of XEyTransformerLayer blocks.
            pos_mlp_dim (int): Hidden width of the input/output PositionsMLP.
            equivariance (str): 'so2' (yaw only, buildings have a canonical
                vertical), 'se2' (so2 features, xy-only CoM, absolute z) or
                'o3' (faithful MiDi). See the module docstring.
        """
        super().__init__()
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.xy_only = equivariance == "se2"

        act = nn.ReLU()
        # The global feature carries only the normalised timestep.
        self.mlp_in_y = nn.Sequential(
            nn.Linear(1, global_dim), act, nn.Linear(global_dim, global_dim), act
        )
        self.mlp_in_X = nn.Sequential(
            nn.Linear(num_node_classes, hidden_dim), act, nn.Linear(hidden_dim, hidden_dim), act
        )
        self.mlp_in_E = nn.Sequential(
            nn.Linear(num_edge_classes, edge_dim), act, nn.Linear(edge_dim, edge_dim), act
        )
        self.mlp_in_pos = PositionsMLP(pos_mlp_dim, xy_only=self.xy_only)

        self.tf_layers = nn.ModuleList([
            XEyTransformerLayer(
                dx=hidden_dim, de=edge_dim, dy=global_dim, n_head=n_head,
                dim_ffX=hidden_dim, dim_ffE=edge_dim, dim_ffy=2 * global_dim,
                dropout=dropout, equivariance=equivariance,
            )
            for _ in range(num_layers)
        ])

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), act, nn.Linear(hidden_dim, num_node_classes)
        )
        self.mlp_out_E = nn.Sequential(
            nn.Linear(edge_dim, edge_dim), act, nn.Linear(edge_dim, num_edge_classes)
        )
        self.mlp_out_pos = PositionsMLP(pos_mlp_dim, xy_only=self.xy_only)

    def forward(self, X_t, R_t, Y_t, t_norm, node_mask=None):
        """
        Args:
            X_t (Tensor): Noisy node categories [B, N, num_node_classes], one-hot.
            R_t (Tensor): Noisy coordinates [B, N, 3], zero centre of mass.
            Y_t (Tensor): Noisy adjacency [B, N, N, num_edge_classes], one-hot,
                symmetric with a zero diagonal.
            t_norm (Tensor): Normalised timestep t / T, shape [B, 1].
            node_mask (Tensor, optional): [B, N], 1 for nodes taking part in
                message passing. Defaults to all ones.

        Returns:
            R_pred (Tensor): Predicted clean coordinates [B, N, 3].
            Y_pred (Tensor): Edge logits [B, N, N, num_edge_classes], symmetric,
                zero on the diagonal.
            X_pred (Tensor): Node category logits [B, N, num_node_classes].
        """
        bs, n, _ = R_t.shape
        if node_mask is None:
            node_mask = torch.ones((bs, n), dtype=R_t.dtype, device=R_t.device)

        diag_mask = ~torch.eye(n, device=X_t.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1)

        # Skip connections straight from the one-hot inputs to the logits.
        X_to_out, E_to_out = X_t, Y_t

        new_E = self.mlp_in_E(Y_t)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        X, E, pos = mask_graph(
            self.mlp_in_X(X_t), new_E, self.mlp_in_pos(R_t, node_mask), node_mask,
            xy_only=self.xy_only,
        )
        y = self.mlp_in_y(t_norm)

        for layer in self.tf_layers:
            X, E, y, pos = layer(X, E, y, pos, node_mask)

        X = self.mlp_out_X(X) + X_to_out
        E = (self.mlp_out_E(E) + E_to_out) * diag_mask
        E = 0.5 * (E + torch.transpose(E, 1, 2))
        pos = self.mlp_out_pos(pos, node_mask)

        X, E, pos = mask_graph(X, E, pos, node_mask, xy_only=self.xy_only)
        return pos, E, X
