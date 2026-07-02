import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Relaxed Equivariant Graph Neural Network (rEGNN) Layer
# ==============================================================================

class rEGNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, m_dim=64):
        """
        Implementation of the relaxed EGNN (rEGNN) layer from the MiDi paper.
        Uses translation-invariant coordinate features relative to the center of mass (0).
        """
        super().__init__()
        # Message MLP: cat(h_i, h_j, delta_r, y_ij) -> message_ij
        # delta_r consists of: ||r_i - r_j||, ||r_i||, ||r_j||, cos(r_i, r_j) (4 dimensions)
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 4 + edge_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, m_dim),
            nn.SiLU()
        )
        
        # Node update MLP: cat(h_i, msg_i) -> h_new
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + m_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, node_dim)
        )
        
        # Coordinate scalar message MLP: msg_ij -> scalar factor
        self.coord_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, 1, bias=False)
        )

    def forward(self, h, R, Y, node_mask=None):
        """
        Args:
            h (Tensor): Node features of shape [B, N, node_dim]
            R (Tensor): Coordinates of shape [B, N, 3] (assumed zero-centered)
            Y (Tensor): Edge features of shape [B, N, N, edge_dim]
            node_mask (Tensor, optional): Mask of shape [B, N] where 1 is active, 0 is padded.
            
        Returns:
            h_new (Tensor): Updated node features [B, N, node_dim]
            R_new (Tensor): Updated coordinates [B, N, 3]
        """
        B, N, _ = R.shape
        
        if node_mask is None:
            node_mask = torch.ones((B, N), dtype=h.dtype, device=h.device)
            
        # 1. Compute invariant coordinate features (delta_r)
        # R_diff_ij = r_i - r_j
        R_diff = R.unsqueeze(2) - R.unsqueeze(1)  # [B, N, N, 3]
        dist = torch.norm(R_diff, p=2, dim=-1, keepdim=True)  # [B, N, N, 1]
        
        norm_node = torch.norm(R, p=2, dim=-1, keepdim=True)  # [B, N, 1]
        norm_i = norm_node.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, 1]
        norm_j = norm_node.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, 1]
        
        dot = torch.sum(R.unsqueeze(2) * R.unsqueeze(1), dim=-1, keepdim=True)  # [B, N, N, 1]
        cos = dot / (norm_i * norm_j + 1e-6)  # [B, N, N, 1]
        
        delta_r = torch.cat([dist, norm_i, norm_j, cos], dim=-1)  # [B, N, N, 4]
        
        # 2. Form messages
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, node_dim]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, node_dim]
        
        inputs = torch.cat([h_i, h_j, delta_r, Y], dim=-1)  # [B, N, N, node_dim*2 + 4 + edge_dim]
        msg = self.message_mlp(inputs)  # [B, N, N, m_dim]
        
        # Mask out messages for padded nodes
        mask = (node_mask.unsqueeze(2) * node_mask.unsqueeze(1)).unsqueeze(-1)  # [B, N, N, 1]
        msg = msg * mask
        
        # 3. Coordinate Update
        coord_weights = self.coord_mlp(msg)  # [B, N, N, 1]
        coord_weights = coord_weights * mask
        
        # r_i <- r_i + sum_j (coord_weights * (r_j - r_i))
        # Note: (r_j - r_i) is -R_diff
        coord_msg = coord_weights * R_diff  # [B, N, N, 3]
        R_update = coord_msg.sum(dim=2)  # [B, N, 3]
        
        R_new = R + R_update
        
        # Projection onto zero-CoM subspace
        masked_R = R_new * node_mask.unsqueeze(-1)
        nodes_count = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6
        mean_R = masked_R.sum(dim=1, keepdim=True) / nodes_count
        R_new = (R_new - mean_R) * node_mask.unsqueeze(-1)
        
        # 4. Node Features Update
        msg_sum = msg.sum(dim=2)  # [B, N, m_dim]
        node_inputs = torch.cat([h, msg_sum], dim=-1)  # [B, N, node_dim + m_dim]
        h_new = h + self.node_mlp(node_inputs)
        h_new = h_new * node_mask.unsqueeze(-1)
        
        return h_new, R_new


# ==============================================================================
# Graph Transformer Network (Equivariant)
# ==============================================================================

class rEGNNTransformer(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64, num_layers=4,
                 num_node_classes=2):
        """
        Transformer style architecture with stacked rEGNN layers.
        
        Args:
            node_in_dim (int): Dimensionality of input node features (num_node_classes
                for one-hot categorical input).
            edge_in_dim (int): Dimensionality of input edge features.
            hidden_dim (int): Hidden dimension for all layers.
            num_layers (int): Number of rEGNN layers.
            num_node_classes (int): Number of node categories for the classification head.
        """
        super().__init__()
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            rEGNNLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Final prediction heads
        # Coordinate head: outputs a SCALAR per node to preserve rotational equivariance.
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Edge head: predicts reconstructed adjacency matrices (binary logits)
        self.edge_head = nn.Linear(hidden_dim, 1)
        # Node category head: predicts class logits (Active vs Virtual)
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_node_classes)
        )

    def forward(self, X_t, R_t, Y_t, node_mask=None):
        """
        Args:
            X_t (Tensor): Noisy node categories [B, N, num_node_classes] (soft one-hot)
            R_t (Tensor): Noisy coordinates [B, N, 3]
            Y_t (Tensor): Noisy edge adjacency [B, N, N, edge_in_dim]
            node_mask (Tensor, optional): Mask of shape [B, N].
            
        Returns:
            R_pred (Tensor): Reconstructed coordinates [B, N, 3]
            Y_pred (Tensor): Reconstructed edge logits [B, N, N]
            X_pred (Tensor): Reconstructed node category logits [B, N, num_node_classes]
        """
        h = self.node_embed(X_t)
        Y = self.edge_embed(Y_t)
        
        R = R_t
        for layer in self.layers:
            h, R = layer(h, R, Y, node_mask)
            
        # Predict coordinate adjustment
        # Equivariant PosMLP: scalar * unit_direction preserves rotational equivariance
        norm_R = torch.norm(R, p=2, dim=-1, keepdim=True)  # [B, N, 1]
        scale = self.coord_head(h)  # [B, N, 1]
        R_pred = R + (R / (norm_R + 1e-6)) * scale
        
        # Center the final predicted coordinates
        if node_mask is not None:
            masked_R = R_pred * node_mask.unsqueeze(-1)
            nodes_count = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6
            mean_R = masked_R.sum(dim=1, keepdim=True) / nodes_count
            R_pred = (R_pred - mean_R) * node_mask.unsqueeze(-1)
            
        # Predict edge logits
        # Compute edge representations by concatenating node representations
        h_i = h.unsqueeze(2).expand(-1, -1, h.size(1), -1)
        h_j = h.unsqueeze(1).expand(-1, h.size(1), -1, -1)
        edge_rep = h_i + h_j  # Symmetric relation
        Y_pred = self.edge_head(edge_rep).squeeze(-1)  # [B, N, N]
        
        # Force symmetry on predicted adjacency matrix
        Y_pred = 0.5 * (Y_pred + Y_pred.transpose(1, 2))
        
        # Predict node category logits
        X_pred = self.node_head(h)  # [B, N, num_node_classes]
        
        return R_pred, Y_pred, X_pred
