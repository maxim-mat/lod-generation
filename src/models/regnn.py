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
        
        # Edge update MLP: cat(Y_ij, msg_ij) -> Y_ij_updated
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + m_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, edge_dim)
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
            Y_new (Tensor): Updated edge features [B, N, N, edge_dim]
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
        
        # Normalize by number of active neighbors to keep updates O(1)
        num_neighbors = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)  # [B, 1, 1]
        R_update = R_update / num_neighbors
        
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
        
        # 5. Edge Features Update
        edge_input = torch.cat([Y, msg], dim=-1)  # [B, N, N, edge_dim + m_dim]
        Y_new = Y + self.edge_mlp(edge_input)
        Y_new = Y_new * mask
        
        return h_new, R_new, Y_new


# ==============================================================================
# Sinusoidal Timestep Embedding
# ==============================================================================

def sinusoidal_embedding(t, embed_dim):
    """
    Computes sinusoidal positional embeddings for diffusion timesteps.
    
    Args:
        t (Tensor): Timestep indices of shape [B] (long or float).
        embed_dim (int): Dimensionality of the embedding.
        
    Returns:
        Tensor: Sinusoidal embeddings of shape [B, embed_dim].
    """
    half_dim = embed_dim // 2
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim
    )
    args = t.float().unsqueeze(1) * freq.unsqueeze(0)  # [B, half_dim]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, embed_dim]


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
        self.hidden_dim = hidden_dim
        
        # Sinusoidal timestep embedding projection
        self.time_embed_dim = 128
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            rEGNNLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Final prediction heads
        # Coordinate head: pairwise displacement for equivariant coordinate updates
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        # Edge head: predicts reconstructed adjacency matrices (binary logits)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Node category head: predicts class logits (Active vs Virtual)
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_node_classes)
        )

    def forward(self, X_t, R_t, Y_t, t, node_mask=None):
        """
        Args:
            X_t (Tensor): Noisy node categories [B, N, num_node_classes] (soft one-hot)
            R_t (Tensor): Noisy coordinates [B, N, 3]
            Y_t (Tensor): Noisy edge adjacency [B, N, N, edge_in_dim]
            t (Tensor): Diffusion timestep indices [B] (long).
            node_mask (Tensor, optional): Mask of shape [B, N].
            
        Returns:
            R_pred (Tensor): Reconstructed coordinates [B, N, 3]
            Y_pred (Tensor): Reconstructed edge logits [B, N, N]
            X_pred (Tensor): Reconstructed node category logits [B, N, num_node_classes]
        """
        B, N, _ = R_t.shape
        
        # Default mask: all nodes active
        if node_mask is None:
            node_mask = torch.ones((B, N), dtype=R_t.dtype, device=R_t.device)
        
        # Embed timestep and inject into node features
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)  # [B, 128]
        t_emb = self.time_mlp(t_emb)  # [B, hidden_dim]
        
        h = self.node_embed(X_t) + t_emb.unsqueeze(1)  # [B, N, hidden_dim]
        Y = self.edge_embed(Y_t)
        
        R = R_t
        for layer in self.layers:
            h, R, Y = layer(h, R, Y, node_mask)
            
        # Predict coordinate adjustment via pairwise displacements
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, hidden_dim]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, hidden_dim]
        h_pair = torch.cat([h_i, h_j], dim=-1)       # [B, N, N, hidden_dim*2]
        
        R_diff = R.unsqueeze(2) - R.unsqueeze(1)      # [B, N, N, 3]
        pair_mask = (node_mask.unsqueeze(2) * node_mask.unsqueeze(1)).unsqueeze(-1)  # [B, N, N, 1]
        
        coord_weights = self.coord_head(h_pair)        # [B, N, N, 1]
        coord_weights = coord_weights * pair_mask
        R_update = (coord_weights * R_diff).sum(dim=2)  # [B, N, 3]
        
        # Normalize by number of active neighbors to keep updates O(1)
        num_neighbors = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)  # [B, 1, 1]
        R_update = R_update / num_neighbors
        
        R_pred = R + R_update
        
        # Center the final predicted coordinates
        masked_R = R_pred * node_mask.unsqueeze(-1)
        nodes_count = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6
        mean_R = masked_R.sum(dim=1, keepdim=True) / nodes_count
        R_pred = (R_pred - mean_R) * node_mask.unsqueeze(-1)
            
        # Predict edge logits via concatenation
        Y_pred = self.edge_head(h_pair).squeeze(-1)  # [B, N, N]
        
        # Force symmetry on predicted adjacency matrix
        Y_pred = 0.5 * (Y_pred + Y_pred.transpose(1, 2))
        
        # Predict node category logits
        X_pred = self.node_head(h)  # [B, N, num_node_classes]
        
        return R_pred, Y_pred, X_pred
