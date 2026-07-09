"""Mixed continuous/discrete noise model, ported from MiDi.

MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation
(Vignac et al., arXiv:2302.09048). Reference implementation:
https://github.com/cvignac/MiDi -- midi/diffusion/noise_model.py and
midi/diffusion/diffusion_utils.py

Coordinates follow a variance-preserving Gaussian process where `alpha_bar` is
the signal *amplitude* (so alpha_bar^2 + sigma_bar^2 = 1), not the DDPM
`alpha_bar` (which is the amplitude squared). Node categories and edges follow
D3PM discrete diffusion with transition matrices

    Q_t     = beta_t * P     + (1 - beta_t) * I
    Qbar_t  = (1 - a_t) * P  + a_t * I

whose limit distribution is the row distribution `P`. With `transition="marginal"`
P is the empirical class distribution of the training set (MiDi's default), so a
sparse adjacency matrix decays toward "mostly no edge" rather than toward a
50%-dense graph as a uniform prior would give.

X and E take independent transitions here, because in this adaptation they are not
analogous variables: X is a node-existence bit, so a uniform prior on it asserts
that half of all n_max nodes exist, while a uniform prior on E asserts a 50%-dense
adjacency matrix. Their pathologies are unrelated and must be ablated separately.

Each feature gets its own cosine schedule exponent `nu`. A *larger* nu retains more
signal: at t = T/2, alpha_bar is 0.919 / 0.715 / 0.494 for nu = 2.5 / 1.5 / 1.0. So
MiDi's nu_pos=2.5 > nu_e=1.5 > nu_x=1.0 noises coordinates the *slowest*, which
means they are resolved *first* when running the chain in reverse, followed by edges
and then node categories (paper, sec. 4.2).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import remove_mean_with_mask

# Order matters: it indexes the schedule buffers.
FEATURES = ("p", "x", "e")
_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURES)}


def cosine_beta_schedule_discrete(timesteps, nu_arr, s=0.008):
    """Cosine schedule raised to a per-feature exponent `nu`.

    Returns betas of shape [timesteps + 1, len(nu_arr)].
    """
    steps = timesteps + 2
    x = np.expand_dims(np.linspace(0, steps, steps), 0)          # [1, steps]
    nu_arr = np.expand_dims(np.array(nu_arr), 1)                 # [components, 1]

    alphas_cumprod = np.cos(0.5 * np.pi * (((x / steps) ** nu_arr) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / np.expand_dims(alphas_cumprod[:, 0], 1)
    alphas = alphas_cumprod[:, 1:] / alphas_cumprod[:, :-1]
    return np.swapaxes(1 - alphas, 0, 1)                          # [steps - 1, components]


def sample_discrete_features(probX, probE, node_mask):
    """Sample node/edge classes from per-entry categorical distributions.

    Edges are sampled on the upper triangle and mirrored, so the result is
    symmetric with a zero diagonal.
    """
    bs, n = node_mask.shape
    mask_bool = node_mask.bool()

    probX = probX.clone()
    probX[~mask_bool] = 1 / probX.shape[-1]
    X_t = probX.reshape(bs * n, -1).multinomial(1).reshape(bs, n)

    inverse_edge_mask = ~(mask_bool.unsqueeze(1) * mask_bool.unsqueeze(2))
    diag_mask = torch.eye(n, dtype=torch.bool, device=node_mask.device).unsqueeze(0).expand(bs, -1, -1)

    probE = probE.clone()
    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask] = 1 / probE.shape[-1]
    E_t = probE.reshape(bs * n * n, -1).multinomial(1).reshape(bs, n, n)

    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + torch.transpose(E_t, 1, 2)
    return X_t, E_t


def zero_diagonal(E):
    """Zero the self-loop entries of a dense edge tensor [B, N, N, K]."""
    n = E.shape[1]
    diag = torch.eye(n, dtype=torch.bool, device=E.device).view(1, n, n, 1)
    return E.masked_fill(diag, 0.0)


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """q(s | t, x0) for every possible value of x0.

    Computes  x_t Qt^T * x0 Qsb / (x0 Qtb x_t^T)  for each x0, yielding
    [bs, N, d0, d_{t-1}]. Summing this against the model's predicted p(x0)
    gives the exact reverse posterior, rather than an approximation of it.
    """
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)     # [bs, N, dt]

    left_term = X_t @ Qt.transpose(-1, -2)                           # [bs, N, d_t-1]
    numerator = left_term.unsqueeze(dim=2) * Qsb.unsqueeze(1)        # [bs, N, d0, d_t-1]

    prod = Qtb @ X_t.transpose(-1, -2)                               # [bs, d0, N]
    denominator = prod.transpose(-1, -2).unsqueeze(-1).clone()       # [bs, N, d0, 1]
    denominator[denominator == 0] = 1e-6
    return numerator / denominator


class GraphNoiseModel(nn.Module):
    """Forward noising and reverse posterior sampling for (pos, X, E)."""

    def __init__(self, T, x_marginals, e_marginals, transition_x="marginal",
                 transition_e="marginal", nu_pos=2.5, nu_x=1.0, nu_e=1.5):
        super().__init__()
        for name, value in (("transition_x", transition_x), ("transition_e", transition_e)):
            if value not in ("marginal", "uniform"):
                raise ValueError(
                    f"Unknown {name}: {value!r}. Must be 'marginal' or 'uniform'."
                )
        self.T = T
        self.transition_x = transition_x
        self.transition_e = transition_e
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)

        betas = cosine_beta_schedule_discrete(T, (nu_pos, nu_x, nu_e))
        betas = torch.from_numpy(betas).float()                      # [T + 1, 3]
        alphas = 1 - torch.clamp(betas, min=0, max=0.9999)
        log_alpha_bar = torch.cumsum(torch.log(alphas), dim=0)

        self.register_buffer("_betas", betas)
        self.register_buffer("_log_alpha_bar", log_alpha_bar)
        self.register_buffer("_alphas_bar", torch.exp(log_alpha_bar))
        self.register_buffer("_sigma2_bar", -torch.expm1(2 * log_alpha_bar))
        self.register_buffer("_sigma_bar", torch.sqrt(-torch.expm1(2 * log_alpha_bar)))

        # The seed of the reverse chain must be the stationary distribution of the
        # forward chain, so the limit marginals follow the transition choice.
        if transition_x == "uniform":
            x_marginals = torch.ones(self.X_classes) / self.X_classes
        if transition_e == "uniform":
            e_marginals = torch.ones(self.E_classes) / self.E_classes

        self.register_buffer("X_marginals", x_marginals.float())
        self.register_buffer("E_marginals", e_marginals.float())
        # Every row of P is the limit distribution.
        Px = x_marginals.float().unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        Pe = e_marginals.float().unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        self.register_buffer("Px", Px.contiguous())
        self.register_buffer("Pe", Pe.contiguous())

    # ------------------------------------------------------------------
    # Schedule accessors. `t_int` is [B, 1] long; returns [B, 1].
    # ------------------------------------------------------------------

    def get_beta(self, t_int, key):
        return self._betas[t_int.long()][..., _FEATURE_INDEX[key]].float()

    def get_alpha_bar(self, t_int, key):
        return self._alphas_bar[t_int.long()][..., _FEATURE_INDEX[key]].float()

    def get_sigma_bar(self, t_int, key):
        return self._sigma_bar[t_int.long()][..., _FEATURE_INDEX[key]].float()

    def get_sigma2_bar(self, t_int, key):
        return self._sigma2_bar[t_int.long()][..., _FEATURE_INDEX[key]].float()

    def get_alpha_pos_ts(self, t_int, s_int):
        log_a_bar = self._log_alpha_bar[..., _FEATURE_INDEX["p"]]
        return torch.exp(log_a_bar[t_int.long()] - log_a_bar[s_int.long()]).float()

    def get_alpha_pos_ts_sq(self, t_int, s_int):
        log_a_bar = self._log_alpha_bar[..., _FEATURE_INDEX["p"]]
        return torch.exp(2 * log_a_bar[t_int.long()] - 2 * log_a_bar[s_int.long()]).float()

    def get_sigma_pos_sq_ratio(self, s_int, t_int):
        log_a_bar = self._log_alpha_bar[..., _FEATURE_INDEX["p"]]
        s2_s = -torch.expm1(2 * log_a_bar[s_int.long()])
        s2_t = -torch.expm1(2 * log_a_bar[t_int.long()])
        return torch.exp(torch.log(s2_s) - torch.log(s2_t)).float()

    def get_x_pos_prefactor(self, s_int, t_int):
        """a_s * (1 - (a_t/a_s)^2 * (s_s^2 / s_t^2))"""
        a_s = self.get_alpha_bar(s_int, "p")
        alpha_ratio_sq = self.get_alpha_pos_ts_sq(t_int=t_int, s_int=s_int)
        sigma_ratio_sq = self.get_sigma_pos_sq_ratio(s_int=s_int, t_int=t_int)
        return (a_s * (1 - alpha_ratio_sq * sigma_ratio_sq)).float()

    # ------------------------------------------------------------------
    # Transition matrices
    # ------------------------------------------------------------------

    def get_Qt(self, t_int):
        """One-step transitions, t-1 -> t. Returns [B, K, K] for X and E."""
        eye_x = torch.eye(self.X_classes, device=t_int.device).unsqueeze(0)
        eye_e = torch.eye(self.E_classes, device=t_int.device).unsqueeze(0)

        bx = self.get_beta(t_int, "x").unsqueeze(1)
        be = self.get_beta(t_int, "e").unsqueeze(1)
        return bx * self.Px + (1 - bx) * eye_x, be * self.Pe + (1 - be) * eye_e

    def get_Qt_bar(self, t_int):
        """Cumulative transitions, 0 -> t. Returns [B, K, K] for X and E."""
        eye_x = torch.eye(self.X_classes, device=t_int.device).unsqueeze(0)
        eye_e = torch.eye(self.E_classes, device=t_int.device).unsqueeze(0)

        ax = self.get_alpha_bar(t_int, "x").unsqueeze(1)
        ae = self.get_alpha_bar(t_int, "e").unsqueeze(1)
        return ax * eye_x + (1 - ax) * self.Px, ae * eye_e + (1 - ae) * self.Pe

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def apply_noise(self, pos, X, E, node_mask, t_int=None):
        """Corrupt a clean graph to timestep t.

        Args:
            pos: [B, N, 3], already on the zero-CoM subspace.
            X:   [B, N, X_classes] one-hot.
            E:   [B, N, N, E_classes] one-hot, symmetric with zero diagonal.
            t_int: optional [B, 1] long, sampled uniformly from [1, T] if omitted.
        """
        B = X.shape[0]
        if t_int is None:
            t_int = torch.randint(1, self.T + 1, size=(B, 1), device=X.device)

        Qtb_x, Qtb_e = self.get_Qt_bar(t_int)
        probX = X @ Qtb_x                                            # [B, N, dx]
        probE = E @ Qtb_e.unsqueeze(1)                               # [B, N, N, de]

        X_idx, E_idx = sample_discrete_features(probX, probE, node_mask)
        X_t = F.one_hot(X_idx, num_classes=self.X_classes).float() * node_mask.unsqueeze(-1)
        E_t = zero_diagonal(F.one_hot(E_idx, num_classes=self.E_classes).float())

        noise = torch.randn_like(pos) * node_mask.unsqueeze(-1)
        noise = remove_mean_with_mask(noise, node_mask)

        a = self.get_alpha_bar(t_int, "p").unsqueeze(-1)             # [B, 1, 1]
        s = self.get_sigma_bar(t_int, "p").unsqueeze(-1)
        pos_t = a * pos + s * noise

        return {"pos_t": pos_t, "X_t": X_t, "E_t": E_t, "t_int": t_int}

    def sample_limit_dist(self, n_samples, n_nodes, device):
        """Draw z_T: Gaussian positions, categorical X/E from the limit distribution."""
        x_limit = self.X_marginals.expand(n_samples, n_nodes, -1)
        e_limit = self.E_marginals[None, None, None, :].expand(n_samples, n_nodes, n_nodes, -1)

        U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(n_samples, n_nodes)
        U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(n_samples, n_nodes, n_nodes)

        U_X = F.one_hot(U_X, num_classes=self.X_classes).float()
        U_E = F.one_hot(U_E, num_classes=self.E_classes).float()

        # Keep only the upper triangle, then mirror it: symmetric, hollow.
        upper = torch.zeros_like(U_E)
        idx = torch.triu_indices(row=n_nodes, col=n_nodes, offset=1, device=U_E.device)
        upper[:, idx[0], idx[1], :] = 1
        U_E = U_E * upper
        U_E = U_E + torch.transpose(U_E, 1, 2)

        node_mask = torch.ones(n_samples, n_nodes, device=device)
        pos = torch.randn(n_samples, n_nodes, 3, device=device)
        pos = remove_mean_with_mask(pos, node_mask)
        return pos, U_X.to(device), U_E.to(device)

    # ------------------------------------------------------------------
    # Reverse process
    # ------------------------------------------------------------------

    def sample_zs_from_zt_and_pred(self, pos_t, X_t, E_t, pred_pos, pred_X, pred_E,
                                   t_int, s_int, node_mask):
        """Sample z_s ~ p(z_s | z_t) given the network's x0 prediction."""
        bs, n, _ = X_t.shape

        # --- positions: Gaussian posterior in the x0 parameterisation
        sigma_sq_ratio = self.get_sigma_pos_sq_ratio(s_int=s_int, t_int=t_int)
        z_t_prefactor = (self.get_alpha_pos_ts(t_int=t_int, s_int=s_int) * sigma_sq_ratio).unsqueeze(-1)
        x_prefactor = self.get_x_pos_prefactor(s_int=s_int, t_int=t_int).unsqueeze(-1)
        mu = z_t_prefactor * pos_t + x_prefactor * pred_pos

        sigma2_t_s = (self.get_sigma2_bar(t_int, "p")
                      - self.get_sigma2_bar(s_int, "p") * self.get_alpha_pos_ts_sq(t_int=t_int, s_int=s_int))
        noise_prefactor = torch.sqrt((sigma2_t_s * sigma_sq_ratio).clamp(min=0)).unsqueeze(-1)

        noise = torch.randn_like(pos_t) * node_mask.unsqueeze(-1)
        noise = remove_mean_with_mask(noise, node_mask)
        pos_s = mu + noise_prefactor * noise

        # --- categorical: exact posterior, marginalised over x0
        Qtb_x, Qtb_e = self.get_Qt_bar(t_int)
        Qsb_x, Qsb_e = self.get_Qt_bar(s_int)
        Qt_x, Qt_e = self.get_Qt(t_int)

        p_X = F.softmax(pred_X, dim=-1)
        p_E = F.softmax(pred_E, dim=-1)

        post_X = compute_batched_over0_posterior_distribution(X_t, Qt_x, Qsb_x, Qtb_x)
        post_E = compute_batched_over0_posterior_distribution(E_t, Qt_e, Qsb_e, Qtb_e)

        unnorm_X = (p_X.unsqueeze(-1) * post_X).sum(dim=2)            # [bs, n, d_t-1]
        unnorm_X[torch.sum(unnorm_X, dim=-1) == 0] = 1e-5
        prob_X = unnorm_X / torch.sum(unnorm_X, dim=-1, keepdim=True)

        p_E = p_E.reshape((bs, -1, p_E.shape[-1]))
        unnorm_E = (p_E.unsqueeze(-1) * post_E).sum(dim=-2)
        unnorm_E[torch.sum(unnorm_E, dim=-1) == 0] = 1e-5
        prob_E = unnorm_E / torch.sum(unnorm_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, self.E_classes)

        X_idx, E_idx = sample_discrete_features(prob_X, prob_E, node_mask)
        X_s = F.one_hot(X_idx, num_classes=self.X_classes).float() * node_mask.unsqueeze(-1)
        E_s = zero_diagonal(F.one_hot(E_idx, num_classes=self.E_classes).float())
        return pos_s, X_s, E_s
