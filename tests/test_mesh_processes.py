"""Diffusion process contracts.

The properties worth a test are the ones a shape check will not catch: the
noising marginal has the variance the schedule promises, x0- and
epsilon-parameterisations are algebraically the same object, and a reverse
trajectory driven by a perfect oracle lands back on the data.
"""
import pytest
import torch

from src.models.mesh_processes import GaussianProcess


def test_corrupt_has_the_scheduled_marginal():
    p = GaussianProcess(noise_steps=1000)
    x0 = torch.zeros(4096, 10, 8)
    t = torch.full((4096,), 0.5)
    x_t, eps = p.corrupt(x0, t)
    idx = p.to_index(t)[0]
    expected = (1 - p.alpha_hat[idx]).sqrt()
    assert x_t.std().item() == pytest.approx(expected.item(), rel=0.05)


def test_corrupt_at_t_zero_is_nearly_the_data():
    p = GaussianProcess(noise_steps=1000)
    x0 = torch.randn(64, 10, 8)
    x_t, _ = p.corrupt(x0, torch.zeros(64))
    assert (x_t - x0).abs().mean().item() < 0.05


def test_target_for_noise_and_original_agree():
    p_eps = GaussianProcess(target="noise")
    p_x0 = GaussianProcess(target="original")
    x0 = torch.randn(16, 10, 8)
    t = torch.rand(16)
    torch.manual_seed(1)
    x_t, aux = p_eps.corrupt(x0, t)
    eps = p_eps.target_for(x0, x_t, t, aux)
    x0_hat = p_x0.target_for(x0, x_t, t, aux)
    idx = p_eps.to_index(t)
    a = p_eps.alpha_hat[idx][:, None, None]
    # x_t = sqrt(a) x0 + sqrt(1-a) eps must hold for both readings.
    assert torch.allclose(a.sqrt() * x0_hat + (1 - a).sqrt() * eps, x_t, atol=1e-5)


def test_oracle_reverse_recovers_the_data():
    torch.manual_seed(0)
    p = GaussianProcess(noise_steps=1000, target="original")
    x0 = torch.randn(8, 10, 8) * 0.3

    def oracle(x_t, t):
        return x0                      # a perfect x0 predictor

    out = p.sample(oracle, x0.shape, x0.device, n_steps=50)
    assert (out - x0).abs().mean().item() < 0.1


def test_sample_is_seed_reproducible():
    p = GaussianProcess(noise_steps=100, target="noise")

    def oracle(x_t, t):
        return torch.zeros_like(x_t)

    torch.manual_seed(7)
    a = p.sample(oracle, (2, 10, 8), torch.device("cpu"), n_steps=10)
    torch.manual_seed(7)
    b = p.sample(oracle, (2, 10, 8), torch.device("cpu"), n_steps=10)
    assert torch.allclose(a, b)


def test_ddim_step_count_is_honoured():
    p = GaussianProcess(noise_steps=1000)
    seen = []

    def oracle(x_t, t):
        seen.append(float(t[0]))
        return torch.zeros_like(x_t)

    p.sample(oracle, (1, 10, 8), torch.device("cpu"), n_steps=25)
    assert len(seen) == 25
    assert seen == sorted(seen, reverse=True)      # time runs backwards


def test_sample_callback_sees_every_step():
    p = GaussianProcess(noise_steps=100)
    steps = []
    p.sample(lambda x, t: torch.zeros_like(x), (1, 10, 8),
             torch.device("cpu"), n_steps=10,
             callback=lambda t, x: steps.append(t))
    assert len(steps) == 10


from src.models.mesh_processes import FlowMatchingProcess


def test_flow_corrupt_interpolates_linearly():
    p = FlowMatchingProcess()
    x0 = torch.ones(4, 10, 8)
    torch.manual_seed(0)
    x_t, noise = p.corrupt(x0, torch.full((4,), 0.25))
    # Convention here: t = 1 is data, t = 0 is noise, matching the reverse loop
    # in BaseProcess.sample which runs t from 1 down to 0.
    assert torch.allclose(x_t, 0.25 * x0 + 0.75 * noise, atol=1e-6)


def test_flow_target_is_the_velocity():
    p = FlowMatchingProcess()
    x0 = torch.randn(4, 10, 8)
    torch.manual_seed(0)
    x_t, noise = p.corrupt(x0, torch.rand(4))
    v = p.target_for(x0, x_t, torch.rand(4), noise)
    assert torch.allclose(v, x0 - noise, atol=1e-6)


def test_flow_oracle_recovers_the_data():
    torch.manual_seed(0)
    p = FlowMatchingProcess()
    x0 = torch.randn(8, 10, 8) * 0.3
    noise_holder = {}

    def oracle(x_t, t):
        # A perfect velocity field for the straight path from the prior we drew.
        return x0 - noise_holder["z"]

    def prior_spy(shape, device):
        noise_holder["z"] = torch.randn(shape, device=device)
        return noise_holder["z"]

    p.prior = prior_spy
    out = p.sample(oracle, x0.shape, x0.device, n_steps=20)
    assert (out - x0).abs().mean().item() < 1e-4


def test_flow_step_count_is_honoured():
    p = FlowMatchingProcess()
    seen = []
    p.sample(lambda x, t: (seen.append(float(t[0])), torch.zeros_like(x))[1],
             (1, 10, 8), torch.device("cpu"), n_steps=12)
    assert len(seen) == 12


def test_flow_to_x0_is_consistent_with_the_path():
    p = FlowMatchingProcess()
    x0 = torch.randn(4, 10, 8)
    torch.manual_seed(0)
    t = torch.full((4,), 0.4)
    x_t, noise = p.corrupt(x0, t)
    v = x0 - noise
    assert torch.allclose(p.to_x0(x_t, 0.4, v), x0, atol=1e-5)


from src.models.mesh_processes import DiscreteProcess


def _state(coord_bins, presence=1):
    """Coordinate bins [B,9,F] -> the [B,10,F] state, occupancy appended."""
    b, _, f = coord_bins.shape
    return torch.cat([coord_bins,
                      torch.full((b, 1, f), presence, dtype=torch.long)], dim=1)


def test_discrete_corrupt_leaves_most_bins_alone_at_low_t():
    p = DiscreteProcess(noise_steps=1000, num_bins=128)
    bins = _state(torch.randint(0, 128, (256, 9, 8)))
    noisy, _ = p.corrupt(bins, torch.full((256,), 0.02))
    assert (noisy[:, :9] == bins[:, :9]).float().mean().item() > 0.9


@pytest.mark.parametrize("transition", ["uniform", "gaussian"])
def test_discrete_terminal_marginal_is_the_sampler_prior(transition):
    """Both transitions must end uniform, or the reverse loop starts from a
    distribution the forward process never produces. For `gaussian` this is the
    (1 - alpha_bar)^2 mixture weight doing its job, and it is the single
    easiest thing to get wrong in that transition."""
    p = DiscreteProcess(noise_steps=1000, num_bins=32, transition=transition)
    bins = _state(torch.zeros(8192, 9, 4, dtype=torch.long))
    noisy, _ = p.corrupt(bins, torch.ones(8192))
    counts = torch.bincount(noisy[:, :9].reshape(-1), minlength=32).float()
    assert counts.std().item() / counts.mean().item() < 0.15


def test_gaussian_transition_moves_bins_locally_mid_trajectory():
    """The whole reason the gaussian transition exists: at moderate noise a
    coordinate should land NEAR where it was, not anywhere on the grid."""
    torch.manual_seed(0)
    uni = DiscreteProcess(noise_steps=1000, num_bins=128, transition="uniform")
    gau = DiscreteProcess(noise_steps=1000, num_bins=128, transition="gaussian",
                          sigma_max=16.0)
    bins = _state(torch.full((4096, 9, 4), 64, dtype=torch.long))
    t = torch.full((4096,), 0.4)
    d_uni = (uni.corrupt(bins, t)[0][:, :9] - 64).abs().float().mean().item()
    d_gau = (gau.corrupt(bins, t)[0][:, :9] - 64).abs().float().mean().item()
    assert d_gau < 0.5 * d_uni


def test_discrete_corrupt_stays_in_range():
    p = DiscreteProcess(noise_steps=100, num_bins=128)
    noisy, _ = p.corrupt(_state(torch.randint(0, 128, (32, 9, 8))), torch.rand(32))
    assert noisy[:, :9].min() >= 0 and noisy[:, :9].max() < 128
    assert set(noisy[:, 9].unique().tolist()) <= {0, 1}


def test_discrete_prior_is_uniform():
    p = DiscreteProcess(noise_steps=100, num_bins=128)
    x = p.prior((4096, 10, 8), torch.device("cpu"))
    counts = torch.bincount(x[:, :9].reshape(-1), minlength=128).float()
    assert counts.std().item() / counts.mean().item() < 0.1


def test_discrete_oracle_reverse_recovers_the_bins():
    torch.manual_seed(0)
    p = DiscreteProcess(noise_steps=1000, num_bins=32)
    bins = torch.randint(0, 32, (4, 9, 8))

    def oracle(x_t, t):
        logits = torch.full((4, 9, 32, 8), -10.0)
        logits.scatter_(2, bins.unsqueeze(2), 10.0)
        return logits, torch.full((4, 8), 10.0)

    out = p.sample(oracle, (4, 10, 8), torch.device("cpu"), n_steps=50)
    assert (out[:, :9] == bins).float().mean().item() > 0.95
    assert (out[:, 9] == 1).float().mean().item() > 0.95   # occupancy chain too


def test_discrete_sample_returns_one_state_tensor():
    """Occupancy is a channel of the state, not a value smuggled alongside it,
    so `DiscreteProcess` needs no `sample` override at all."""
    p = DiscreteProcess(noise_steps=100, num_bins=32)
    out = p.sample(lambda x, t: (torch.zeros(2, 9, 32, 8), torch.zeros(2, 8)),
                   (2, 10, 8), torch.device("cpu"), n_steps=5)
    assert out.shape == (2, 10, 8) and out.dtype == torch.long
    assert set(out[:, 9].unique().tolist()) <= {0, 1}


def test_discrete_presence_is_corrupted_toward_a_coin_flip():
    """Occupancy has its own two-state chain: nearly intact at low t, and a
    fair coin at t = 1, which is what makes `prior` its terminal marginal."""
    p = DiscreteProcess(noise_steps=1000, num_bins=32)
    st = _state(torch.zeros(8192, 9, 4, dtype=torch.long), presence=1)
    lo, _ = p.corrupt(st, torch.full((8192,), 0.02))
    hi, _ = p.corrupt(st, torch.ones(8192))
    assert lo[:, 9].float().mean().item() > 0.9
    assert abs(hi[:, 9].float().mean().item() - 0.5) < 0.05


def test_discrete_presence_transition_is_uniform_even_under_gaussian():
    """The discretized-Gaussian kernel exploits ordinality, and {absent,
    present} has none -- there is no sense in which absent is *near* present."""
    torch.manual_seed(0)
    for transition in ("uniform", "gaussian"):
        p = DiscreteProcess(noise_steps=1000, num_bins=32, transition=transition)
        st = _state(torch.zeros(4096, 9, 4, dtype=torch.long), presence=1)
        out, _ = p.corrupt(st, torch.full((4096,), 0.5))
        assert set(out[:, 9].unique().tolist()) <= {0, 1}
