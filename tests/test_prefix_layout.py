"""Position ids and the 4-D attention mask of the aligned image prefix (spec §3.1, §3.5)."""
import torch

from src.models.mesh_transformer import OPTSinusoidalPositions
from src.models.prefix_layout import NO_POSITION, mesh_position_ids, prefix_attention_mask


def _b(rows):
    return torch.tensor(rows, dtype=torch.bool)


def test_mesh_positions_skip_pads_and_ignore_the_prefix():
    keep = _b([[1, 1, 0, 1]])
    assert mesh_position_ids(2, keep).tolist() == [[-2, -2, 0, 1, -1, 2]]
    assert mesh_position_ids(0, keep).tolist() == [[0, 1, -1, 2]]
    assert NO_POSITION == -2


def _mask(img, cond, tgt, bidirectional):
    return prefix_attention_mask(_b(img), _b(cond), _b(tgt), bidirectional)[0, 0]


def test_causal_mask_matches_the_table():
    m = _mask([[1, 1]], [[1, 1]], [[1, 1]], False)
    assert torch.equal(m, torch.tril(torch.ones(6, 6, dtype=torch.bool)))


def test_bidirectional_mask_matches_the_table():
    m = _mask([[1, 1]], [[1, 1]], [[1, 1]], True)
    assert m[:2, :4].all() and not m[:2, 4:].any()          # image: image + all LOD1
    assert m[2:4, :2].all()                                  # LOD1 sees every image token
    assert not m[2, 3]                                       # direct LOD1 stays causal
    tril = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    assert torch.equal(m[4:], tril[4:])                      # target is causal


def test_masked_image_keeps_only_its_diagonal():
    for bidirectional in (False, True):
        m = _mask([[0, 0]], [[1, 1]], [[1, 1]], bidirectional)
        cols = m[:, :2].clone()
        assert cols[0, 0] and cols[1, 1]
        cols[0, 0] = cols[1, 1] = False
        assert not cols.any()
        assert m.any(dim=1).all()                            # no all-masked row


def test_pad_keys_are_masked():
    m = _mask([[1, 1]], [[1, 0]], [[1, 1]], True)
    assert not m[:, 3].any()


def test_no_position_sentinel_is_a_zero_vector():
    pos = OPTSinusoidalPositions(8)
    out = pos(torch.ones(1, 4, dtype=torch.long), position_ids=torch.tensor([[-2, -1, 0, 3]]))[0]
    assert torch.equal(out[0], torch.zeros(8))
    assert torch.equal(out[1], out[2])
    assert out[3].abs().sum() > 0


def test_vertex_xy_bins_condition_and_target():
    from src.models.prefix_layout import vertex_xy_bins
    nb, pad, bos = 16, 18, 16
    bx, by, has = vertex_xy_bins(torch.tensor([[1, 2, 3, 4, 5, 6, pad, pad, pad]]), 0, nb)
    assert has.long().tolist() == [[0, 1, 1, 0, 1, 1, 0, 0, 0]]
    assert bx[0, [1, 2, 4, 5]].tolist() == [1, 1, 4, 4]
    assert by[0, [1, 2, 4, 5]].tolist() == [2, 2, 5, 5]
    bx, by, has = vertex_xy_bins(torch.tensor([[bos, 1, 2, 3, 4]]), 1, nb)
    assert has.long().tolist() == [[0, 0, 1, 1, 0]]
    assert (bx[0, 2].item(), by[0, 2].item()) == (1, 2)
    assert (bx[0, 3].item(), by[0, 3].item()) == (1, 2)
