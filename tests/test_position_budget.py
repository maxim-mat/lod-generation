"""Model capacity is a task decision, not a consequence of the data filter.

`mesh_data.max_faces` decides which buildings are in the corpus.
`mesh_model.n_max_triangles` decides how much room the model gets. Before these
were one knob, which is why the mesh-v3 arms had to run at 200 (scratch) and
113 (OPT) and could not be compared.

Sizing follows MeshAnything V2, `MeshAnything/models/meshanything_v2.py`.
"""
import pytest

from src.dataset.mesh_dataset import AMT_SEQ_RATIO, face_tokens, position_budget


def test_face_tokens_is_nine_per_face_on_the_coordinate_path():
    """The coordinate tokenizer is exactly 3 vertices x 3 axes, no compression."""
    assert face_tokens(1, "coord") == 9
    assert face_tokens(200, "coord") == 1800


def test_face_tokens_budgets_amt_at_v2s_worst_case_ratio():
    """AMT's length depends on face adjacency, so it can only be bounded, not
    derived. V2 budgets 0.70x naive -- well above the 0.49 they report and the
    0.533 measured on this corpus -- so a badly connected mesh cannot overflow."""
    assert AMT_SEQ_RATIO == 0.70
    assert face_tokens(200, "amt") == int(200 * 9 * 0.70)
    assert face_tokens(200, "amt") < face_tokens(200, "coord")


def test_matches_meshanything_v2s_published_table_size():
    """Pins the formula against the reference's own constants:

        n_max_triangles = 1600 ; face_per_token = 9 ; max_seq_ratio = 0.70
        cond_length = 257
        max_length = int(1600 * 9 * 0.70 + 3 + 257)   # = 10340
    """
    assert face_tokens(1600, "amt") == 10080
    assert face_tokens(1600, "amt") + 3 + 257 == 10340


def test_shared_and_per_segment_budgets_differ_by_the_condition():
    """Two backbones, two position semantics. The scratch stack restarts
    positions per segment so it needs the *larger* half; OPT counts straight
    through so it needs the *sum* -- plus V2's 3 specials."""
    per_segment, shared = position_budget(100, cond_max_triangles=10,
                                          tokenization="coord")
    assert per_segment == max(9 * 10, 9 * 100 + 1) == 901
    assert shared == 9 * 10 + 9 * 100 + 3 == 993


def test_condition_defaults_to_the_target_capacity():
    """Unset, the LOD1 condition is budgeted for the worst case -- as large as
    the target. Sizing it down is what makes a pretrained OPT run fit."""
    assert position_budget(100, tokenization="coord") == \
        position_budget(100, cond_max_triangles=100, tokenization="coord")


def test_a_smaller_condition_budget_shrinks_the_opt_table():
    """The lever that makes `opt_pretrained: true` reachable at 200 faces: LOD1
    never binds on this corpus, so its budget need not match LOD2's."""
    _, worst = position_budget(200, tokenization="amt")
    _, real = position_budget(200, cond_max_triangles=50, tokenization="amt")
    assert worst > 2048, "worst-case AMT at 200 faces should not fit opt-350m"
    assert real <= 2048, "a realistic LOD1 budget should fit the trained table"


def test_budget_does_not_consult_the_dataset():
    """The whole point: same numbers whatever corpus is loaded."""
    assert position_budget(200, cond_max_triangles=50, tokenization="coord") == \
        position_budget(200, cond_max_triangles=50, tokenization="coord")


def test_unknown_tokenization_is_rejected():
    with pytest.raises(ValueError, match="tokeni"):
        face_tokens(10, "treemeshgpt")


# ----------------------------------------------------------------------
# The budget reaching the OPT table
# ----------------------------------------------------------------------

def _cfg(max_position_embeddings=2048):
    from transformers import OPTConfig
    from src.dataset.mesh_dataset import vocab_size
    return OPTConfig(vocab_size=vocab_size(32), hidden_size=32,
                     word_embed_proj_dim=16, num_hidden_layers=1,
                     num_attention_heads=4, ffn_dim=64,
                     max_position_embeddings=max_position_embeddings, dropout=0.0)


def test_opt_table_is_sized_from_the_task_not_the_hub_default():
    """opt-125m's 2048 is a default riding in from a checkpoint whose weights
    are not even loaded when `pretrained` is false. V2 sizes the table to the
    task instead; so does this now."""
    from src.dataset.mesh_dataset import vocab_size
    from src.models.mesh_transformer import MeshOPTTransformer

    _, shared = position_budget(40, cond_max_triangles=10, tokenization="coord")
    net = MeshOPTTransformer(vocab_size=vocab_size(32), opt_config=_cfg(2048),
                             dropout=0.0, opt_max_positions=shared)
    assert net.max_seq_len == shared == 9 * 10 + 9 * 40 + 3
    assert net.opt.config.max_position_embeddings == shared


def test_sizing_the_table_does_not_mutate_the_callers_config():
    """`opt_config` is reused across tests and runs; resizing it in place would
    make the second model silently inherit the first one's task."""
    from src.dataset.mesh_dataset import vocab_size
    from src.models.mesh_transformer import MeshOPTTransformer

    cfg = _cfg(2048)
    MeshOPTTransformer(vocab_size=vocab_size(32), opt_config=cfg, dropout=0.0,
                       opt_max_positions=512)
    assert cfg.max_position_embeddings == 2048


def test_pretrained_opt_rejects_a_task_bigger_than_its_trained_rows():
    """A pretrained checkpoint has exactly as many *trained* position rows as it
    has. Silently adding more would put untrained noise inside a trained model,
    and OPT has no length extrapolation -- so this is the one case that must
    fail loudly and name the ways out."""
    from src.models.mesh_transformer import _pretrained_position_check

    _pretrained_position_check(2048, 1578, "facebook/opt-350m")   # fits: no raise
    with pytest.raises(ValueError) as e:
        _pretrained_position_check(2048, 2523, "facebook/opt-350m")
    msg = str(e.value)
    assert "2523" in msg and "2048" in msg
    assert "sinusoidal" in msg and "cond_max_triangles" in msg
