import numpy as np
import pandas as pd
import pytest

from samileides.coverage import chrf3_by_vocab_coverage, coverage_report
from samileides.preprocess import SRC_COLUMN, TGT_COLUMN

torch = pytest.importorskip("torch")


def _tiny_model(vocab=40):
    from transformers import MarianConfig, MarianMTModel

    cfg = MarianConfig(
        vocab_size=vocab, d_model=32, encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=64, decoder_ffn_dim=64, max_position_embeddings=64,
        scale_embedding=True, share_encoder_decoder_embeddings=True,
        pad_token_id=3, eos_token_id=2, bos_token_id=1, decoder_start_token_id=3,
    )
    return MarianMTModel(cfg)


def test_add_target_tag_row_grows_vocab():
    from samileides.attach import add_target_tag_row

    m = _tiny_model(40)
    new_id = add_target_tag_row(m)
    assert new_id == 40
    assert m.config.vocab_size == 41
    assert m.get_input_embeddings().weight.shape[0] == 41


def test_graft_freezes_all_but_tag_row_and_adapters():
    from samileides.attach import (
        add_target_tag_row, freeze_for_graft, graft_adapters,
    )

    m = _tiny_model(40)
    new_id = add_target_tag_row(m)
    adapters = graft_adapters(m, dim=8)
    freeze_for_graft(m, new_id, adapters)

    # Snapshot frozen base weights (a decoder attention projection + all but the
    # new embedding row).
    dec_w = m.model.decoder.layers[0].self_attn.q_proj.weight.detach().clone()
    emb_before = m.get_input_embeddings().weight.detach().clone()

    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1.0)
    ids = torch.tensor([[new_id, 5, 6, 2]])
    labels = torch.tensor([[7, 8, 2]])
    out = m(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    out.loss.backward()
    opt.step()

    # Frozen decoder projection unchanged.
    assert torch.equal(dec_w, m.model.decoder.layers[0].self_attn.q_proj.weight)
    emb_after = m.get_input_embeddings().weight.detach()
    # Only the new tag row moved.
    assert not torch.equal(emb_before[new_id], emb_after[new_id])
    assert torch.equal(emb_before[:new_id], emb_after[:new_id])


def test_coverage_report_flags_unseen_ot_vocab():
    nt = ["Jesus loves the people", "grace and peace"]
    ot_ref = ["Behold the leviathan and the ephod", "grace to the people"]
    ot_hyp = ["Behold the leviathan", "grace to the people"]
    rep = coverage_report(nt, ot_ref, ot_hyp)
    # leviathan/ephod/behold never appear in the NT targets
    assert rep["type_oov_rate"] > 0
    assert rep["ot_ref_word_types"] > 0
    assert 0.0 <= rep["proper_noun_copy_acc"] <= 1.0


def test_chrf3_split_by_coverage_runs():
    nt = ["the people", "grace and peace"]
    refs = ["the people", "the leviathan"]
    hyps = ["the people", "a leviathan"]
    out = chrf3_by_vocab_coverage(refs, hyps, nt)
    assert out["n_seen_vocab"] == 1 and out["n_unseen_vocab"] == 1
    assert "chrF3_seen_vocab" in out and "chrF3_unseen_vocab" in out


def test_anchor_collator_shapes_single_slot_memory():
    from samileides.attach import AnchorCollator

    coll = AnchorCollator(pad_id=3)
    batch = [
        {"anchor": np.ones(8, dtype=np.float32), "labels": [5, 6, 2]},
        {"anchor": np.zeros(8, dtype=np.float32), "labels": [7, 2]},
    ]
    out = coll(batch)
    mem = out["encoder_outputs"].last_hidden_state
    assert mem.shape == (2, 1, 8)               # single-slot memory
    assert out["attention_mask"].shape == (2, 1)
    assert out["labels"].shape == (2, 3)
    assert out["decoder_input_ids"].shape == (2, 3)
