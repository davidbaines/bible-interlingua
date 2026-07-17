from samileides.config import ExperimentConfig
from samileides.data import repo_root


def test_load_smoke_config():
    cfg = ExperimentConfig.load(repo_root() / "configs" / "experiments" / "smoke.yaml")
    assert cfg.name == "smoke"
    assert cfg.data.source == "greek"
    assert cfg.model.d_model == 256
    assert cfg.tokenizer.vocab_size == 4000
    assert cfg.training.per_device_batch_size == 64
    assert cfg.inference.beam == 5
    assert cfg.probe is None


def test_unknown_yaml_keys_ignored(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "name: x\nphase: one-to-many\n"
        "data:\n  selection: a.csv\n  holdouts: b.yaml\n  future_key: 1\n"
        "model:\n  d_model: 128\n",
        encoding="utf-8",
    )
    cfg = ExperimentConfig.load(p)
    assert cfg.data.selection == "a.csv"
    assert cfg.model.d_model == 128


def test_load_ms4_config():
    cfg = ExperimentConfig.load(
        repo_root() / "configs" / "experiments" / "ms4_ie_shareable.yaml"
    )
    assert cfg.name == "ms4_ie_shareable"
    assert cfg.data.pairing == "multi-source"
    assert cfg.data.k == 4 and cfg.data.k_min == 1
    assert cfg.data.max_src_len == 384 and cfg.data.max_len == 192
    assert cfg.data.max_ratio == 0
    assert cfg.data.selection == "experiments/selection-ie-shareable.csv"
    assert cfg.data.holdouts == "configs/holdouts-interlingua.yaml"
    assert cfg.model.d_model == 1024
    assert cfg.training.lr_scheduler == "cosine"
    assert cfg.training.per_device_batch_size == 64
    assert cfg.training.gradient_accumulation == 4
    assert cfg.probe is not None
    assert cfg.attach is None


def test_load_ms8_config():
    cfg = ExperimentConfig.load(
        repo_root() / "configs" / "experiments" / "ms8_ie_shareable.yaml"
    )
    assert cfg.data.k == 8
    assert cfg.data.max_src_len == 640
    assert cfg.training.per_device_batch_size == 32
    assert cfg.training.gradient_accumulation == 8


def test_load_smoke_ms_config():
    cfg = ExperimentConfig.load(
        repo_root() / "configs" / "experiments" / "smoke_ms.yaml"
    )
    assert cfg.data.pairing == "multi-source"
    assert cfg.data.k == 3
    assert cfg.data.max_src_len == 256
    assert cfg.probe is not None


def test_attach_config_parses(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text(
        "name: attach_x\nphase: attach\n"
        "data:\n  selection: a.csv\n  holdouts: b.yaml\n"
        "attach:\n  mode: graft\n  base_run: checkpoints/base\n"
        "  translation: nld1939\n  adapter_dim: 32\n",
        encoding="utf-8",
    )
    cfg = ExperimentConfig.load(p)
    assert cfg.attach is not None
    assert cfg.attach.mode == "graft"
    assert cfg.attach.translation == "nld1939"
    assert cfg.attach.adapter_dim == 32
    assert cfg.attach.nt_dev_size == 500
