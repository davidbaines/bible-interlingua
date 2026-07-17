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


# Multi-source experiment configs (ms4/ms8/base_no_nld/attach) get their
# assertions added as the configs land -- see todo.md task 2.
