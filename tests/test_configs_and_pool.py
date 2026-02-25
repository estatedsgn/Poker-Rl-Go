import json
from pathlib import Path

from evaluate import load_eval_config
from evaluate_pool import evaluate_checkpoint_pool
from train import load_train_config


def test_load_train_and_eval_config_from_json(tmp_path):
    cfg = {
        "train": {"steps": 3, "batch_size": 4, "obs_dim": 8, "num_actions": 5, "backend": "synthetic"},
        "eval": {"checkpoint": None, "obs_dim": 8, "num_actions": 5, "hands": 10, "seed": 7},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    tcfg = load_train_config(str(p))
    ecfg = load_eval_config(str(p))

    assert tcfg.steps == 3
    assert tcfg.batch_size == 4
    assert tcfg.obs_dim == 8
    assert ecfg.hands == 10
    assert ecfg.seed == 7


def test_evaluate_checkpoint_pool_ranking(monkeypatch, tmp_path):
    ckpt1 = tmp_path / "a.pt"
    ckpt2 = tmp_path / "b.pt"
    ckpt1.write_text("x", encoding="utf-8")
    ckpt2.write_text("y", encoding="utf-8")

    def fake_run(cfg):
        return 1.0 if cfg.checkpoint.endswith("a.pt") else 2.0

    monkeypatch.setattr("evaluate_pool.run_evaluation", fake_run)
    ranking = evaluate_checkpoint_pool(str(tmp_path), hands=5)

    assert len(ranking) == 2
    assert ranking[0][0].endswith("b.pt")
    assert ranking[0][1] == 2.0
