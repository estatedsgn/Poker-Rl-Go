from pathlib import Path
import json
from evaluate import EvalConfig, format_evaluation_output, load_eval_config, summarize_scores
from evaluate_pool import evaluate_checkpoint_pool
from train import load_train_config


def test_load_train_and_eval_config_from_json(tmp_path):
    cfg = {
        "train": {
            "steps": 3,
            "batch_size": 4,
            "obs_dim": 8,
            "num_actions": 5,
            "backend": "synthetic",
            "eval_hands": 12,
            "eval_seed_start": 13,
            "eval_num_seeds": 2,
            "selection_metric": "lcb",
            "selection_ci_penalty": 1.5,
            "early_stop_patience_checkpoints": 4,
        },
        "eval": {
            "checkpoint": None,
            "obs_dim": 8,
            "num_actions": 5,
            "hands": 10,
            "seed": 7,
            "seed_start": 9,
            "num_seeds": 4,
        },
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    tcfg = load_train_config(str(p))
    ecfg = load_eval_config(str(p))

    assert tcfg.steps == 3
    assert tcfg.batch_size == 4
    assert tcfg.obs_dim == 8
    assert tcfg.eval_hands == 12
    assert tcfg.eval_seed_start == 13
    assert tcfg.eval_num_seeds == 2
    assert tcfg.selection_metric == "lcb"
    assert tcfg.selection_ci_penalty == 1.5
    assert tcfg.early_stop_patience_checkpoints == 4
    assert ecfg.hands == 10
    assert ecfg.seed == 7
    assert ecfg.seed_start == 9
    assert ecfg.num_seeds == 4


def test_evaluate_multi_seed_output(monkeypatch):
    class Summary:
        mean_payoff = 0.5
        std_payoff = 0.1
        ci95_half_width = 0.05
        num_seeds = 3

    monkeypatch.setattr("evaluate.run_evaluation_over_seeds", lambda **_: Summary())

    out = format_evaluation_output(EvalConfig(num_seeds=3, seed_start=10))
    assert "mean=0.500000" in out
    assert "n=3" in out


def test_summarize_scores_ci_behavior():
    summary = summarize_scores([1.0, 2.0, 3.0])
    assert summary.mean_payoff == 2.0
    assert summary.std_payoff > 0.0
    assert summary.ci95_half_width > 0.0
    assert summary.num_seeds == 3


def test_evaluate_checkpoint_pool_ranking(monkeypatch, tmp_path):
    ckpt1 = tmp_path / "a.pt"
    ckpt2 = tmp_path / "b.pt"
    ckpt1.write_text("x", encoding="utf-8")
    ckpt2.write_text("y", encoding="utf-8")

    def fake_run_over_seeds(checkpoint, seeds, hands, obs_dim, num_actions):
        class S:
            pass

        s = S()
        s.mean_payoff = 1.0 if checkpoint.endswith("a.pt") else 2.0
        s.std_payoff = 0.1
        s.ci95_half_width = 0.2
        s.num_seeds = len(seeds)
        return s

    monkeypatch.setattr("evaluate_pool.run_evaluation_over_seeds", fake_run_over_seeds)
    ranking = evaluate_checkpoint_pool(str(tmp_path), hands=5, seed_start=7, num_seeds=2)

    assert len(ranking) == 2
    assert ranking[0][0].endswith("b.pt")
    assert ranking[0][1] == 2.0
    assert ranking[0][4] == 2


def test_evaluate_checkpoint_pool_ranking_lcb(monkeypatch, tmp_path):
    ckpt1 = tmp_path / "stable.pt"
    ckpt2 = tmp_path / "noisy.pt"
    ckpt1.write_text("x", encoding="utf-8")
    ckpt2.write_text("y", encoding="utf-8")

    def fake_run_over_seeds(checkpoint, seeds, hands, obs_dim, num_actions):
        class S:
            pass

        s = S()
        if checkpoint.endswith("stable.pt"):
            s.mean_payoff = 1.8
            s.ci95_half_width = 0.1
        else:
            s.mean_payoff = 2.0
            s.ci95_half_width = 0.5
        s.std_payoff = 0.2
        s.num_seeds = len(seeds)
        return s

    monkeypatch.setattr("evaluate_pool.run_evaluation_over_seeds", fake_run_over_seeds)

    ranking = evaluate_checkpoint_pool(
        str(tmp_path),
        hands=5,
        seed_start=7,
        num_seeds=2,
        rank_metric="lcb",
        ci_penalty=1.0,
    )

    assert ranking[0][0].endswith("stable.pt")


def test_repo_train_profiles_exist():
    debug_cfg = Path("configs/train_debug.json")
    working_cfg = Path("configs/train_working.json")
    assert debug_cfg.exists()
    assert working_cfg.exists()

    debug_payload = json.loads(debug_cfg.read_text(encoding="utf-8"))
    working_payload = json.loads(working_cfg.read_text(encoding="utf-8"))

    assert debug_payload["train"]["steps"] < working_payload["train"]["steps"]
    assert working_payload["train"]["eval_num_seeds"] >= 1
