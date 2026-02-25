import pytest

torch = pytest.importorskip("torch")

from train import (
    TrainConfig,
    choose_action_from_logits,
    compute_gae,
    masked_log_prob_actions,
    ppo_policy_loss,
    run_training,
)


def test_choose_action_from_logits_respects_legal_actions():
    logits = torch.tensor([10.0, 1.0, 2.0, 3.0])
    action = choose_action_from_logits(logits, legal_actions=[1, 3])
    assert action in {1, 3}
    assert action == 3


def test_compute_gae_shapes_and_returns_relation():
    rewards = torch.tensor([1.0, 0.0, -0.5])
    values = torch.tensor([0.2, 0.1, -0.1])
    dones = torch.tensor([0.0, 0.0, 1.0])

    advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert torch.allclose(returns, advantages + values)


def test_masked_log_prob_and_ppo_loss_shapes():
    logits = torch.tensor([[2.0, 1.0, -1.0], [0.5, 0.1, 0.0]])
    actions = torch.tensor([0, 2], dtype=torch.long)
    mask = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool)
    old_lp = torch.tensor([-0.2, -0.5])
    adv = torch.tensor([1.0, -1.0])

    new_lp = masked_log_prob_actions(logits, actions, mask)
    loss = ppo_policy_loss(new_lp, old_lp, adv, clip_ratio=0.2)
    assert new_lp.shape == torch.Size([2])
    assert loss.dim() == 0


def test_run_training_smoke(tmp_path):
    cfg = TrainConfig(
        steps=4,
        batch_size=8,
        obs_dim=16,
        num_actions=6,
        checkpoint_interval=2,
        out_dir=str(tmp_path / "ckpts"),
        seed=123,
        backend="synthetic",
        ppo_epochs=1,
        mini_batch_size=4,
    )
    metrics = run_training(cfg)

    assert metrics["step"] == 4.0
    assert metrics["num_checkpoints"] >= 2.0
    assert metrics["league_size"] >= 2.0

    ckpt_files = list((tmp_path / "ckpts").glob("*.pt"))
    assert len(ckpt_files) >= 2

    metrics_log = tmp_path / "ckpts" / "metrics.jsonl"
    summary_file = tmp_path / "ckpts" / "summary.json"
    assert metrics_log.exists()
    assert summary_file.exists()

    lines = [ln for ln in metrics_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == cfg.steps


def test_run_training_with_checkpoint_eval(monkeypatch, tmp_path):
    def fake_eval(cfg, ckpt_path):
        return {
            "eval_mean_payoff": 0.1 if "000002" in str(ckpt_path) else 0.2,
            "eval_std_payoff": 0.01,
            "eval_ci95": 0.02,
            "eval_num_seeds": 2.0,
        }

    monkeypatch.setattr("train._evaluate_checkpoint_over_seeds", fake_eval)

    cfg = TrainConfig(
        steps=4,
        batch_size=8,
        obs_dim=16,
        num_actions=6,
        checkpoint_interval=2,
        out_dir=str(tmp_path / "ckpts_eval"),
        seed=123,
        backend="synthetic",
        ppo_epochs=1,
        mini_batch_size=4,
        eval_hands=10,
        eval_num_seeds=2,
        eval_seed_start=50,
    )

    metrics = run_training(cfg)
    assert metrics["best_eval_mean_payoff"] == 0.2
    assert metrics["best_eval_step"] == 4.0

    summary_file = tmp_path / "ckpts_eval" / "summary.json"
    assert summary_file.exists()


def test_run_training_early_stop_with_lcb_selection(monkeypatch, tmp_path):
    call = {"n": 0}

    def fake_eval(cfg, ckpt_path):
        call["n"] += 1
        # checkpoint 1: strong score by LCB, checkpoint 2: worse, checkpoint 3: worse
        if call["n"] == 1:
            return {"eval_mean_payoff": 1.0, "eval_std_payoff": 0.1, "eval_ci95": 0.1, "eval_num_seeds": 2.0}
        if call["n"] == 2:
            return {"eval_mean_payoff": 1.1, "eval_std_payoff": 0.3, "eval_ci95": 0.5, "eval_num_seeds": 2.0}
        return {"eval_mean_payoff": 1.05, "eval_std_payoff": 0.2, "eval_ci95": 0.4, "eval_num_seeds": 2.0}

    monkeypatch.setattr("train._evaluate_checkpoint_over_seeds", fake_eval)

    cfg = TrainConfig(
        steps=12,
        batch_size=8,
        obs_dim=16,
        num_actions=6,
        checkpoint_interval=2,
        out_dir=str(tmp_path / "ckpts_early"),
        seed=123,
        backend="synthetic",
        ppo_epochs=1,
        mini_batch_size=4,
        eval_hands=10,
        eval_num_seeds=2,
        eval_seed_start=50,
        selection_metric="lcb",
        selection_ci_penalty=1.0,
        early_stop_patience_checkpoints=2,
    )

    metrics = run_training(cfg)
    assert metrics["early_stop_triggered"] == 1.0
    assert metrics["step"] < 12.0
    assert metrics["best_eval_step"] == 2.0



def test_run_training_smoke_rlcard_backend_with_mock_collector(monkeypatch, tmp_path):
    class FakeRLCardBatchGenerator:
        def __init__(self, cfg, model):
            self.cfg = cfg

        def sample(self):
            obs = torch.randn(self.cfg.batch_size, self.cfg.obs_dim)
            action_mask = torch.ones(self.cfg.batch_size, self.cfg.num_actions, dtype=torch.bool)
            actions = torch.randint(low=0, high=self.cfg.num_actions, size=(self.cfg.batch_size,), dtype=torch.long)
            rewards = torch.zeros(self.cfg.batch_size)
            dones = torch.zeros(self.cfg.batch_size)
            return {
                "obs": obs,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "action_mask": action_mask,
            }

    monkeypatch.setattr("train.RLCardBatchGenerator", FakeRLCardBatchGenerator)

    cfg = TrainConfig(
        steps=3,
        batch_size=6,
        obs_dim=12,
        num_actions=5,
        checkpoint_interval=2,
        out_dir=str(tmp_path / "ckpts_rlcard_mock"),
        seed=7,
        backend="rlcard",
        ppo_epochs=1,
        mini_batch_size=3,
    )

    metrics = run_training(cfg)
    assert metrics["step"] == 3.0
    assert metrics["num_checkpoints"] >= 2.0
