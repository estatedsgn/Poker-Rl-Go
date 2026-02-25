import pytest

torch = pytest.importorskip("torch")

from train import TrainConfig, choose_action_from_logits, compute_gae, run_training


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
    )
    metrics = run_training(cfg)

    assert metrics["step"] == 4.0
    assert metrics["num_checkpoints"] >= 2.0
    assert metrics["league_size"] >= 2.0

    ckpt_files = list((tmp_path / "ckpts").glob("*.pt"))
    assert len(ckpt_files) >= 2
