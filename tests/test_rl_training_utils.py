import pytest

torch = pytest.importorskip("torch")

from src.rl_training_utils import backpropagation_step, total_actor_critic_loss


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 8)
        self.policy = torch.nn.Linear(8, 4)
        self.value = torch.nn.Linear(8, 1)

    def forward(self, x):
        h = torch.tanh(self.backbone(x))
        return self.policy(h), self.value(h)


def test_total_loss_and_backprop_step_runs():
    model = TinyModel()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(6, 4)
    logits, values = model(x)

    actions = torch.tensor([0, 1, 2, 3, 1, 0], dtype=torch.long)
    advantages = torch.randn(6)
    returns = torch.randn(6)
    action_mask = torch.ones(6, 4, dtype=torch.bool)

    loss, metrics = total_actor_critic_loss(
        logits=logits,
        actions=actions,
        advantages=advantages,
        values=values,
        returns=returns,
        action_mask=action_mask,
    )

    out = backpropagation_step(model, optim, loss, grad_clip_norm=1.0)

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert out["loss"] == pytest.approx(float(loss.detach().item()))
    assert out["grad_norm"] >= 0.0
