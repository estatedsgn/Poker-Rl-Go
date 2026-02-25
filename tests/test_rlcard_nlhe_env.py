import sys
import types

from src.rlcard_nlhe_env import _fit_obs_dim, _flatten_obs


def test_flatten_obs_and_fit_obs_dim_padding_and_truncation():
    obs = [[1, 2], [3, 4]]
    flat = _flatten_obs(obs)
    assert flat == [1.0, 2.0, 3.0, 4.0]

    padded = _fit_obs_dim(flat, 6)
    assert len(padded) == 6
    assert padded[:4] == flat
    assert padded[4:] == [0.0, 0.0]

    truncated = _fit_obs_dim(flat, 2)
    assert len(truncated) == 2
    assert truncated == flat[:2]


class _FakeEnv:
    def __init__(self):
        self._step = 0

    def reset(self):
        self._step = 0
        return {"obs": [1.0, 2.0], "legal_actions": {1: None, 3: None}}, {}

    def step(self, _action):
        self._step += 1
        if self._step >= 2:
            return {"obs": [0.0], "legal_actions": {0: None}}, {}
        return {"obs": [3.0], "legal_actions": {0: None, 2: None}}, {}

    def is_over(self):
        return self._step >= 2

    def get_payoffs(self):
        return [0.75, -0.75]


def test_collector_sample_and_eval_with_mocked_rlcard(monkeypatch):
    import src.rlcard_nlhe_env as env_mod

    fake_rlcard = types.SimpleNamespace(make=lambda game_name, config: _FakeEnv())
    monkeypatch.setitem(sys.modules, "rlcard", fake_rlcard)

    collector = env_mod.RLCardNLHECollector(env_mod.RLCardConfig(obs_dim=4, num_actions=5))
    batch = collector.sample_batch(batch_size=3, policy_fn=None)

    assert len(batch["obs"]) == 3
    assert len(batch["actions"]) == 3
    assert len(batch["action_mask"]) == 3
    assert all(len(o) == 4 for o in batch["obs"])
    assert all(len(m) == 5 for m in batch["action_mask"])
    assert any(d == 1.0 for d in batch["dones"])

    avg = collector.evaluate_policy(num_hands=2, policy_fn=None)
    assert avg == 0.75


def test_collector_import_error_message(monkeypatch):
    import src.rlcard_nlhe_env as env_mod

    monkeypatch.delitem(sys.modules, "rlcard", raising=False)

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "rlcard":
            raise ModuleNotFoundError("no rlcard")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    try:
        env_mod.RLCardNLHECollector(env_mod.RLCardConfig())
        assert False, "Expected ImportError"
    except ImportError as exc:
        assert "pip install rlcard" in str(exc)
