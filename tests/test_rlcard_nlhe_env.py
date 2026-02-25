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
