from src.nlhe_action_mapper import (
    BettingContext,
    GameConfig,
    enumerate_legal_decisions,
    map_discrete_action,
)


def test_fold_maps_to_check_if_fold_illegal():
    ctx = BettingContext(legal_actions=["check", "raise"], min_raise_to=10, max_raise_to=100)
    decision = map_discrete_action(0, ctx, pot=40)
    assert decision.action == "check"


def test_check_call_priority():
    ctx = BettingContext(legal_actions=["call"], min_raise_to=10, max_raise_to=100)
    decision = map_discrete_action(1, ctx, pot=40)
    assert decision.action == "call"


def test_raise_bucket_clamped_to_range():
    ctx = BettingContext(legal_actions=["raise"], min_raise_to=25, max_raise_to=60)
    decision = map_discrete_action(4, ctx, raise_buckets=[0.5, 1.0, 3.0], pot=20)
    assert decision.action == "raise"
    assert 25 <= decision.amount <= 60


def test_enumerate_actions_contains_core_actions_and_all_in():
    ctx = BettingContext(
        legal_actions=["fold", "call", "raise"],
        min_raise_to=10,
        max_raise_to=150,
    )
    decisions = enumerate_legal_decisions(ctx, pot=100, raise_step=0.25)

    assert any(d.action == "fold" for d in decisions)
    assert any(d.action == "call" for d in decisions)
    raise_sizes = [d.amount for d in decisions if d.action == "raise"]
    assert raise_sizes
    assert 150 in raise_sizes  # all-in included explicitly


def test_game_config_allows_up_to_8_players_only():
    GameConfig(max_players=8)

    try:
        GameConfig(max_players=9)
        assert False, "Expected ValueError for max_players=9"
    except ValueError:
        pass
