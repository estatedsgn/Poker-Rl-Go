from src.competitive_env_config import (
    ActionAbstraction,
    BlindsConfig,
    CompetitiveEnvConfig,
    RakeConfig,
    build_raise_ladder,
    estimate_rake,
)


def test_raise_ladder_contains_all_in_and_respects_bounds():
    abstraction = ActionAbstraction(pot_fractions=(0.01, 0.5, 1.0), include_all_in=True)
    ladder = build_raise_ladder(pot=200, min_raise_to=50, max_raise_to=500, abstraction=abstraction)
    assert ladder[0] >= 50
    assert ladder[-1] == 500


def test_competitive_env_players_limit():
    CompetitiveEnvConfig(players=8)
    try:
        CompetitiveEnvConfig(players=9)
        assert False, "Expected ValueError for players=9"
    except ValueError:
        pass


def test_rake_estimation_with_cap():
    rake = RakeConfig(enabled=True, percent=0.05, cap_bb=1.0)
    # pot*5% = 200 chips, cap = 1bb = 100 chips => capped at 100
    assert estimate_rake(pot=4000, big_blind=100, rake=rake) == 100


def test_blinds_validation():
    try:
        BlindsConfig(small_blind=100, big_blind=100)
        assert False, "Expected ValueError when SB >= BB"
    except ValueError:
        pass
