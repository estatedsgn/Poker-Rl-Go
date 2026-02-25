import random

from src.self_play_league import (
    MatchResult,
    PolicySnapshot,
    SelfPlayLeague,
    create_initial_league,
)


def _mk_snapshot(i: int, step: int = 0) -> PolicySnapshot:
    return PolicySnapshot(snapshot_id=f"p{i}", path=f"/tmp/p{i}.pt", step=step)


def test_create_initial_league_sets_first_as_active():
    league = create_initial_league(["a.pt", "b.pt"])
    assert league.active_id == "seed_000"
    assert len(league.entries) == 2


def test_elo_updates_move_in_expected_direction():
    league = SelfPlayLeague()
    league.add_snapshot(_mk_snapshot(1), elo=1500, set_active=True)
    league.add_snapshot(_mk_snapshot(2), elo=1500)

    league.update_elo_batch([MatchResult(a_id="p1", b_id="p2", score_a=1.0)], k_factor=20)

    assert league.entries["p1"].elo > 1500
    assert league.entries["p2"].elo < 1500


def test_promote_if_strong_switches_active():
    league = SelfPlayLeague()
    league.add_snapshot(_mk_snapshot(1), elo=1500, set_active=True)

    promoted = league.promote_if_strong(_mk_snapshot(3), candidate_elo=1540, margin=35)
    assert promoted is True
    assert league.active_id == "p3"


def test_pfsp_sampling_returns_existing_opponent():
    league = SelfPlayLeague()
    league.add_snapshot(_mk_snapshot(1), elo=1500, set_active=True)
    league.add_snapshot(_mk_snapshot(2), elo=1510)
    league.add_snapshot(_mk_snapshot(3), elo=1900)

    opponent = league.sample_opponent_pfsp("p1", rng=random.Random(7))
    assert opponent in {"p2", "p3"}
    assert opponent != "p1"


def test_top_k_ids_sorted_by_elo():
    league = SelfPlayLeague()
    league.add_snapshot(_mk_snapshot(1), elo=1600, set_active=True)
    league.add_snapshot(_mk_snapshot(2), elo=1500)
    league.add_snapshot(_mk_snapshot(3), elo=1700)

    assert league.top_k_ids(2) == ["p3", "p1"]
