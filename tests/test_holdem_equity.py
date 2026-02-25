from src.holdem_equity import estimate_equity, evaluate_five, evaluate_seven


def test_evaluate_five_straight_flush_beats_pair():
    sf = evaluate_five(["As", "Ks", "Qs", "Js", "Ts"])
    pair = evaluate_five(["Ah", "Ad", "7c", "5s", "2h"])
    assert sf > pair


def test_evaluate_seven_chooses_best_5_cards():
    score = evaluate_seven(["As", "Ks", "Qs", "Js", "Ts", "2d", "3c"])
    expected = evaluate_five(["As", "Ks", "Qs", "Js", "Ts"])
    assert score == expected


def test_equity_is_1_for_unbeatable_showdown_board_state():
    # Hero has royal flush; board is complete and does not contain As/Ks.
    # Opponent cannot tie or beat with unique deck cards.
    result = estimate_equity(
        hole_cards=["As", "Ks"],
        board_cards=["Qs", "Js", "Ts", "2d", "3c"],
        num_opponents=1,
        simulations=500,
        seed=42,
    )
    assert result.equity == 1.0
    assert result.loss_prob == 0.0


def test_equity_rejects_too_many_opponents():
    try:
        estimate_equity(["As", "Ah"], [], num_opponents=8, simulations=10)
        assert False, "Expected ValueError"
    except ValueError:
        pass
