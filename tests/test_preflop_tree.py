from solver.preflop_tree import (
    Action,
    apply_action,
    enumerate_all_terminal_histories,
    initial_state,
    legal_actions,
    to_call,
)

STACK_DEPTH = 100.0


def test_initial_state_pot_and_stacks():
    state = initial_state(STACK_DEPTH)
    assert state.pot == 1.5
    assert state.stacks == (99.5, 99.0)
    assert state.committed == (0.5, 1.0)
    assert state.player_to_act == 0


def test_to_call_at_root_is_half_bb():
    state = initial_state(STACK_DEPTH)
    assert to_call(state) == 0.5  # SB owes 0.5 more to match the BB


def test_tree_is_finite():
    terminals = enumerate_all_terminal_histories(STACK_DEPTH)
    assert len(terminals) > 0
    assert len(terminals) < 10_000


def test_every_terminal_is_fold_or_showdown():
    terminals = enumerate_all_terminal_histories(STACK_DEPTH)
    for t in terminals:
        assert t.is_terminal
        assert (t.folded_player is not None) != t.is_showdown


def test_chip_conservation_at_every_terminal():
    total_start = 2 * STACK_DEPTH
    terminals = enumerate_all_terminal_histories(STACK_DEPTH)
    for t in terminals:
        assert abs(t.pot + sum(t.stacks) - total_start) < 1e-9


def test_no_negative_stacks_or_commitments():
    terminals = enumerate_all_terminal_histories(STACK_DEPTH)
    for t in terminals:
        assert all(s >= -1e-9 for s in t.stacks)
        assert all(c >= 0 for c in t.committed)


def test_fold_terminal_has_no_further_actions():
    state = initial_state(STACK_DEPTH)
    folded = apply_action(state, Action.FOLD)
    assert folded.is_terminal
    assert not folded.is_showdown
    assert legal_actions(folded) == []


def test_all_in_call_reaches_showdown_with_zero_stacks():
    state = initial_state(STACK_DEPTH)
    state = apply_action(state, Action.ALL_IN)
    assert not state.is_terminal  # opponent must still respond
    state = apply_action(state, Action.CALL)
    assert state.is_terminal
    assert state.is_showdown
    assert state.stacks == (0.0, 0.0)
    assert abs(state.pot - 2 * STACK_DEPTH) < 1e-9


def test_raise_count_is_capped():
    from solver.preflop_tree import MAX_RAISES

    terminals = enumerate_all_terminal_histories(STACK_DEPTH)
    for t in terminals:
        n_raise_actions = sum(1 for a in t.history if a in (Action.RAISE, Action.ALL_IN))
        assert n_raise_actions <= MAX_RAISES + 1  # +1 allows a final all-in beyond the cap
