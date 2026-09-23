from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

SB = 0.5
BB = 1.0
MAX_RAISES = 4

# open, 3-bet, 4-bet, 5-bet
RAISE_MULTIPLIERS = [2.5, 3.0, 2.3, 2.0]


class Action(Enum):
    FOLD = auto()
    CALL = auto()
    RAISE = auto()
    ALL_IN = auto()


@dataclass(frozen=True)
class GameState:
    history: tuple[Action, ...]
    player_to_act: int  # 0 or 1
    pot: float
    stacks: tuple[float, float]  # remaining behind, indexed by player
    committed: tuple[float, float]  # already put in this street, indexed by player
    n_raises: int
    is_terminal: bool
    is_showdown: bool
    folded_player: int | None = None


def initial_state(stack_depth: float = 100.0) -> GameState:
    """Player 0 (SB/button) posts SB and acts first; player 1 (BB) posts BB."""
    stacks = (stack_depth - SB, stack_depth - BB)
    return GameState(
        history=(),
        player_to_act=0,
        pot=SB + BB,
        stacks=stacks,
        committed=(SB, BB),
        n_raises=0,
        is_terminal=False,
        is_showdown=False,
    )


def to_call(state: GameState) -> float:
    p = state.player_to_act
    return state.committed[1 - p] - state.committed[p]


def is_facing_all_in(state: GameState) -> bool:
    return state.stacks[state.player_to_act] == 0.0


def legal_actions(state: GameState) -> list[Action]:
    if state.is_terminal:
        return []
    p = state.player_to_act
    call_amt = to_call(state)

    if is_facing_all_in(state):
        actions = [Action.CALL]
        if call_amt > 0:
            actions.insert(0, Action.FOLD)
        return actions

    actions: list[Action] = []
    if call_amt > 0:
        actions.append(Action.FOLD)
    actions.append(Action.CALL)
    if state.stacks[p] > call_amt and state.n_raises < MAX_RAISES:
        actions.append(Action.RAISE)
    if state.stacks[p] > call_amt:
        actions.append(Action.ALL_IN)
    return actions


def _raise_target_total(state: GameState) -> float:
    mult = RAISE_MULTIPLIERS[min(state.n_raises, len(RAISE_MULTIPLIERS) - 1)]
    prev_bet = max(state.committed)
    return prev_bet * mult


def apply_action(state: GameState, action: Action) -> GameState:
    p = state.player_to_act
    opp = 1 - p
    stacks = list(state.stacks)
    committed = list(state.committed)

    if action == Action.FOLD:
        return GameState(
            history=state.history + (action,),
            player_to_act=p,
            pot=state.pot,
            stacks=state.stacks,
            committed=state.committed,
            n_raises=state.n_raises,
            is_terminal=True,
            is_showdown=False,
            folded_player=p,
        )

    if action == Action.CALL:
        call_amt = min(to_call(state), stacks[p])
        stacks[p] -= call_amt
        committed[p] += call_amt
        pot = state.pot + call_amt
        # single-street (preflop-only) game: a call always closes the hand
        return GameState(
            history=state.history + (action,),
            player_to_act=opp,
            pot=pot,
            stacks=tuple(stacks),
            committed=tuple(committed),
            n_raises=state.n_raises,
            is_terminal=True,
            is_showdown=True,
        )

    if action == Action.RAISE:
        target_total = _raise_target_total(state)
        add = min(target_total - committed[p], stacks[p])
        stacks[p] -= add
        committed[p] += add
        pot = state.pot + add
        return GameState(
            history=state.history + (action,),
            player_to_act=opp,
            pot=pot,
            stacks=tuple(stacks),
            committed=tuple(committed),
            n_raises=state.n_raises + 1,
            is_terminal=False,
            is_showdown=False,
        )

    if action == Action.ALL_IN:
        add = stacks[p]
        stacks[p] = 0.0
        committed[p] += add
        pot = state.pot + add
        return GameState(
            history=state.history + (action,),
            player_to_act=opp,
            pot=pot,
            stacks=tuple(stacks),
            committed=tuple(committed),
            n_raises=state.n_raises + 1,
            is_terminal=False,  # opponent still needs to respond (call/fold)
            is_showdown=False,
        )

    raise ValueError(f"unhandled action: {action}")


def enumerate_all_terminal_histories(stack_depth: float = 100.0) -> list[GameState]:
    """dfs the whole tree"""
    terminals: list[GameState] = []
    root = initial_state(stack_depth)

    def dfs(state: GameState) -> None:
        if state.is_terminal:
            terminals.append(state)
            return
        for action in legal_actions(state):
            dfs(apply_action(state, action))

    dfs(root)
    return terminals
