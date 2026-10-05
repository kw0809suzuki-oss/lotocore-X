"""T+1 Time Imagination — G1 playground.

Independent play branch. Does not modify CORE/X/Random or Master vs Flow.
Purpose: turn each LOTO7 draw into a tiny State and observe T -> T+1 movement.
No prediction claim is made here.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class State:
    mean: float
    width: int


def to_state(numbers: Sequence[int]) -> State:
    if len(numbers) != 7:
        raise ValueError("LOTO7 draw must contain exactly 7 numbers")
    if len(set(numbers)) != 7:
        raise ValueError("numbers must be unique")
    return State(mean=sum(numbers) / 7.0, width=max(numbers) - min(numbers))


def movement(a: State, b: State) -> tuple[str, str]:
    vertical = "HIGH" if b.mean > a.mean else "LOW" if b.mean < a.mean else "SAME"
    shape = "OPEN" if b.width > a.width else "CLOSE" if b.width < a.width else "SAME"
    return vertical, shape


def walk(draws: Iterable[tuple[int, Sequence[int]]]) -> None:
    previous_no = None
    previous_state = None

    for draw_no, numbers in draws:
        state = to_state(numbers)
        if previous_state is None:
            print(f"{draw_no}: mean={state.mean:.2f} width={state.width}")
        else:
            vertical, shape = movement(previous_state, state)
            print(
                f"{previous_no}->{draw_no}: "
                f"mean {previous_state.mean:.2f}->{state.mean:.2f} ({vertical}), "
                f"width {previous_state.width}->{state.width} ({shape})"
            )
        previous_no = draw_no
        previous_state = state


if __name__ == "__main__":
    # Known recent draw used in the conversation. Add older/next draws here only
    # when their seven main numbers are verified from the repository/source.
    draws = [
        (694, [3, 13, 24, 26, 30, 31, 36]),
    ]
    walk(draws)
