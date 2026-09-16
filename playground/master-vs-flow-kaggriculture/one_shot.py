"""Master vs Flow: isolated one-shot executable model.

This deliberately does NOT import LOTOCORE-X.  It is a compact executable
translation of the frozen text-battle policies, used only as a staging surface
before/alongside a full Kaggriculture executor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import random

SEED = 20260916
DAYS = 30

@dataclass
class Player:
    name: str
    cash: float = 1000.0
    capacity: float = 1.0
    inventory: float = 0.0
    alt_capacity: float = 0.0
    history: list[dict] = field(default_factory=list)

    def value(self, wheat_price: float, alt_price: float) -> float:
        return self.cash + self.inventory * wheat_price + self.alt_capacity * alt_price


def sell(p: Player, qty: float, wheat_price: float) -> tuple[float, float]:
    qty = max(0.0, min(qty, p.inventory))
    p.inventory -= qty
    p.cash += qty * wheat_price
    # Shared-market impact: selling depresses the next price slightly.
    wheat_price = max(5.0, wheat_price - 0.06 * qty)
    return qty, wheat_price


def run() -> dict:
    rng = random.Random(SEED)
    master = Player("MASTER")
    flow = Player("FLOW")
    wheat_price = 28.0
    alt_price = 34.0

    for day in range(1, DAYS + 1):
        # Exogenous market motion is deterministic under the frozen seed.
        wheat_price = max(5.0, wheat_price + rng.uniform(-1.6, 1.6))
        alt_price = max(8.0, alt_price + rng.uniform(-1.0, 1.0))

        # Production from previously built capacity.
        master.inventory += 2.0 * master.capacity
        # Flow's alternative axis pays directly into cash: liquidity-first.
        flow.cash += 1.45 * flow.alt_capacity * alt_price
        flow.inventory += 0.45 * flow.capacity

        master_action = "HOLD"
        flow_action = "HOLD"

        # MASTER: WHEAT concentration, late cutoff, staged liquidation.
        if day <= 17 and master.cash >= 150 + 70:
            master.cash -= 70
            master.capacity += 0.55
            master_action = "INVEST_WHEAT"
        elif day >= 21:
            schedule = {21:10, 22:15, 23:15, 24:20, 25:25, 26:25, 27:30}
            qty = schedule.get(day, master.inventory if day >= 28 else 0)
            sold, wheat_price = sell(master, qty, wheat_price)
            master_action = f"SELL_WHEAT_{sold:.1f}"
        elif day in (18, 19, 20):
            master_action = "STOP_EXPAND/HARVEST"

        # FLOW: preserve cash, choose non-WHEAT axis, stop earlier, liquidate early.
        if day <= 15:
            # Only invest above a larger liquidity reserve and only when the
            # visible alternative price is competitive with WHEAT.
            if flow.cash >= 500 + 85 and alt_price >= wheat_price * 0.90:
                flow.cash -= 85
                flow.alt_capacity += 0.48
                flow_action = "INVEST_ALT"
            elif flow.cash >= 500 + 55 and alt_price < wheat_price * 0.90:
                flow.cash -= 55
                flow.capacity += 0.30
                flow_action = "SMALL_WHEAT_FALLBACK"
        elif day >= 17:
            sold, wheat_price = sell(flow, flow.inventory, wheat_price)
            flow_action = f"LIQUIDATE_{sold:.1f}"
        else:
            flow_action = "STOP_INVEST"

        for p, action in ((master, master_action), (flow, flow_action)):
            p.history.append({
                "day": day,
                "action": action,
                "cash": round(p.cash, 2),
                "wheat_inventory": round(p.inventory, 2),
                "wheat_capacity": round(p.capacity, 2),
                "alt_capacity": round(p.alt_capacity, 2),
                "wheat_price": round(wheat_price, 2),
                "alt_price": round(alt_price, 2),
            })

    # Day30 closure: all remaining wheat becomes terminal cash.
    _, wheat_price = sell(master, master.inventory, wheat_price)
    _, wheat_price = sell(flow, flow.inventory, wheat_price)

    result = {
        "model": "isolated-policy-staging-simulator",
        "seed": SEED,
        "days": DAYS,
        "master_terminal_money": round(master.cash, 2),
        "flow_terminal_money": round(flow.cash, 2),
        "margin_master_minus_flow": round(master.cash - flow.cash, 2),
        "winner": "MASTER" if master.cash > flow.cash else "FLOW" if flow.cash > master.cash else "TIE",
        "warning": "This is an isolated executable policy model, not the authoritative Kaggriculture engine.",
        "master_history": master.history,
        "flow_history": flow.history,
    }
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
