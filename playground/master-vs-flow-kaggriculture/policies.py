"""Frozen policy mapping for the Master-vs-Flow one-shot.

This file intentionally does not contain the Kaggriculture executor.  It fixes
policy intent before the executable adapter sees a result, so the one-shot
cannot be tuned after observation.
"""

MASTER = {
    "name": "master",
    "objective": "terminal_money",
    "primary_axis": "WHEAT",
    "cash_reserve": 150,
    "expansion_stop_day": 18,
    "liquidation_start_day": 21,
    "terminal_liquidation_day": 28,
    "principle": "concentrate production, carry inventory, then liquidate late",
}

FLOW = {
    "name": "flow",
    "objective": "terminal_money",
    "primary_axis": "BEST_NON_WHEAT_BY_VISIBLE_PRICE_AND_TIME_TO_TERMINAL",
    "cash_reserve": 500,
    "expansion_stop_day": 16,
    "liquidation_start_day": 17,
    "terminal_liquidation_day": 28,
    "principle": "preserve liquidity, choose one efficient alternative axis, liquidate early",
}

ONE_SHOT = {
    "episode_steps": 720,
    "seed": 20260916,
    "seat_master": 0,
    "seat_flow": 1,
    "rematch": False,
    "post_result_tuning": False,
    "winner_metric": "terminal_money",
}
