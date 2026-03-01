from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class History:
    """
    策略执行的历史数据
    """

    values: pd.DataFrame
    positions: pd.DataFrame


@dataclass
class Context:
    strategy: Callable
    date: pd.Timestamp
    value: float
    positions: pd.DataFrame
    prices: pd.DataFrame


def run(
    strategy: Callable,
    prices: pd.DataFrame,
    history: History | None,
):
    """
    Run the trading strategy.
    """
    market_start = prices["date"].min()
    if history is None:
        history = History(
            values=pd.DataFrame(
                {
                    "date": [market_start],
                    "value": [1.0],
                }
            ),
            positions=pd.DataFrame(columns=["date", "instrument", "value", "weight"]),
        )
    else:
        history = History(
            values=history.values.copy(), positions=history.positions.copy()
        )

    last_date = history.values["date"].max()
    ctx = Context(
        strategy=strategy,
        date=last_date,
        value=history.values.loc[last_date, "value"],
        positions=history.positions[history.positions["date"] == last_date],
        prices=prices[prices["date"] == last_date],
    )
    for date, date_prices in prices.groupby("date"):
        if date <= ctx.date:
            continue
        value, position = _run1d(ctx, history)

        last_date = date
        last_positions = position
        last_prices = date_prices


def _run1d(ctx: Context, history: History) -> tuple[float, pd.DataFrame]:
    return 1.0, pd.DataFrame(columns=["date", "instrument", "value", "weight"])
