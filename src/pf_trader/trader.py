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


def run(
    strategy: Callable,
    prices: pd.DataFrame,
    history: History | None,
):
    """
    Run the trading strategy.
    """
    market_start = prices["date"].min()
    market_end = prices["date"].max()
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

    for date, date_prices in prices.groupby("date"):
        pass
