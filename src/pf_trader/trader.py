from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class History:
    """
    策略执行的历史数据

    Attributes:
        values: 包含日期和策略总价值的 DataFrame，列名为 "date" 和 "value"
        positions: 包含日期、持仓信息和权重的 DataFrame，列名为 "date","instrument","value","weight","price"
    """

    values: pd.DataFrame
    positions: pd.DataFrame


@dataclass
class Context:
    """
    策略执行的上下文信息
     Attributes:
        date: 当前日期
        value: 当前策略总价值
        positions: 当前持仓信息，包含 "instrument", "value", "weight" 和 "price" 列
    """

    date: pd.Timestamp
    value: float
    positions: pd.DataFrame


def run(
    strategy: Callable,
    rebalance_period: str,
    prices: pd.DataFrame,
    history: History | None = None,
) -> History:
    """
    执行策略

        Args:
        strategy: 交易策略函数，接受 Context 和 History 参数
        rebalance_period: 调仓周期，支持 "w"（周）、"m"（月）、"q"（季度）和 "y"（年）
        prices: 包含日期和价格数据的 DataFrame，列名为 "date", "instrument", "price"
        history: 可选的历史数据，包含之前执行的结果
    Returns:
        History: 包含完整执行历史的数据对象

    """
    history = _init_history(prices, history)
    ctx = _init_ctx(history)

    for date, date_prices in prices.groupby("date"):
        date = pd.Timestamp(str(date))
        if date <= ctx.date:
            continue
        date_prices = date_prices[["instrument", "price"]]
        value, position = _run1d(
            strategy, rebalance_period, date, date_prices, ctx, history
        )
        history = _append_history(history, date, value, position)
        ctx = _update_ctx(ctx, date, value, position)
    return history


def _init_ctx(history: History) -> Context:
    last_date = history.values["date"].max()
    return Context(
        date=last_date,
        value=history.values[history.values["date"] == last_date]["value"].iloc[0],
        positions=history.positions[history.positions["date"] == last_date][
            "instrument", "value", "weight", "price"
        ].copy(),
    )


def _update_ctx(
    ctx: Context,
    date: pd.Timestamp,
    value: float,
    position: pd.DataFrame,
) -> Context:
    ctx.date = date
    ctx.value = value
    ctx.positions = position
    return ctx


def _init_history(prices: pd.DataFrame, history: History | None) -> History:
    if history is None:
        return History(
            values=pd.DataFrame(
                {
                    "date": [prices["date"].min()],
                    "value": [1.0],
                }
            ),
            positions=pd.DataFrame(
                columns=["date", "instrument", "value", "weight", "price"]
            ),
        )
    else:
        return History(values=history.values.copy(), positions=history.positions.copy())


def _append_history(
    history: History, date: pd.Timestamp, value: float, position: pd.DataFrame
) -> History:
    history.values = pd.concat(
        [
            history.values,
            pd.DataFrame({"date": [date], "value": [value]}),
        ],
        ignore_index=True,
    )
    position["date"] = date
    history.positions = pd.concat(
        [
            history.positions,
            position,
        ],
        ignore_index=True,
    )
    return history


def _is_rebalance_date(rebalance_period: str, date: pd.Timestamp, ctx: Context) -> bool:
    """
    检测是否为调仓日.
    """
    if rebalance_period == "w":
        return ctx.date.isocalendar()[:2] != date.isocalendar()[:2]
    elif rebalance_period == "m":
        return ctx.date.month != date.month
    elif rebalance_period == "q":
        return ctx.date.quarter != date.quarter
    else:
        return ctx.date.year != date.year


def _run1d(
    strategy: Callable,
    rebalance_period: str,
    date: pd.Timestamp,
    prices: pd.DataFrame,
    ctx: Context,
    history: History,
) -> tuple[float, pd.DataFrame]:
    # 持仓数据与价格数据合并，计算当前持仓的价值
    positions = pd.merge(ctx.positions, prices, on="instrument", how="left")
    positions = positions.rename(columns={"price_x": "pre_price", "price_y": "price"})
    # 如果价格数据中有缺失值，使用前一天的价格填充
    positions["price"] = positions["price"].fillna(positions["pre_price"])
    # 计算当前持仓的价值
    positions["value"] = (
        positions["value"] * positions["price"] / positions["pre_price"]
    )
    # 计算持仓的总价值
    value = positions["value"].sum()
    # 计算当前持仓的权重
    positions["weight"] = positions["value"] / value if value > 0 else 0
    # 删除临时列
    positions = positions.drop(columns=["pre_price"])

    # 调仓日执行策略函数，更新持仓信息
    if _is_rebalance_date(rebalance_period, date, ctx):
        # 策略函数返回新的持仓信息，包含 "instrument" 和 "weight" 列
        positions = strategy(date, history)
        # 将新的持仓信息与价格数据合并
        positions = pd.merge(positions, prices, on="instrument", how="left")
        # 如果价格数据中有缺失值，去除对应的持仓
        positions = positions.dropna()
        # 重新计算持仓权重
        positions["weight"] = positions["weight"] / positions["weight"].sum()
        # 计算新的持仓的价值
        positions["value"] = value * positions["weight"]

    return value, positions
