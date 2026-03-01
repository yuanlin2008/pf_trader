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
    ctx: Context | None = None,
) -> History:
    """
    执行策略

        Args:
        strategy: 交易策略函数，接受 Context 和 History 参数
        rebalance_period: 调仓周期，支持 "w"（周）、"m"（月）、"q"（季度）和 "y"（年）
        prices: 包含日期和价格数据的 DataFrame，列名为 "date", "instrument", "price"
        ctx: 可选的上下文信息，用于初始化策略执行环境
    Returns:
        History: 包含完整执行历史的数据对象

    """
    if ctx is None:
        ctx = Context(
            date=prices["date"].min(),
            value=1.0,
            positions=pd.DataFrame(columns=["instrument", "value", "weight", "price"]),
        )

    values = []
    positions = []

    for date, date_prices in prices.groupby("date"):
        date = pd.Timestamp(str(date))
        if date <= ctx.date:
            continue
        date_prices = date_prices[["instrument", "price"]]
        v, p = _run1d(strategy, rebalance_period, date, date_prices, ctx)
        ctx.date = date
        ctx.value = v
        ctx.positions = p.copy()
        values.append((date, v))
        p.insert(0, "date", date)
        positions.append(p)

    return History(
        values=pd.DataFrame(values, columns=["date", "value"]),
        positions=pd.concat(positions, ignore_index=True),
    )


def _is_rebalance_date(rebalance_period: str, date: pd.Timestamp, ctx: Context) -> bool:
    """
    检测是否为调仓日.
    """
    if rebalance_period == "d":
        return True
    elif rebalance_period == "w":
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
    value = positions["value"].sum() if not positions.empty else ctx.value
    # 计算当前持仓的权重
    positions["weight"] = positions["value"] / value if value > 0 else 0
    # 删除临时列
    positions = positions.drop(columns=["pre_price"])

    # 调仓日执行策略函数，更新持仓信息
    if _is_rebalance_date(rebalance_period, date, ctx):
        # 策略函数返回新的持仓信息，包含 "instrument" 和 "weight" 列
        positions = strategy(date)
        # 将新的持仓信息与价格数据合并
        positions = pd.merge(positions, prices, on="instrument", how="left")
        # 如果价格数据中有缺失值，去除对应的持仓
        positions = positions.dropna()
        # 重新计算持仓权重
        positions["weight"] = positions["weight"] / positions["weight"].sum()
        # 计算新的持仓的价值
        positions["value"] = value * positions["weight"]

    return value, positions
