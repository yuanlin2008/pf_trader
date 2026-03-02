from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class History:
    """
    策略执行的历史数据

    Attributes:
        values: 包含日期和策略总价值的 DataFrame，列名为 "date" 和 "value"
        positions: 包含日期、持仓信息和权重的 DataFrame，列名为 "date","instrument","value","weight","price"
        trades: 包含日期、交易信息和交易成本的 DataFrame，列名为 "date","instrument","weight_from","weight_to","cost"
    """

    values: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame


@dataclass(frozen=True)
class State:
    """
    策略执行的状态信息
     Attributes:
        date: 当前日期
        value: 当前策略总价值
        positions: 当前持仓信息，包含 "instrument", "value", "weight" 和 "price" 列
    """

    date: pd.Timestamp
    value: float
    positions: pd.DataFrame


@dataclass(frozen=True)
class Context:
    """
    策略执行的上下文信息

     Attributes:
        date: 当前日期
        is_rebalance_date: 是否为调仓日
        value: 当前策略总价值
        positions: 当前持仓信息，包含 "instrument", "value", "weight" 和 "price" 列
    """

    date: pd.Timestamp
    is_rebalance_date: bool
    value: float
    positions: pd.DataFrame


def run(
    strategy: Callable,
    rebalance_period: str,
    buy_cost: float,
    sell_cost: float,
    slip_cost: float,
    prices: pd.DataFrame,
    state: State | None = None,
) -> History:
    """
    执行策略

        Args:
        strategy: 交易策略函数，接受 Context 和 History 参数
        rebalance_period: 调仓周期，支持 "w"（周）、"m"（月）、"q"（季度）和 "y"（年）
        buy_cost: 买入成本率
        sell_cost: 卖出成本率
        slip_cost: 滑点成本率
        prices: 包含日期和价格数据的 DataFrame，列名为 "date", "instrument", "price"
        state: 策略执行的状态信息，默认为 None:
    Returns:
        History: 包含完整执行历史的数据对象

    """
    if state is None:
        state = State(
            date=prices["date"].min(),
            value=1.0,
            positions=pd.DataFrame(columns=["instrument", "value", "weight", "price"]),
        )

    values = []
    positions = []
    trades = []

    for date, date_prices in prices.groupby("date"):
        date = pd.Timestamp(str(date))
        if date <= state.date:
            continue
        date_prices = date_prices[["instrument", "price"]]
        v, p, t = _run1d(
            strategy,
            rebalance_period,
            buy_cost,
            sell_cost,
            slip_cost,
            date,
            date_prices,
            state,
        )
        state = State(date=date, value=v, positions=p.copy())

        values.append((date, v))

        p.insert(0, "date", date)
        positions.append(p)
        t.insert(0, "date", date)
        trades.append(t)

    return History(
        values=pd.DataFrame(values, columns=["date", "value"]),
        positions=pd.concat(positions, ignore_index=True),
        trades=pd.concat(trades, ignore_index=True),
    )


def _is_rebalance_date(rebalance_period: str, date: pd.Timestamp, state: State) -> bool:
    """
    检测是否为调仓日.
    """
    if rebalance_period == "d":
        return True
    elif rebalance_period == "w":
        return state.date.isocalendar()[:2] != date.isocalendar()[:2]
    elif rebalance_period == "m":
        return state.date.month != date.month
    elif rebalance_period == "q":
        return state.date.quarter != date.quarter
    else:
        return state.date.year != date.year


def _settlement(state: State, prices: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    if state.positions.empty:
        return state.value, state.positions.copy()
    # 原持仓总价值
    last_pos_value = state.value - state.positions["value"].sum()
    # 持仓数据与价格数据合并，计算当前持仓的价值
    positions = pd.merge(state.positions, prices, on="instrument", how="left")
    positions = positions.rename(columns={"price_x": "pre_price", "price_y": "price"})
    # 如果价格数据中有缺失值，使用前一天的价格填充
    positions["price"] = positions["price"].fillna(positions["pre_price"])
    # 计算当前持仓的价值
    positions["value"] = (
        positions["value"] * positions["price"] / positions["pre_price"]
    )
    # 删除临时列
    positions = positions.drop(columns=["pre_price"])
    # 计算持仓的总价值
    new_pos_value = positions["value"].sum()
    return state.value + (new_pos_value - last_pos_value), positions


def _trade(
    value: float,
    positions: pd.DataFrame,
    new_positions: pd.DataFrame,
    buy_cost: float,
    sell_cost: float,
    slip_cost: float,
) -> pd.DataFrame:
    trades = pd.merge(
        positions[["instrument", "weight"]],
        new_positions[["instrument", "weight"]],
        on="instrument",
        how="outer",
        suffixes=("_from", "_to"),
    ).fillna(0)
    trades = trades[trades["weight_from"] != trades["weight_to"]]
    trades["cost"] = np.where(
        trades["weight_to"] > trades["weight_from"],
        (trades["weight_to"] - trades["weight_from"]) * value * (buy_cost + slip_cost),
        (trades["weight_from"] - trades["weight_to"]) * value * (sell_cost + slip_cost),
    )
    return trades


def _run1d(
    strategy: Callable,
    rebalance_period: str,
    buy_cost: float,
    sell_cost: float,
    slip_cost: float,
    date: pd.Timestamp,
    prices: pd.DataFrame,
    state: State,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    # 结算当前总价值
    value, positions = _settlement(state, prices)

    # 策略函数返回新的持仓信息，包含 "instrument" 和 "weight" 列
    new_positions = strategy(
        Context(
            date=date,
            is_rebalance_date=_is_rebalance_date(rebalance_period, date, state),
            value=value,
            positions=positions.copy(),
        )
    )
    if new_positions is None:
        # 如果策略函数返回 None，则保持当前持仓不变
        return (
            value,
            positions,
            pd.DataFrame(columns=["instrument", "weight_from", "weight_to", "cost"]),
        )
    else:
        # 仓位有变化
        total_weight = new_positions["weight"].sum()
        if total_weight > 1:
            total_weight = 1
        # 将新的持仓信息与价格数据合并
        new_positions = pd.merge(new_positions, prices, on="instrument", how="left")
        # 如果价格数据中有缺失值，去除对应的持仓
        new_positions = new_positions.dropna()
        # 重新计算持仓权重
        new_positions["weight"] = new_positions["weight"] / total_weight
        # 计算交易成本，更新总价值
        trades = _trade(
            value=value,
            positions=positions,
            new_positions=new_positions,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            slip_cost=slip_cost,
        )
        value -= trades["cost"].sum()
        # 计算新的持仓的价值
        new_positions["value"] = value * new_positions["weight"]
        return value, new_positions, trades
