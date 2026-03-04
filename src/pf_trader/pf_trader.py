from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class Strategy(Protocol):
    """策略函数协议

    策略函数接收 Context 对象，返回新的持仓信息。
    如果返回 None，则保持当前持仓不变。
    如果返回 DataFrame，必须包含 "instrument" 和 "weight" 列。
    """
    def __call__(self, context: Context) -> pd.DataFrame | None: ...


def run(
    strategy: Strategy,
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
    """
    计算持仓结算后的账户价值和调整后的持仓信息

    Args:
        state (State): 包含当前账户状态的对象，需包含value和positions属性
        prices (pd.DataFrame): 包含各标的当前价格的数据框，需包含instrument列和price列

    Returns:
        tuple[float, pd.DataFrame]: 包含两个元素的元组：
            - 结算后的总账户价值
            - 调整后的持仓数据框（包含value和weight列）

    Note:
        - 当positions为空时直接返回原账户价值和持仓副本
        - 处理逻辑：
            1. 计算原持仓总价值
            2. 合并持仓与价格数据
            3. 处理价格缺失值（使用前一日价格填充）
            4. 重新计算持仓价值和权重
    """
    if state.positions.empty:
        return state.value, state.positions.copy()
    # 原持仓总价值
    last_pos_value = state.positions["value"].sum()
    # 持仓数据与价格数据合并，计算当前持仓的价值
    positions = pd.merge(
        state.positions, prices, on="instrument", how="left", suffixes=("_last", "")
    )
    # 如果价格数据中有缺失值，使用前一天的价格填充
    positions["price"] = positions["price"].fillna(positions["price_last"])
    # 计算当前持仓的价值
    positions["value"] = (
        positions["value"] * positions["price"] / positions["price_last"]
    )
    # 删除临时列
    positions = positions.drop(columns=["price_last"])
    # 计算持仓的总价值
    new_pos_value = positions["value"].sum()
    positions["weight"] = positions["value"] / new_pos_value
    return state.value + (new_pos_value - last_pos_value), positions


def _trade(
    value: float,
    positions: pd.DataFrame,
    new_positions: pd.DataFrame,
    buy_cost: float,
    sell_cost: float,
    slip_cost: float,
) -> pd.DataFrame:
    """
    计算并返回调仓交易详情及交易成本

    Args:
        value (float): 投资组合总价值
        positions (pd.DataFrame): 当前持仓数据，包含'instrument'和'weight'列
        new_positions (pd.DataFrame): 目标持仓数据，包含'instrument'和'weight'列
        buy_cost (float): 买入交易成本率
        sell_cost (float): 卖出交易成本率
        slip_cost (float): 滑点成本率

    Returns:
        pd.DataFrame: 包含以下列的调仓交易详情:
            - instrument: 交易标的
            - weight_from: 原持仓权重
            - weight_to: 目标持仓权重
            - cost: 交易成本
    """
    # 计算调仓交易详情
    trades = pd.merge(
        positions[["instrument", "weight"]],
        new_positions[["instrument", "weight"]],
        on="instrument",
        how="outer",
        suffixes=("_from", "_to"),
    ).fillna(0)
    # 如果权重变化小于阈值，则视为无变化
    weight_diff = np.abs(trades["weight_to"] - trades["weight_from"])
    trades = trades[weight_diff > 1e-9]
    # 计算交易成本
    trades["cost"] = np.where(
        trades["weight_to"] > trades["weight_from"],
        (trades["weight_to"] - trades["weight_from"]) * value * (buy_cost + slip_cost),
        (trades["weight_from"] - trades["weight_to"]) * value * (sell_cost + slip_cost),
    )
    return trades


def _run1d(
    strategy: Strategy,
    rebalance_period: str,
    buy_cost: float,
    sell_cost: float,
    slip_cost: float,
    date: pd.Timestamp,
    prices: pd.DataFrame,
    state: State,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    # 结算当前总价值
    """
    执行一维资产组合的再平衡操作，包括结算、策略执行和交易成本计算

    Args:
        strategy (Callable): 策略函数，接收Context对象并返回新的持仓信息
        rebalance_period (str): 再平衡周期标识
        buy_cost (float): 买入交易成本率
        sell_cost (float): 卖出交易成本率
        slip_cost (float): 滑点成本率
        date (pd.Timestamp): 当前日期
        prices (pd.DataFrame): 资产价格数据
        state (State): 当前组合状态对象

    Returns:
        tuple[float, pd.DataFrame, pd.DataFrame]: 包含三个元素的元组：
            - 当前组合总价值
            - 新的持仓信息DataFrame
            - 交易记录DataFrame(包含instrument、weight_from、weight_to和cost列)
    """
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
