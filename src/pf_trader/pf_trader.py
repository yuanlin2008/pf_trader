from dataclasses import dataclass
from typing import Callable, TypeAlias

import numpy as np
import pandas as pd

C_Prices = ["instrument", "price"]
C_Daily = ["value", "rebalance"]
C_Positions = ["instrument", "value", "weight", "price"]
C_Settlements = ["instrument", "last_value", "value", "pct_change", "value_change"]
C_Trades = ["instrument", "weight_from", "weight_to", "cost"]


@dataclass
class History:
    """
    策略执行的历史数据

    Attributes:
        daily:  每日状态数据， 列名为 "date" + C_Daily
        positions:  每日持仓状态数据， 列名为 "date" + C_Positions
        settlements: 每日结算数据， 列名为 "date" + C_Settlements
        trades: 每日成交数据， 列名为 "date" + C_Trades
    """

    daily: pd.DataFrame
    positions: pd.DataFrame
    settlements: pd.DataFrame
    trades: pd.DataFrame


@dataclass(frozen=True)
class State:
    """
    策略执行的状态信息
     Attributes:
        date: 当前日期
        last_rebalance: 上次调仓日期
        value: 当前策略总价值
        positions: 当前持仓信息，列名为 C_Positions
    """

    date: pd.Timestamp
    last_rebalance: pd.Timestamp | None
    value: float
    positions: pd.DataFrame


# 策略执行结果类型
# None: 不进行调仓操作
# pd.DataFrame: 调仓操作，列名为 instrument, weight
# list[tuple[str, float]]: 调仓操作
# dict[str, float]: 调仓操作
StrategyResult: TypeAlias = (
    pd.DataFrame | list[tuple[str, float]] | dict[str, float] | None
)

# 策略函数类型
# State: 策略执行的状态信息
# StrategyResult: 策略执行的结果
Strategy: TypeAlias = Callable[[State], StrategyResult]


@dataclass(frozen=True)
class TraderSetting:
    """
    交易设置
    Attributes:
        buy_cost: 买入成本，默认为 0.0
        sell_cost: 卖出成本，默认为 0.0
        slip_cost: 滑点成本，默认为 0.0
    """

    buy_cost: float = 0.0
    sell_cost: float = 0.0
    slip_cost: float = 0.0


def run(
    strategy: Strategy,
    prices: pd.DataFrame,
    state: State | None = None,
    setting: TraderSetting = TraderSetting(),
) -> History:
    """
    执行策略

        Args:
        strategy: 交易策略函数，接受 State 参数
        prices: 包含日期和价格数据的 DataFrame，列名为 "date" + C_Prices
        state: 策略执行的状态信息，默认为 None:
        setting: 交易设置
    Returns:
        History: 包含完整执行历史的数据对象

    """
    if state is None:
        # 如果没有传入状态信息，则使用 prices 中的最小日期作为初始日期
        state = State(
            date=prices["date"].min(),
            last_rebalance=None,
            value=1.0,
            positions=pd.DataFrame(columns=C_Positions),
        )

    # 初始化历史数据对象
    daily = []
    positions = []
    settlements = []
    trades = []

    # 初始化上次调仓日期
    last_rebalance = state.last_rebalance

    prices = prices.sort_values("date")
    date_groups = list(prices.groupby("date"))

    # 遍历日期分组
    for date, date_prices in date_groups:
        date = pd.Timestamp(str(date))
        if date <= state.date:
            # 跳过已执行过的日期
            continue

        # 获得当日行情数据
        date_prices = date_prices[["instrument", "price"]]

        # 执行策略
        v, p, s, t = _run1d(
            strategy,
            setting,
            date,
            date_prices,
            state,
        )

        if t is not None:
            # 如果有成交记录，则更新上次调仓日期
            last_rebalance = date

        # 根据运行结果更新状态
        state = State(
            date=date, last_rebalance=last_rebalance, value=v, positions=p.copy()
        )

        # 记录历史数据
        daily.append((date, v, t is not None))
        p.insert(0, "date", date)
        positions.append(p)
        s.insert(0, "date", date)
        settlements.append(s)
        if t is not None:
            t.insert(0, "date", date)
            trades.append(t)

    return History(
        daily=(
            pd.DataFrame(daily, columns=["date", *C_Daily])
            if len(daily) > 0
            else pd.DataFrame(columns=["date", *C_Daily])
        ),
        positions=(
            pd.concat(positions, ignore_index=True)
            if len(positions) > 0
            else pd.DataFrame(columns=["date", *C_Positions])
        ),
        settlements=(
            pd.concat(settlements, ignore_index=True)
            if len(settlements) > 0
            else pd.DataFrame(columns=["date", *C_Settlements])
        ),
        trades=(
            pd.concat(trades, ignore_index=True)
            if len(trades) > 0
            else pd.DataFrame(columns=["date", *C_Trades])
        ),
    )


def _settlement(
    state: State, prices: pd.DataFrame
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    根据当前状态和价格数据进行结算.

    Args:
        state (State): 包含当前账户状态的对象
        prices (pd.DataFrame): 包含各标的当前价格的数据框，列名为 C_Prices

    Returns:
        tuple[float, pd.DataFrame, pd.DataFrame]: 包含三个元素的元组：
            - 结算后的总账户价值
            - 结算后的持仓数据框（包含C_Positions列）
            - 结算记录数据框 (包含C_Settlements列)
    """
    if state.positions.empty:
        # 如果持仓为空，则直接返回原账户价值和持仓副本
        return (
            state.value,
            pd.DataFrame(columns=C_Positions),
            pd.DataFrame(columns=C_Settlements),
        )
    # 持仓数据与价格数据合并，计算当前持仓的价值
    positions = pd.merge(
        state.positions, prices, on="instrument", how="left", suffixes=("_last", "")
    )
    # 如果价格数据中有缺失值，使用前一天的价格填充
    positions["price"] = positions["price"].fillna(positions["price_last"])
    # 计算当前持仓的价值
    positions.rename(columns={"value": "last_value"}, inplace=True)
    positions["value"] = (
        positions["last_value"] * positions["price"] / positions["price_last"]
    )
    # 计算持仓的总价值
    last_pos_value = positions["last_value"].sum()
    new_pos_value = positions["value"].sum()
    positions["weight"] = positions["value"] / new_pos_value

    # 计算结算记录
    settlement = positions[["instrument", "last_value", "value"]].copy()
    settlement["pct_change"] = settlement["value"] / settlement["last_value"] - 1
    settlement["value_change"] = settlement["value"] - settlement["last_value"]

    # 删除临时列
    positions = positions.drop(columns=["price_last", "last_value"])

    return state.value + (new_pos_value - last_pos_value), positions, settlement


def _trade(
    value: float,
    positions: pd.DataFrame,
    new_positions: pd.DataFrame,
    setting: TraderSetting,
) -> pd.DataFrame:
    """
    计算并返回调仓交易详情及交易成本

    Args:
        value (float): 投资组合总价值
        positions (pd.DataFrame): 当前持仓数据，包含'instrument'和'weight'列
        new_positions (pd.DataFrame): 目标持仓数据，包含'instrument'和'weight'列
        setting (TraderSetting): 交易设置

    Returns:
        pd.DataFrame: 包含交易详情的数据框，列名为C_Trades
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
        (trades["weight_to"] - trades["weight_from"])
        * value
        * (setting.buy_cost + setting.slip_cost),
        (trades["weight_from"] - trades["weight_to"])
        * value
        * (setting.sell_cost + setting.slip_cost),
    )
    return trades


def _run1d(
    strategy: Strategy,
    setting: TraderSetting,
    date: pd.Timestamp,
    prices: pd.DataFrame,
    state: State,
) -> tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    # 结算当前总价值
    """
    执行一维资产组合的再平衡操作，包括结算、策略执行和交易成本计算

    Args:
        strategy (Strategy): 策略函数，接收State对象并返回新的持仓信息
        setting (TraderSetting): 交易设置
        date (pd.Timestamp): 当前日期
        prices (pd.DataFrame): 资产价格数据
        state (State): 当前组合状态对象

    Returns:
        tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame|None]: 包含四个元素的元组：
            - 当前组合总价值
            - 新的持仓信息DataFrame
            - 结算记录DataFrame
            - 交易记录DataFrame
    """
    # 首先根据昨天的状态和最新行情进行结算.
    value, positions, settlement = _settlement(state, prices)

    # 更新计算后状态.
    state = State(
        date=date,
        last_rebalance=state.last_rebalance,
        value=value,
        positions=positions.copy(),
    )

    # 以结算后状态运行策略，获得调仓结果.
    weights = strategy(state)

    # 将各种格式统一转换为 DataFrame
    if weights is None:
        # 如果策略函数返回 None，则保持当前持仓不变
        return (value, positions, settlement, None)
    elif isinstance(weights, list):
        weights = pd.DataFrame(weights, columns=["instrument", "weight"])
    elif isinstance(weights, dict):
        weights = pd.DataFrame(list(weights.items()), columns=["instrument", "weight"])

    # 总权重.
    total_weight = weights["weight"].sum()
    if total_weight > 1:
        total_weight = 1

    # 将新的持仓信息与价格数据合并，开始构建新的positions
    new_positions = pd.merge(weights, prices, on="instrument", how="left")
    # 如果价格数据中有缺失值，去除对应的持仓
    new_positions = new_positions.dropna()
    # 重新计算持仓权重
    new_positions["weight"] = new_positions["weight"] / total_weight
    # 计算交易成本，更新总价值
    trades = _trade(
        value=value, positions=positions, new_positions=new_positions, setting=setting
    )
    value -= trades["cost"].sum()
    # 计算新的持仓的价值
    new_positions["value"] = value * new_positions["weight"]
    return value, new_positions, settlement, trades
