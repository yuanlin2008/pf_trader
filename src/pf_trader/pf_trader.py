from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    try:
        from tqdm import tqdm as TqdmType
    except ImportError:
        TqdmType = type(None)
else:
    TqdmType = type(None)

StrategyResult: TypeAlias = (
    pd.DataFrame | list[tuple[str, float]] | dict[str, float] | None
)


@dataclass
class History:
    """
    策略执行的历史数据

    Attributes:
        daily:  每日状态数据，
                列名为 "date", "value", "rebalance"
        positions:  每日持仓状态数据，
                    列名为 "date","instrument","value","weight","price"
        settlements: 每日结算数据，
                     列名为 "date","instrument","last_value","value", 'pct_change', 'value_change'
        trades: 每日成交数据，
                列名为 "date","instrument","weight_from","weight_to","cost"
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
        positions: 当前持仓信息，包含 "instrument", "value", "weight" 和 "price" 列
    """

    date: pd.Timestamp
    last_rebalance: pd.Timestamp | None
    value: float
    positions: pd.DataFrame


@runtime_checkable
class Strategy(Protocol):
    """策略函数协议

    策略函数接收 State 对象，返回新的持仓信息。
    如果返回 None，则保持当前持仓不变。
    支持以下返回类型：
        - DataFrame: 必须包含 "instrument" 和 "weight" 列
        - list: 列表元素为 (instrument, weight) 元组
        - dict: 键为 instrument，值为 weight
    """

    def __call__(self, state: State) -> StrategyResult: ...


def run(
    strategy: Strategy,
    prices: pd.DataFrame,
    state: State | None = None,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    slip_cost: float = 0.0,
    show_progress: bool = False,
) -> History:
    """
    执行策略

        Args:
        strategy: 交易策略函数，接受 State 参数
        prices: 包含日期和价格数据的 DataFrame，列名为 "date", "instrument", "price"
        state: 策略执行的状态信息，默认为 None:
        buy_cost: 买入成本率
        sell_cost: 卖出成本率
        slip_cost: 滑点成本率
        show_progress: 是否显示进度条，需要安装 tqdm
    Returns:
        History: 包含完整执行历史的数据对象

    """
    if state is None:
        state = State(
            date=prices["date"].min(),
            last_rebalance=None,
            value=1.0,
            positions=pd.DataFrame(columns=["instrument", "value", "weight", "price"]),
        )

    daily = []
    positions = []
    settlements = []
    trades = []
    last_rebalance = state.last_rebalance

    prices = prices.sort_values("date")
    date_groups = list(prices.groupby("date"))
    if show_progress:
        try:
            from tqdm import tqdm

            date_groups = tqdm(date_groups, desc="Running strategy")
        except ImportError:
            show_progress = False

    # 遍历日期分组
    for date, date_prices in date_groups:
        date = pd.Timestamp(str(date))
        if date <= state.date:
            # 跳过已执行过的日期
            continue

        date_prices = date_prices[["instrument", "price"]]
        v, p, s, t = _run1d(
            strategy,
            buy_cost,
            sell_cost,
            slip_cost,
            date,
            date_prices,
            state,
        )
        if t is not None:
            last_rebalance = date
        state = State(
            date=date, last_rebalance=last_rebalance, value=v, positions=p.copy()
        )

        daily.append((date, v, t is not None))
        p.insert(0, "date", date)
        positions.append(p)
        s.insert(0, "date", date)
        settlements.append(s)
        if t is not None:
            t.insert(0, "date", date)
            trades.append(t)

    if len(daily) == 0:
        # 没有执行过任何交易，返回空的历史数据
        return History(
            daily=pd.DataFrame(columns=["date", "value", "rebalance"]),
            positions=pd.DataFrame(
                columns=["date", "instrument", "value", "weight", "price"]
            ),
            settlements=pd.DataFrame(
                columns=[
                    "date",
                    "instrument",
                    "last_value",
                    "value",
                    "pct_change",
                    "value_change",
                ]
            ),
            trades=pd.DataFrame(
                columns=["date", "instrument", "weight_from", "weight_to", "cost"]
            ),
        )
    else:
        # 返回完整的历史数据
        return History(
            daily=pd.DataFrame(daily, columns=["date", "value", "rebalance"]),
            positions=pd.concat(positions, ignore_index=True),
            settlements=pd.concat(settlements, ignore_index=True),
            trades=pd.concat(trades, ignore_index=True),
        )


def _settlement(
    state: State, prices: pd.DataFrame
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    计算持仓结算后的账户价值和调整后的持仓信息

    Args:
        state (State): 包含当前账户状态的对象，需包含value和positions属性
        prices (pd.DataFrame): 包含各标的当前价格的数据框，需包含instrument列和price列

    Returns:
        tuple[float, pd.DataFrame, pd.DataFrame]: 包含三个元素的元组：
            - 结算后的总账户价值
            - 调整后的持仓数据框（包含value和weight列）
            - 结算记录数据框

    Note:
        - 当positions为空时直接返回原账户价值和持仓副本
        - 处理逻辑：
            1. 计算原持仓总价值
            2. 合并持仓与价格数据
            3. 处理价格缺失值（使用前一日价格填充）
            4. 重新计算持仓价值和权重
    """
    if state.positions.empty:
        return (
            state.value,
            pd.DataFrame(columns=["instrument", "value", "weight", "price"]),
            pd.DataFrame(
                columns=[
                    "instrument",
                    "last_value",
                    "value",
                    "pct_change",
                    "value_change",
                ]
            ),
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
    buy_cost: float,
    sell_cost: float,
    slip_cost: float,
    date: pd.Timestamp,
    prices: pd.DataFrame,
    state: State,
) -> tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    # 结算当前总价值
    """
    执行一维资产组合的再平衡操作，包括结算、策略执行和交易成本计算

    Args:
        strategy (Strategy): 策略函数，接收State对象并返回新的持仓信息
        buy_cost (float): 买入交易成本率
        sell_cost (float): 卖出交易成本率
        slip_cost (float): 滑点成本率
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
    value, positions, settlement = _settlement(state, prices)

    # 策略函数返回新的持仓信息，支持 DataFrame / list / dict
    new_positions = strategy(
        State(
            date=date,
            last_rebalance=state.last_rebalance,
            value=value,
            positions=positions.copy(),
        )
    )

    # 将各种格式统一转换为 DataFrame
    if new_positions is None:
        # 如果策略函数返回 None，则保持当前持仓不变
        return (value, positions, settlement, None)
    elif isinstance(new_positions, list):
        # list[(instrument, weight)] -> DataFrame
        new_positions = pd.DataFrame(new_positions, columns=["instrument", "weight"])
    elif isinstance(new_positions, dict):
        # dict[instrument, weight] -> DataFrame
        new_positions = pd.DataFrame(
            list(new_positions.items()), columns=["instrument", "weight"]
        )
    # 此时 new_positions 必然是 DataFrame
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
    return value, new_positions, settlement, trades
