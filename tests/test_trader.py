import pandas as pd
import pytest
from datetime import datetime

from pf_trader.trader import (
    History,
    Context,
    run,
    _init_ctx,
    _update_ctx,
    _init_history,
    _append_history,
    _is_rebalance_date,
    _run1d,
)


def test_init_history_with_none():
    """测试初始化空历史记录"""
    prices = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
        "instrument": ["A", "B"],
        "price": [100.0, 200.0],
    })
    history = _init_history(prices, None)

    assert len(history.values) == 1
    assert history.values["date"].iloc[0] == pd.Timestamp("2024-01-01")
    assert history.values["value"].iloc[0] == 1.0
    assert list(history.positions.columns) == ["date", "instrument", "value", "weight", "price"]
    assert len(history.positions) == 0


def test_init_history_with_existing():
    """测试使用现有历史记录初始化"""
    existing = History(
        values=pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "value": [1.0]}),
        positions=pd.DataFrame(columns=["date", "instrument", "value", "weight", "price"]),
    )
    prices = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01")],
        "instrument": ["A"],
        "price": [100.0],
    })
    history = _init_history(prices, existing)

    assert len(history.values) == 1
    assert history.values["date"].iloc[0] == pd.Timestamp("2024-01-01")
    assert not history.values is existing.values
    assert not history.positions is existing.positions


def test_append_history():
    """测试追加历史记录"""
    history = History(
        values=pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "value": [1.0]}),
        positions=pd.DataFrame(columns=["date", "instrument", "value", "weight", "price"]),
    )
    date = pd.Timestamp("2024-01-02")
    value = 1.05
    position = pd.DataFrame({
        "date": [date],
        "instrument": ["A"],
        "value": [1.05],
        "weight": [1.0],
        "price": [105.0],
    })

    new_history = _append_history(history, date, value, position)

    assert len(new_history.values) == 2
    assert new_history.values["date"].iloc[1] == date
    assert new_history.values["value"].iloc[1] == value
    assert len(new_history.positions) == 1


def test_init_ctx():
    """测试初始化上下文"""
    history = History(
        values=pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
            "value": [1.0, 1.05],
        }),
        positions=pd.DataFrame({
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
            "instrument": ["A", "B"],
            "value": [0.5, 0.55],
            "weight": [0.476, 0.524],
            "price": [100.0, 200.0],
        }),
    )

    ctx = _init_ctx(history)

    assert ctx.date == pd.Timestamp("2024-01-02")
    assert ctx.value == 1.05
    assert len(ctx.positions) == 2
    assert list(ctx.positions.columns) == ["instrument", "value", "weight", "price"]


def test_update_ctx():
    """测试更新上下文"""
    history = History(
        values=pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")],
            "value": [1.0],
        }),
        positions=pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")],
            "instrument": ["A"],
            "value": [1.0],
            "weight": [1.0],
            "price": [100.0],
        }),
    )
    ctx = _init_ctx(history)
    date = pd.Timestamp("2024-01-02")
    value = 1.05
    position = pd.DataFrame({
        "date": [date],
        "instrument": ["A"],
        "value": [1.05],
        "weight": [1.0],
        "price": [105.0],
    })

    new_ctx = _update_ctx(ctx, date, value, position)

    assert new_ctx.date == date
    assert new_ctx.value == value
    assert len(new_ctx.positions) == 1


def test_is_rebalance_date_weekly():
    """测试周度调仓检测"""
    ctx = Context(
        date=pd.Timestamp("2024-01-01"),
        value=1.0,
        positions=pd.DataFrame(),
    )

    # 同一周不调仓
    assert not _is_rebalance_date("w", pd.Timestamp("2024-01-03"), ctx)
    # 不同周调仓
    assert _is_rebalance_date("w", pd.Timestamp("2024-01-08"), ctx)


def test_is_rebalance_date_monthly():
    """测试月度调仓检测"""
    ctx = Context(
        date=pd.Timestamp("2024-01-01"),
        value=1.0,
        positions=pd.DataFrame(),
    )

    # 同一月不调仓
    assert not _is_rebalance_date("m", pd.Timestamp("2024-01-15"), ctx)
    # 不同月调仓
    assert _is_rebalance_date("m", pd.Timestamp("2024-02-01"), ctx)


def test_is_rebalance_date_quarterly():
    """测试季度调仓检测"""
    ctx = Context(
        date=pd.Timestamp("2024-01-01"),
        value=1.0,
        positions=pd.DataFrame(),
    )

    # 同一季度不调仓
    assert not _is_rebalance_date("q", pd.Timestamp("2024-02-01"), ctx)
    # 不同季度调仓
    assert _is_rebalance_date("q", pd.Timestamp("2024-04-01"), ctx)


def test_is_rebalance_date_yearly():
    """测试年度调仓检测"""
    ctx = Context(
        date=pd.Timestamp("2024-01-01"),
        value=1.0,
        positions=pd.DataFrame(),
    )

    # 同一年不调仓
    assert not _is_rebalance_date("y", pd.Timestamp("2024-06-01"), ctx)
    # 不同年调仓
    assert _is_rebalance_date("y", pd.Timestamp("2025-01-01"), ctx)


def test_run1d_no_rebalance():
    """测试不调仓的单日运行"""
    prices = pd.DataFrame({
        "instrument": ["A", "B"],
        "price": [105.0, 210.0],
    })
    history = History(
        values=pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")],
            "value": [1.0],
        }),
        positions=pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "instrument": ["A", "B"],
            "value": [0.5, 0.5],
            "weight": [0.5, 0.5],
            "price": [100.0, 200.0],
        }),
    )
    ctx = _init_ctx(history)

    def strategy(date, history):
        return pd.DataFrame()

    value, positions = _run1d(strategy, "m", pd.Timestamp("2024-01-02"), prices, ctx, history)

    # 价格涨了5%，价值也应该涨5%
    assert value == pytest.approx(1.05)
    assert len(positions) == 2


def test_run1d_with_rebalance():
    """测试调仓的单日运行"""
    prices = pd.DataFrame({
        "instrument": ["A", "B", "C"],
        "price": [105.0, 210.0, 300.0],
    })
    history = History(
        values=pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")],
            "value": [1.0],
        }),
        positions=pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")],
            "instrument": ["A"],
            "value": [1.0],
            "weight": [1.0],
            "price": [100.0],
        }),
    )
    ctx = _init_ctx(history)

    def strategy(date, history):
        # 返回新持仓，包含instrument和weight
        return pd.DataFrame({
            "instrument": ["B", "C"],
            "weight": [0.5, 0.5],
        })

    value, positions = _run1d(strategy, "m", pd.Timestamp("2024-02-01"), prices, ctx, history)

    # 价格涨了5%，价值1.05，按权重分配给B和C
    assert value == pytest.approx(1.05)
    assert len(positions) == 2
    assert positions["value"].sum() == pytest.approx(1.05)


def test_run_simple_strategy():
    """测试简单策略运行"""
    prices = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")] * 2,
        "instrument": ["A", "A", "A", "B", "B", "B"],
        "price": [100.0, 105.0, 110.0, 200.0, 210.0, 220.0],
    })

    def strategy(date, history):
        # 简单等权策略
        return pd.DataFrame({
            "instrument": ["A", "B"],
            "weight": [0.5, 0.5],
        })

    history = run(strategy, "d", prices)

    assert len(history.values) == 3
    assert history.values["value"].iloc[0] == 1.0
    # 第一天初始持仓为空，调仓时才产生持仓
    assert history.positions["date"].nunique() == 2


def test_run_with_existing_history():
    """测试基于现有历史运行"""
    prices = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")] * 2,
        "instrument": ["A", "A", "B", "B"],
        "price": [105.0, 110.0, 210.0, 220.0],
    })

    existing_history = History(
        values=pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")],
            "value": [1.0],
        }),
        positions=pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")],
            "instrument": ["A"],
            "value": [1.0],
            "weight": [1.0],
            "price": [100.0],
        }),
    )

    def strategy(date, history):
        return pd.DataFrame({
            "instrument": ["A"],
            "weight": [1.0],
        })

    history = run(strategy, "m", prices, existing_history)

    assert len(history.values) == 3
    assert history.values["date"].iloc[0] == pd.Timestamp("2024-01-01")
