import numpy as np
import pandas as pd
import pytest

from pf_trader import pf_trader as trader


class TestDataClasses:
    """测试数据类"""

    def test_history_creation(self):
        """测试History数据类的创建"""
        values = pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "value": [1.0]})
        positions = pd.DataFrame(
            {"date": [], "instrument": [], "value": [], "weight": [], "price": []}
        )
        trades = pd.DataFrame(
            {
                "date": [],
                "instrument": [],
                "weight_from": [],
                "weight_to": [],
                "cost": [],
            }
        )

        history = trader.History(values=values, positions=positions, trades=trades)

        assert history.values.equals(values)
        assert history.positions.equals(positions)
        assert history.trades.equals(trades)

    def test_state_creation(self):
        """测试State数据类的创建"""
        positions = pd.DataFrame(
            {"instrument": ["A"], "value": [0.5], "weight": [1.0], "price": [100.0]}
        )
        state = trader.State(
            date=pd.Timestamp("2025-01-01"), value=1.0, positions=positions
        )

        assert state.date == pd.Timestamp("2025-01-01")
        assert state.value == 1.0
        assert len(state.positions) == 1

    def test_state_immutable(self):
        """测试State数据类的不可变性"""
        positions = pd.DataFrame(
            {"instrument": ["A"], "value": [0.5], "weight": [1.0], "price": [100.0]}
        )
        state = trader.State(
            date=pd.Timestamp("2025-01-01"), value=1.0, positions=positions
        )

        with pytest.raises(AttributeError):
            state.value = 2.0

    def test_context_creation(self):
        """测试Context数据类的创建"""
        positions = pd.DataFrame(
            {"instrument": ["A"], "value": [0.5], "weight": [1.0], "price": [100.0]}
        )
        context = trader.Context(
            date=pd.Timestamp("2025-01-01"),
            is_rebalance_date=True,
            value=1.0,
            positions=positions,
        )

        assert context.date == pd.Timestamp("2025-01-01")
        assert context.is_rebalance_date is True
        assert context.value == 1.0
        assert len(context.positions) == 1


class TestIsRebalanceDate:
    """测试_is_rebalance_date函数"""

    def test_daily_rebalance(self):
        """测试每日调仓"""
        state = trader.State(
            date=pd.Timestamp("2025-01-01"),
            value=1.0,
            positions=pd.DataFrame(columns=["instrument", "value", "weight", "price"]),
        )
        date = pd.Timestamp("2025-01-02")
        assert trader._is_rebalance_date("d", date, state) is True

    def test_weekly_rebalance(self):
        """测试每周调仓"""
        # 周一转到下一周
        state = trader.State(
            date=pd.Timestamp("2025-01-06"),  # 周一
            value=1.0,
            positions=pd.DataFrame(columns=["instrument", "value", "weight", "price"]),
        )
        date = pd.Timestamp("2025-01-07")  # 周二
        assert trader._is_rebalance_date("w", date, state) is False

        date = pd.Timestamp("2025-01-13")  # 下一周周一
        assert trader._is_rebalance_date("w", date, state) is True

    def test_monthly_rebalance(self):
        """测试每月调仓"""
        state = trader.State(
            date=pd.Timestamp("2025-01-15"),
            value=1.0,
            positions=pd.DataFrame(columns=["instrument", "value", "weight", "price"]),
        )
        date = pd.Timestamp("2025-01-20")
        assert trader._is_rebalance_date("m", date, state) is False

        date = pd.Timestamp("2025-02-01")
        assert trader._is_rebalance_date("m", date, state) is True

    def test_quarterly_rebalance(self):
        """测试季度调仓"""
        state = trader.State(
            date=pd.Timestamp("2025-02-15"),
            value=1.0,
            positions=pd.DataFrame(columns=["instrument", "value", "weight", "price"]),
        )
        date = pd.Timestamp("2025-03-15")
        assert trader._is_rebalance_date("q", date, state) is False

        date = pd.Timestamp("2025-04-01")
        assert trader._is_rebalance_date("q", date, state) is True

    def test_yearly_rebalance(self):
        """测试年度调仓"""
        state = trader.State(
            date=pd.Timestamp("2025-06-15"),
            value=1.0,
            positions=pd.DataFrame(columns=["instrument", "value", "weight", "price"]),
        )
        date = pd.Timestamp("2025-12-31")
        assert trader._is_rebalance_date("y", date, state) is False

        date = pd.Timestamp("2026-01-01")
        assert trader._is_rebalance_date("y", date, state) is True


class TestSettlement:
    """测试_settlement函数"""

    def test_empty_positions(self):
        """测试空持仓情况"""
        state = trader.State(
            date=pd.Timestamp("2025-01-01"),
            value=1.0,
            positions=pd.DataFrame(columns=["instrument", "value", "weight", "price"]),
        )
        prices = pd.DataFrame({"instrument": ["A"], "price": [100.0]})

        value, positions = trader._settlement(state, prices)

        assert value == 1.0
        assert positions.empty

    def test_settlement_with_positions(self):
        """测试有持仓的结算"""
        positions = pd.DataFrame(
            {
                "instrument": ["A", "B"],
                "value": [0.5, 0.5],
                "weight": [0.5, 0.5],
                "price": [100.0, 200.0],
            }
        )
        state = trader.State(
            date=pd.Timestamp("2025-01-01"), value=1.0, positions=positions
        )
        prices = pd.DataFrame({"instrument": ["A", "B"], "price": [110.0, 190.0]})

        value, new_positions = trader._settlement(state, prices)

        # 持仓价值应随价格变化
        assert np.isclose(value, (110.0 / 100.0) * 0.5 + (190.0 / 200.0) * 0.5)
        # 权重应重新归一化
        assert np.isclose(new_positions["weight"].sum(), 1.0)

    def test_settlement_missing_price(self):
        """测试价格缺失时使用前一日价格"""
        positions = pd.DataFrame(
            {
                "instrument": ["A", "B"],
                "value": [0.5, 0.5],
                "weight": [0.5, 0.5],
                "price": [100.0, 200.0],
            }
        )
        state = trader.State(
            date=pd.Timestamp("2025-01-01"), value=1.0, positions=positions
        )
        prices = pd.DataFrame({"instrument": ["A", "B"], "price": [110.0, np.nan]})

        value, new_positions = trader._settlement(state, prices)

        # B使用前一日价格200，股价从200跌到200 (因为缺失)，所以不亏不赚
        assert (
            new_positions.loc[new_positions["instrument"] == "B", "price"].values[0]
            == 200.0
        )


class TestTrade:
    """测试_trade函数"""

    def test_no_change_positions(self):
        """测试持仓无变化"""
        positions = pd.DataFrame({"instrument": ["A", "B"], "weight": [0.5, 0.5]})
        new_positions = pd.DataFrame({"instrument": ["A", "B"], "weight": [0.5, 0.5]})

        trades = trader._trade(1.0, positions, new_positions, 0.001, 0.001, 0.0001)

        # 权重变化小于阈值，应无交易
        assert len(trades) == 0

    def test_buy_and_sell(self):
        """测试买入和卖出"""
        positions = pd.DataFrame({"instrument": ["A", "B"], "weight": [0.6, 0.4]})
        new_positions = pd.DataFrame(
            {"instrument": ["A", "B", "C"], "weight": [0.4, 0.3, 0.3]}
        )

        trades = trader._trade(1.0, positions, new_positions, 0.001, 0.001, 0.0001)

        assert len(trades) == 3
        # 检查交易成本计算
        for _, row in trades.iterrows():
            if row["weight_to"] > row["weight_from"]:
                # 买入
                expected_cost = (
                    (row["weight_to"] - row["weight_from"]) * 1.0 * (0.001 + 0.0001)
                )
            else:
                # 卖出
                expected_cost = (
                    (row["weight_from"] - row["weight_to"]) * 1.0 * (0.001 + 0.0001)
                )
            assert np.isclose(row["cost"], expected_cost)

    def test_new_instrument(self):
        """测试新买入标的"""
        positions = pd.DataFrame({"instrument": ["A"], "weight": [1.0]})
        new_positions = pd.DataFrame({"instrument": ["A", "B"], "weight": [0.5, 0.5]})

        trades = trader._trade(1.0, positions, new_positions, 0.001, 0.001, 0.0001)

        # 应该有B的买入交易
        b_trade = trades[trades["instrument"] == "B"]
        assert len(b_trade) == 1
        assert b_trade["weight_from"].values[0] == 0.0

    def test_sell_all(self):
        """测试全部卖出"""
        positions = pd.DataFrame({"instrument": ["A"], "weight": [1.0]})
        new_positions = pd.DataFrame({"instrument": [], "weight": []})

        # 使用concat创建空DataFrame但有正确的列
        new_positions = pd.DataFrame(columns=["instrument", "weight"])

        trades = trader._trade(1.0, positions, new_positions, 0.001, 0.001, 0.0001)

        # 应该有A的卖出交易
        assert len(trades) == 1
        assert trades["weight_from"].values[0] == 1.0


class TestRun:
    """测试run函数"""

    def test_basic_run(self):
        """测试基本运行"""
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
                "instrument": ["A", "A", "A"],
                "price": [100.0, 101.0, 102.0],
            }
        )

        def strategy(context: trader.Context) -> pd.DataFrame:
            return pd.DataFrame({"instrument": ["A"], "weight": [1.0]})

        result = trader.run(
            strategy=strategy,
            rebalance_period="d",
            buy_cost=0.0,
            sell_cost=0.0,
            slip_cost=0.0,
            prices=prices,
        )

        assert isinstance(result, trader.History)
        assert len(result.values) > 0

    def test_run_with_no_rebalance(self):
        """测试非调仓日返回None"""
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
                "instrument": ["A", "A", "A"],
                "price": [100.0, 101.0, 102.0],
            }
        )

        call_count = [0]

        def strategy(context: trader.Context) -> pd.DataFrame | None:
            call_count[0] += 1
            if not context.is_rebalance_date:
                return None
            return pd.DataFrame({"instrument": ["A"], "weight": [1.0]})

        result = trader.run(
            strategy=strategy,
            rebalance_period="m",
            buy_cost=0.0,
            sell_cost=0.0,
            slip_cost=0.0,
            prices=prices,
        )

        # 策略只在调仓日被调用
        assert call_count[0] <= len(prices["date"].unique())

    def test_run_with_costs(self):
        """测试带交易成本的运行"""
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "instrument": ["A", "A"],
                "price": [100.0, 100.0],
            }
        )

        def strategy(context: trader.Context) -> pd.DataFrame:
            return pd.DataFrame({"instrument": ["A"], "weight": [1.0]})

        result = trader.run(
            strategy=strategy,
            rebalance_period="d",
            buy_cost=0.001,
            sell_cost=0.001,
            slip_cost=0.0001,
            prices=prices,
        )

        # 交易成本应被扣除
        assert result.trades["cost"].sum() > 0

    def test_run_with_initial_state(self):
        """测试带初始状态的运行"""
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "instrument": ["A", "A"],
                "price": [100.0, 101.0],
            }
        )

        initial_positions = pd.DataFrame(
            {"instrument": ["A"], "value": [1.0], "weight": [1.0], "price": [100.0]}
        )

        def strategy(context: trader.Context) -> pd.DataFrame:
            return pd.DataFrame({"instrument": ["A"], "weight": [1.0]})

        initial_state = trader.State(
            date=pd.Timestamp("2025-01-01"), value=1.0, positions=initial_positions
        )

        result = trader.run(
            strategy=strategy,
            rebalance_period="d",
            buy_cost=0.0,
            sell_cost=0.0,
            slip_cost=0.0,
            prices=prices,
            state=initial_state,
        )

        assert isinstance(result, trader.History)

    def test_run_weight_normalization(self):
        """测试权重归一化"""
        # 使用不同的价格数据来区分两天
        prices_A = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "instrument": ["A", "A"],
                "price": [100.0, 100.0],
            }
        )
        prices_B = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "instrument": ["B", "B"],
                "price": [200.0, 200.0],
            }
        )
        prices = pd.concat([prices_A, prices_B])

        def strategy(context: trader.Context) -> pd.DataFrame:
            # 返回总和等于1的权重
            return pd.DataFrame({"instrument": ["A", "B"], "weight": [0.5, 0.5]})

        result = trader.run(
            strategy=strategy,
            rebalance_period="d",
            buy_cost=0.0,
            sell_cost=0.0,
            slip_cost=0.0,
            prices=prices,
        )

        # 策略只在2025-01-02被调用(2025-01-01被跳过因为是初始日期)
        # 检查最后一天的权重
        last_date = result.positions["date"].max()
        positions = result.positions[result.positions["date"] == last_date]
        total_weight = positions["weight"].sum()
        # 权重应该保持为1.0
        assert np.isclose(
            total_weight, 1.0
        ), f"Expected weight sum 1.0, got {total_weight}"


class TestRun1d:
    """测试_run1d函数"""

    def test_run1d_basic(self):
        """测试_run1d基本功能"""
        state = trader.State(
            date=pd.Timestamp("2025-01-01"),
            value=1.0,
            positions=pd.DataFrame(columns=["instrument", "value", "weight", "price"]),
        )
        prices = pd.DataFrame({"instrument": ["A"], "price": [100.0]})

        def strategy(context: trader.Context) -> pd.DataFrame:
            return pd.DataFrame({"instrument": ["A"], "weight": [1.0]})

        value, positions, trades = trader._run1d(
            strategy=strategy,
            rebalance_period="d",
            buy_cost=0.0,
            sell_cost=0.0,
            slip_cost=0.0,
            date=pd.Timestamp("2025-01-02"),
            prices=prices,
            state=state,
        )

        assert value > 0
        assert len(positions) > 0

    def test_run1d_returns_none(self):
        """测试策略返回None时保持持仓"""
        positions = pd.DataFrame(
            {"instrument": ["A"], "value": [1.0], "weight": [1.0], "price": [100.0]}
        )
        state = trader.State(
            date=pd.Timestamp("2025-01-01"), value=1.0, positions=positions
        )
        prices = pd.DataFrame({"instrument": ["A"], "price": [100.0]})

        def strategy(context: trader.Context) -> None:
            return None

        value, new_positions, trades = trader._run1d(
            strategy=strategy,
            rebalance_period="d",
            buy_cost=0.0,
            sell_cost=0.0,
            slip_cost=0.0,
            date=pd.Timestamp("2025-01-02"),
            prices=prices,
            state=state,
        )

        assert value == 1.0
        assert len(trades) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
