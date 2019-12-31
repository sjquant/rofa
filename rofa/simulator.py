from typing import Union, TYPE_CHECKING
from functools import lru_cache
import abc
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.dates import date2num
from pandas.plotting import register_matplotlib_converters
import seaborn as sns

from .weight_model import EqualWeight
from .exceptions import (
    DailyReturnsNotRegistered,
    SimulationNotFinished,
    SimulationAlreadyRun,
)
from .store import Store
from .logger import logger
from .constants import TRADING_DAYS_IN_YEAR


if TYPE_CHECKING:
    from .weight_model import WeightModel

sns.set()
register_matplotlib_converters()


class BaseSimulator(abc.ABC):
    def __init__(
        self,
        factor: pd.DataFrame,
        rebalance_freq: int = 20,
        weight_model: "WeightModel" = EqualWeight(),
        delay: int = 1,
        **config,
    ):
        self.factor = factor.shift(delay).dropna(how="all")
        self.rebalance_freq = rebalance_freq
        self.weight_model = weight_model
        self.initial_money = 1e7
        self.commission_rate: float = config.get("commission_rate", 0.00015)
        self.tax_rate: float = config.get("tax_rate", 0.003)
        self.slippage: float = config.get("slippage", 0.0)
        self.finished = False

        self._daily_returns = pd.DataFrame()
        self._cumulative_returns = pd.DataFrame()
        self._total_returns = pd.DataFrame()
        self._quarterly_returns = pd.DataFrame()

    @property
    @lru_cache()
    def closes(self):
        store = Store.get()
        closes = store.closes
        if closes is None:
            try:
                closes = pd.read_pickle("data/closes.pkl")
            except FileNotFoundError:
                raise DailyReturnsNotRegistered
        return closes

    @abc.abstractmethod
    def run(self):
        pass

    @property
    def daily_returns(self) -> pd.DataFrame:
        if not self._daily_returns.empty:
            return self._daily_returns
        else:
            raise SimulationNotFinished("daily_returns")

    @daily_returns.setter
    def daily_returns(self, daily_returns):
        self._daily_returns = daily_returns

    @property
    def cumulative_returns(self) -> pd.DataFrame:
        if self._cumulative_returns.empty:
            self._cumulative_returns = (self.daily_returns + 1).cumprod() - 1
        return self._cumulative_returns

    @property
    def total_returns(self) -> pd.Series:
        if self._total_returns.empty:
            self._total_returns = self.cumulative_returns.iloc[-1]
        return self._total_returns

    @property
    def quarterly_returns(self) -> pd.DataFrame:
        if self._quarterly_returns.empty:
            self._quarterly_returns = (self.daily_returns + 1).groupby(
                pd.Grouper(freq="Q")
            ).prod() - 1
        return self._quarterly_returns

    @abc.abstractmethod
    def plot_portfolio_returns(self):
        pass

    @abc.abstractmethod
    def plot_performance_metrics(self, risk_fre=0.0):
        pass

    @abc.abstractmethod
    def plot_rolling_performance(self):
        pass

    def plot(self, risk_free=0.0, **kwargs):
        self.plot_portfolio_returns()
        self.plot_performance_metrics(risk_free)
        self.plot_rolling_performance()

        save = kwargs.pop("save", False)
        if save:
            try:
                path = kwargs["path"]
                plt.savefig(path + ".png")
            except KeyError:
                raise KeyError("path must be specified to save")


class QuantileSimulator(BaseSimulator):

    """
    Quantile Simulator divides data into # of bins
    based on specified quantile, and track simulation results of each.
    """

    def __init__(
        self,
        factor: pd.DataFrame,
        rebalance_freq: int = 20,
        weight_model: "WeightModel" = EqualWeight(),
        bins: int = 10,
        delay: int = 1,
        **config,
    ) -> None:
        super().__init__(factor, rebalance_freq, weight_model, delay, **config)
        if bins < 0 or bins > 10:
            raise OverflowError("bins must be integer between 1 and 10")
        self.bins = bins

    def _discretize_based_on_quantile(self, data: pd.DataFrame) -> pd.DataFrame:
        # The higher the values, the higher the rank
        rank = data.rank(method="first", axis=1, ascending=True)
        labels = range(1, self.bins + 1)
        discretized = rank.apply(
            lambda x: pd.qcut(x, self.bins, labels), axis=1, raw=True
        )
        return discretized

    def _calculate_port_value(self, weight_df: pd.DataFrame) -> pd.Series:
        last_port_value = 1e7
        port_values = []
        for i in range(len(weight_df)):
            weights = weight_df.iloc[i].dropna()
            amount = (last_port_value * weights).divide(
                self.closes.loc[weights.name, weights.index]
            )
            try:
                next_rebalance_dt = weight_df.iloc[i + 1].name
                partitioned_closes = self.closes.loc[
                    weights.name : next_rebalance_dt, weights.index
                ].iloc[:-1]
            except IndexError:
                next_rebalance_dt = self.factor.iloc[-1].name
                partitioned_closes = self.closes.loc[
                    weights.name : next_rebalance_dt, weights.index
                ]
            amount = (
                pd.DataFrame(amount)
                .T.reindex(partitioned_closes.index)
                .fillna(method="ffill")
            )
            partitioned_port_values = partitioned_closes.multiply(amount).sum(axis=1)
            last_port_value = (
                self.closes.loc[next_rebalance_dt, weights.index]
                .multiply(amount.iloc[-1])
                .sum()
            )
            port_values.append(partitioned_port_values)
        port_values = pd.concat(port_values)
        return port_values

    def calculate_cagr(self):
        cagr = (1 + self.total_returns) ** (
            TRADING_DAYS_IN_YEAR / len(self.daily_returns)
        ) - 1
        return cagr

    def calculate_adjusted_returns(
        self, risk_free: Union[float, pd.Series] = 0.0
    ) -> pd.DataFrame:
        if isinstance(risk_free, (int, float)):
            adjusted_returns = self.daily_returns - risk_free
        else:
            adjusted_returns = self.daily_returns.subtract(risk_free)
        return adjusted_returns

    def calculate_sharpe_ratio(
        self, risk_free: Union[float, pd.Series] = 0.0
    ) -> pd.Series:
        adjusted_returns = self.calculate_adjusted_returns(risk_free)
        ann_avg_returns = adjusted_returns.mean() * TRADING_DAYS_IN_YEAR
        ann_returns_std = adjusted_returns.std() * math.sqrt(TRADING_DAYS_IN_YEAR)
        sharpe = ann_avg_returns / ann_returns_std
        return sharpe

    @lru_cache()
    def calculate_drawdown(self) -> pd.DataFrame:
        roll_max = (
            (1 + self.cumulative_returns)
            .rolling(TRADING_DAYS_IN_YEAR, min_periods=1)
            .max()
        )
        drawdown = (1 + self.cumulative_returns).divide(roll_max, axis=0) - 1
        return drawdown

    def calculate_max_drawdown(self):
        drawdown = self.calculate_drawdown()
        max_drawdown = drawdown.min()
        return max_drawdown

    def calculate_calmar_ratio(self):
        cagr = self.calculate_cagr()
        mdd = self.calculate_max_drawdown()
        calmar = -cagr / mdd
        return calmar

    def calculate_rolling_cagr(
        self, months: int = 12, min_months: int = 12
    ) -> pd.DataFrame:

        window = months * 21
        min_window = min_months * 21

        # rolling_cagr is calculated as follows
        # rolling_returns = rollilng_last_value / rolling_first_value - 1
        # rolling_cagr = (rolling_returns + 1) ** (TRADING_DAYS_IN_YEAR/window)

        rolling_cagr = (
            (self.cumulative_returns + 1)
            .rolling(window=window, min_periods=min_window)
            .apply(
                lambda x: ((x[-1] / x[0]) ** (TRADING_DAYS_IN_YEAR / window)) - 1,
                raw=True,
            )
        )
        return rolling_cagr

    def calculate_rolling_sharpe(
        self,
        months: int = 12,
        min_months: int = 12,
        risk_free: Union[float, pd.Series] = 0.0,
    ) -> pd.DataFrame:
        window = months * 21
        min_window = min_months * 21

        def _calc_sharpe(rows: np.array):
            if isinstance(risk_free, (int, float)):
                adjusted_returns = rows - risk_free
            else:
                adjusted_returns = rows.subtract(risk_free)
            ann_avg_returns = adjusted_returns.mean() * TRADING_DAYS_IN_YEAR
            ann_returns_std = adjusted_returns.std() * math.sqrt(TRADING_DAYS_IN_YEAR)
            sharpe = ann_avg_returns / ann_returns_std
            return sharpe

        rolling_sharpe = self.daily_returns.rolling(
            window=window, min_periods=min_window
        ).apply(_calc_sharpe, raw=True)

        return rolling_sharpe

    def run(self):
        """run simulation"""
        logger.info(f"Start running simulation...")
        if self.finished:
            raise SimulationAlreadyRun
        daily_returns = dict()
        factor_to_rebalance = self.factor.iloc[:: self.rebalance_freq]
        discretized = self._discretize_based_on_quantile(factor_to_rebalance)
        for i in range(1, self.bins + 1):
            logger.info(f"Start bins: {i}")
            grouped = factor_to_rebalance[~discretized[discretized == i].isnull()]
            weight_df = self.weight_model.calculate(grouped)
            port_value = self._calculate_port_value(weight_df)
            daily_returns[i] = port_value.pct_change()
            logger.info(f"Finish bins: {i}")
        self.daily_returns = pd.DataFrame(daily_returns)
        self.finished = True

    def _plot_quantile_quarterly_returns(
        self, ax: Axis, quarterly_returns: pd.DataFrame
    ) -> None:

        width = 85 / len(quarterly_returns.columns)

        for i, col in enumerate(quarterly_returns.columns):
            ax.bar(
                date2num(quarterly_returns.index) + i * width,
                quarterly_returns[col].values,
                width=width,
                color=sns.color_palette()[i],
                linewidth=0,
            )

        ax.xaxis_date()
        ax.autoscale(tight=True)

    def plot_portfolio_returns(self) -> None:
        """
        plot portfolio returns
        This method plots 3 axes:
        1) cumulative returns 2) monthly returns 3) drawdown
        """
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(16, 9))
        fig.suptitle("Portfolio Returns", fontsize=20)

        axes[0].set_ylabel("Cumulative Returns")
        axes[1].set_ylabel("Quarterly Returns")
        axes[2].set_ylabel("Drawdown")
        cum_returns = self.cumulative_returns * 100
        quarterly_returns = self.quarterly_returns * 100
        drawdown = self.calculate_drawdown() * 100

        sns.lineplot(data=cum_returns, ax=axes[0], dashes=False)
        self._plot_quantile_quarterly_returns(axes[1], quarterly_returns)
        sns.lineplot(data=drawdown, ax=axes[2], dashes=False, legend=False)
        plt.xlabel("")
        plt.show()

    def _plot_cagr(self, ax: Axis):
        cagr = self.calculate_cagr() * 100
        ax.set_title("CAGR")
        for i, each in enumerate(cagr.iteritems()):
            ax.bar(each[0], each[1], color=sns.color_palette()[i], label=each[0])

    def _plot_mdd(self, ax: Axis):
        mdd = self.calculate_max_drawdown() * 100
        ax.set_title("MDD")
        for i, each in enumerate(mdd.iteritems()):
            ax.bar(each[0], each[1], color=sns.color_palette()[i])

    def _plot_sharpe(self, ax: Axis, risk_free: Union[float, pd.Series] = 0):
        sharpe = self.calculate_sharpe_ratio(risk_free)
        ax.set_title("Sharpe Ratio")
        for i, each in enumerate(sharpe.iteritems()):
            ax.bar(each[0], each[1], color=sns.color_palette()[i])

    def _plot_calmar(self, ax: Axis):
        calmar = self.calculate_calmar_ratio()
        ax.set_title("Calmar Ratio")
        for i, each in enumerate(calmar.iteritems()):
            ax.bar(each[0], each[1], color=sns.color_palette()[i])

    def plot_performance_metrics(self, risk_free=0.0):
        """
        plot performance metrics
        This method plots 2 x 2 axes:
        1) CAGR 2) MDD
        3) Sharpe 4) Calmar
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(16, 9))
        fig.suptitle("Performance Metrics", fontsize=20)

        self._plot_cagr(axes[0][0])
        self._plot_mdd(axes[0][1])
        self._plot_sharpe(axes[1][0], risk_free)
        self._plot_calmar(axes[1][1])
        axes[0][0].legend()
        plt.show()

    def plot_rolling_performance(self):
        """
        plot rolling performance
        This method plots 4 axes:
        1) 12m rolling cagr
        2) 36m rolling cagr
        3) 60m rolling cagr
        4) 12m rolling sharpe
        """
        logger.info("It takes some time to calculate rolling peroformance...")

        fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(16, 9))
        fig.suptitle("Rolling Performance", fontsize=20)

        cagr_12m = self.calculate_rolling_cagr(12) * 100
        cagr_36m = self.calculate_rolling_cagr(36) * 100
        cagr_60m = self.calculate_rolling_cagr(60) * 100
        sharpe_12m = self.calculate_rolling_sharpe(12)

        axes[0].set_title("Rolling CAGR", fontsize=16)
        axes[0].set_ylabel("1Y CAGR")
        axes[1].set_ylabel("3Y CAGR")
        axes[2].set_ylabel("5Y CAGR")
        axes[3].set_title("Rolling Sharpe", fontsize=16)
        axes[3].set_ylabel("1Y Sharpe")

        sns.lineplot(data=cagr_12m, ax=axes[0], dashes=False)
        sns.lineplot(data=cagr_36m, ax=axes[1], dashes=False, legend=False)
        sns.lineplot(data=cagr_60m, ax=axes[2], dashes=False, legend=False)
        sns.lineplot(data=sharpe_12m, ax=axes[3], dashes=False, legend=False)
        plt.xlabel("")
        axes[0].legend()
        plt.show()
