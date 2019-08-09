from typing import TYPE_CHECKING, Optional, Union
import os
import uuid
import abc

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from pandas.plotting import register_matplotlib_converters
import seaborn as sns

if TYPE_CHECKING:
    from .simulator import Simulator, QuantileSimulator  # noqa
    from matplotlib.axis import Axis

sns.set()
register_matplotlib_converters()


class BasePlotter(abc.ABC):
    def __init__(self, simulator: Optional["Simulator"] = None) -> None:
        if simulator:
            self.simulator = simulator

    def register_simulator(self, simulator: "Simulator") -> None:
        """register simulator"""
        self.simulator = simulator

    def savefig(self, fname: Optional[str]):
        fname = fname or uuid.uuid4().hex
        path = os.path.join(os.getcwd(), fname)
        plt.savefig(path + ".png")

    @abc.abstractmethod
    def plot_portfolio_returns(self):
        pass

    @abc.abstractmethod
    def plot_performance_metrics(self):
        pass

    @abc.abstractmethod
    def plot_rolling_performance(self):
        pass

    @abc.abstractmethod
    def plot_all(self):
        pass


class QuantilePlotter(BasePlotter):
    def _plot_quantile_quarterly_returns(
        self, ax: "Axis", quarterly_returns: pd.DataFrame
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

    def plot_portfolio_returns(self, save: bool = False, **kwargs) -> None:
        """
        plot portfolio returns
        This method plots 3 axes:
        1) cumulative returns 2) monthly returns 3) drawdown

        Args:
            save: when this options is True, save graph as a file

        kwargs:
            fname (Optional[str]): filename when save is True
        """
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(16, 9))

        fig.suptitle("Portfolio Returns", fontsize=20)

        axes[0].set_ylabel("Cumulative Returns")
        axes[1].set_ylabel("Quarterly Returns")
        axes[2].set_ylabel("Drawdown")

        self.simulator: "QuantileSimulator"

        cum_returns = self.simulator.cumulative_returns * 100
        quarterly_returns = self.simulator.quarterly_returns * 100
        drawdown = self.simulator.calculate_drawdown() * 100

        sns.lineplot(data=cum_returns, ax=axes[0], dashes=False)
        self._plot_quantile_quarterly_returns(axes[1], quarterly_returns)
        sns.lineplot(data=drawdown, ax=axes[2], dashes=False, legend=False)
        plt.xlabel("")

        if save:
            self.savefig(kwargs.get("fname"))

        plt.show()

    def _plot_cagr(self, ax: "Axis"):
        cagr = self.simulator.calculate_cagr() * 100
        ax.set_title("CAGR")
        for i, each in enumerate(cagr.iteritems()):
            ax.bar(each[0], each[1], color=sns.color_palette()[i], label=each[0])

    def _plot_mdd(self, ax: "Axis"):
        mdd = self.simulator.calculate_max_drawdown() * 100
        ax.set_title("MDD")
        for i, each in enumerate(mdd.iteritems()):
            ax.bar(each[0], each[1], color=sns.color_palette()[i])

    def _plot_sharpe(self, ax: "Axis", risk_free: Union[float, pd.Series] = 0):
        sharpe = self.simulator.calculate_sharpe_ratio(risk_free)
        ax.set_title("Sharpe Ratio")
        for i, each in enumerate(sharpe.iteritems()):
            ax.bar(each[0], each[1], color=sns.color_palette()[i])

    def _plot_calmar(self, ax: "Axis"):
        calmar = self.simulator.calculate_calmar_ratio()
        ax.set_title("Calmar Ratio")
        for i, each in enumerate(calmar.iteritems()):
            ax.bar(each[0], each[1], color=sns.color_palette()[i])

    def plot_performance_metrics(self, save: bool = False, **kwargs):
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
        self._plot_sharpe(axes[1][0], kwargs.get("risk_free", 0.0))
        self._plot_calmar(axes[1][1])

        if save:
            self.savefig(kwargs.get("fname"))

        axes[0][0].legend()
        plt.show()

    def plot_rolling_performance(self, save: bool = False, **kwargs):
        """
        plot rolling performance
        This method plots 4 axes:
        1) 12m rolling cagr
        2) 36m rolling cagr
        3) 60m rolling cagr
        4) 12m rolling sharpe
        """
        print("It takes some time to calculate rolling peroformance...")

        fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(16, 9))
        fig.suptitle("Rolling Performance", fontsize=20)

        cagr_12m = self.simulator.calculate_rolling_cagr(12) * 100
        cagr_36m = self.simulator.calculate_rolling_cagr(36) * 100
        cagr_60m = self.simulator.calculate_rolling_cagr(60) * 100
        sharpe_12m = self.simulator.calculate_rolling_sharpe(12)

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

        if save:
            self.savefig(kwargs.get("fname"))

        plt.show()

    def plot_all(self):
        """Plot all"""
        self.plot_portfolio_returns()
        self.plot_performance_metrics()
        self.plot_rolling_performance()
