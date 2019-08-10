from typing import Any, Union, TYPE_CHECKING
from functools import lru_cache
import abc
import math
import os

import numpy as np
import pandas as pd

from .weight_model import EqualWeight
from .plotter import QuantilePlotter
from .utils import check_simulation_finished
from .exceptions import DailyReturnsNotRegistered

if TYPE_CHECKING:
    from .weight_model import WeightModel
DAYS_PER_YEAR = 252

returns = None


def register_returns(df: pd.DataFrame, save=True) -> None:
    global returns

    print("Registering daily returns...")
    returns = df

    if not os.path.isdir("data"):
        os.mkdir("data")
        returns.to_pickle("data/returns.pkl")
    print("Daily returns were successfuly registerd.")


class Simulator:
    def __init__(self, data: pd.DataFrame, rebalance_freq: int, **kwargs):
        self.data = data
        self.rebalance_freq = rebalance_freq
        self.weight_model = kwargs.get("weight_model", None)

        self.commission_rate: float = kwargs.get("commission_rate", 0.00015)
        self.tax_rate: float = kwargs.get("tax_rate", 0.003)
        self.slippage: float = kwargs.get("slippage", 0.0)

        self.finished = False

    @property
    @lru_cache()
    def returns(self):
        global returns

        if returns is None:
            try:
                returns = pd.read_pickle("data/returns.pkl")
            except FileNotFoundError:
                raise DailyReturnsNotRegistered
        return returns

    def register_app(self, app: Any):
        app.register_simulator(self)

    def _adjust_weights(
        self, weighted: pd.DataFrame, long: bool = True
    ) -> pd.DataFrame:
        """
        Adjust weights day by day to reflect on daily change of returns

        Args:
            weighted: weighted dataframe
            long: whether it is for long or short
        """
        returns = self.returns[weighted.columns]
        returns = returns.loc[weighted.index[0] :]

        weighted = weighted.reindex(returns.index)

        sign = 1 if long else -1
        for i in range(len(weighted)):
            # if all values are note np.nan,
            # it is rebalancing weights, so jump to next iteration
            if not weighted.iloc[i].isnull().all():
                continue

            # ajdust weights to reflect returns
            # final_weights are calculated as follows
            if long:
                # If long plus return has positive effects on weights
                affected_weight = weighted.iloc[i - 1] * (1 + returns.iloc[i])
            else:
                # otherwise, it has negative effects on weights
                affected_weight = weighted.iloc[i - 1] * (1 - returns.iloc[i])

            weighted.iloc[i] = affected_weight / affected_weight.sum()

        return sign * weighted

    def calculate_daily_returns(self, weighted: pd.DataFrame) -> pd.Series:
        """
        calculate daily returns using weighted and returns

        Args:
            weighted: weighted dataframe adjusted from `_adjust_weights`

        Returns:
            daily_returns: daily return dataframe
        """
        returns = self.returns[weighted.columns]
        returns = returns.loc[weighted.index[0] :]

        daily_returns = (weighted.shift(1) * returns).sum(axis=1)
        return daily_returns

    @abc.abstractmethod
    def run(self):
        pass


class QuantileSimulator(Simulator):

    """
    Quantile Simulator divides data into # of bins
    based on specified quantile, and track simulation results of each.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        rebalance_freq: int = 20,
        weight_model: "WeightModel" = EqualWeight(),
        bins: int = 10,
        **kwargs
    ) -> None:
        super().__init__(data, rebalance_freq, weight_model=weight_model, **kwargs)
        self.bins = bins

        self.daily_returns = pd.DataFrame()
        self._cumulative_returns = pd.DataFrame()
        self._total_returns = pd.Series()
        self._quarterly_returns = pd.DataFrame()

        self.plotter = QuantilePlotter(self)

    @property
    @check_simulation_finished("cumulative_returns")
    def cumulative_returns(self) -> pd.DataFrame:
        if self._cumulative_returns.empty:
            self._cumulative_returns = (self.daily_returns + 1).cumprod() - 1
        return self._cumulative_returns

    @property
    @check_simulation_finished("total_returns")
    def total_returns(self) -> pd.Series:
        if self._total_returns.empty:
            self._total_returns = self.cumulative_returns.iloc[-1]
        return self._total_returns

    @property
    @check_simulation_finished("quarterly_returns")
    def quarterly_returns(self) -> pd.DataFrame:
        if self._quarterly_returns.empty:
            self._quarterly_returns = (self.daily_returns + 1).groupby(
                pd.Grouper(freq="Q")
            ).prod() - 1
        return self._quarterly_returns

    def _discretize_based_on_quantile(self, data: pd.DataFrame) -> pd.DataFrame:
        # The higher the values, the higher the rank
        rank = data.rank(method="first", axis=1, ascending=True)
        labels = range(1, self.bins + 1)
        discretized = rank.apply(
            lambda x: pd.qcut(x, self.bins, labels), axis=1, raw=True
        )
        return discretized

    def _create_generater(self):
        data_to_rebalance = self.data.iloc[:: self.rebalance_freq]
        discretized = self._discretize_based_on_quantile(data_to_rebalance)
        for i in range(1, self.bins + 1):
            grouped = data_to_rebalance[~discretized[discretized == i].isnull()]
            weighted = self.weight_model.calculate(grouped)
            weights_adjusted = self._adjust_weights(weighted)
            daily_returns = self.calculate_daily_returns(weights_adjusted)
            # Apply Tax and Commission
            daily_returns.loc[data_to_rebalance.index] -= (
                self.tax_rate + self.commission_rate + self.slippage
            )
            yield daily_returns

    def calculate_cagr(self):
        cagr = (1 + self.total_returns) ** (DAYS_PER_YEAR / len(self.daily_returns)) - 1
        return cagr

    def _calculate_adjusted_returns(
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
        adjusted_returns = self._calculate_adjusted_returns(risk_free)
        ann_avg_returns = adjusted_returns.mean() * DAYS_PER_YEAR
        ann_returns_std = adjusted_returns.std() * math.sqrt(DAYS_PER_YEAR)
        sharpe = ann_avg_returns / ann_returns_std
        return sharpe

    @lru_cache()
    def calculate_drawdown(self) -> pd.DataFrame:
        roll_max = (
            (1 + self.cumulative_returns).rolling(DAYS_PER_YEAR, min_periods=1).max()
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

        # rolling_cagr is calculated as folows
        # rolling_returns = rollilng_last_value / rolling_first_value - 1
        # rooling_cagr = (rolling_returns + 1) ** (DAYS_PER_YEAR/window)

        rolling_cagr = (
            (self.cumulative_returns + 1)
            .rolling(window=window, min_periods=min_window)
            .apply(lambda x: ((x[-1] / x[0]) ** (DAYS_PER_YEAR / window)) - 1, raw=True)
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
            ann_avg_returns = adjusted_returns.mean() * DAYS_PER_YEAR
            ann_returns_std = adjusted_returns.std() * math.sqrt(DAYS_PER_YEAR)
            sharpe = ann_avg_returns / ann_returns_std
            return sharpe

        rolling_sharpe = self.daily_returns.rolling(
            window=window, min_periods=min_window
        ).apply(_calc_sharpe, raw=True)

        return rolling_sharpe

    def run(self):
        """run simulation"""
        daily_returns = dict()
        for i, each in enumerate(self._create_generater()):
            daily_returns[i + 1] = each

        self.daily_returns = pd.DataFrame(daily_returns)
        self.finished = True


class LongShortSimulator(Simulator):

    """
    Long-Short Simulator picks data top and bottom P%,
    (which is determined by `ls_percentage` parameter)
    and long top group and short bottom group.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        rebalance_freq: int = 20,
        weight_model: "WeightModel" = EqualWeight(),
        ls_percentage: float = 0.2,
        gross_leverage: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__(data, rebalance_freq, weight_model=weight_model, **kwargs)
        self.ls_percentage = ls_percentage
        self.gross_leverage = gross_leverage

        self.daily_returns = None
        self._cumulative_returns = None

    @property
    @check_simulation_finished("cumulative_returns")
    def cumulative_returns(self) -> pd.Series:

        daily_returns: pd.DataFrame = self.daily_returns
        if not self._cumulative_returns:
            self._cumulative_returns = (daily_returns + 1).cumprod() - 1

        return self._cumulative_returns

    def _split_long_short(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        split data into long(1) and short(-1)
        for certain percentage (ls_percentage)
        of each long and short
        """
        # The higher the values, the higher the rank
        rank = data.rank(method="first", axis=1, ascending=True)

        def split_long_short(row):
            count = int(len(row.dropna()) * self.ls_percentage)
            long = row.nlargest(count).where(row.isnull(), 1)
            short = row.nsmallest(count).where(row.isnull(), -1)
            return pd.concat([long, short])

        splited = rank.apply(split_long_short, axis=1)
        return splited

    def run(self):
        """run simulation"""
        data_to_rebalance = self.data.iloc[:: self.rebalance_freq]
        splited = self._split_long_short(data_to_rebalance)
        long_data = data_to_rebalance[~splited[splited == 1].isnull()]
        short_data = data_to_rebalance[~splited[splited == -1].isnull()]

        # cacluate adjusted weights for long and short
        long_weighted = self.weight_model.calculate(long_data)
        short_weighted = self.weight_model.calculate(short_data)
        long_weights_adjusted = self._adjust_weights(long_weighted)
        short_weights_adjusted = self._adjust_weights(short_weighted, long=False)
        # and combined them
        weights_adjusted = long_weights_adjusted.fillna(short_weights_adjusted) / (
            2 / self.gross_leverage
        )

        daily_returns = self.calculate_daily_returns(weights_adjusted)
        # Apply Tax and Commission
        daily_returns.loc[data_to_rebalance.index] -= (
            self.tax_rate + self.commission_rate + self.slippage
        )
        self.daily_returns = daily_returns
        self.finished = True


# TODO
class NakedSimulator(Simulator):
    """
    Naked Simulator determine weights and long-short based on values itself.
    It longs assets with positive value and shorts assetes with negative value,
    and do nothing with 0 valued assets.
    It's similar to use QuantileSimulator with OriginalWeightModel, but
    it does not divides assets into quantile.
    In that sense, it has some similarity to LongShortSimulator
    """

    pass
