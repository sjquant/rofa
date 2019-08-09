import abc
import pandas as pd


class WeightModel(abc.ABC):
    @abc.abstractmethod
    def calculate(self, data: pd.DataFrame):
        pass


class EqualWeight(WeightModel):
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        # replace all non-Na values with 1
        # and divide with non-Na count
        weighted = data.where(data.isnull(), 1).divide(data.count(axis=1), axis=0)
        return weighted


class OriginalWeight(WeightModel):
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        weighted = data.divide(data.sum(axis=1), axis=0)
        return weighted


class ExponentialWeight(WeightModel):
    def __init__(self, exponent: float = 2.0) -> None:
        if exponent < 1:
            raise ValueError("exponent must be less than or eqault to 1.0")
        self.exponent = exponent

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data ** self.exponent
        weighted = data.divide(data.sum(axis=1), axis=0)
        return weighted
