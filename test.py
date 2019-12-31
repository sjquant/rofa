import pandas as pd

from rofa import QuantileSimulator

returns = pd.read_pickle("data/returns.pkl").loc["2002-01-01":]
sim = QuantileSimulator(
    returns.drop(["008080"], axis=1), delay=1, rebalance_freq=5, bins=5
)
sim.run()
sim.plot()
