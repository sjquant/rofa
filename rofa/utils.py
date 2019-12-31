import os

import pandas as pd

from .store import Store


def register_closes(df: pd.DataFrame, save=True) -> None:
    print("Registering daily close prices...")
    store = Store.get()
    store.returns = df
    if not os.path.isdir("data"):
        os.mkdir("data")
        df.to_pickle("data/close.pkl")
    print("Daily close prices were successfuly registerd.")
