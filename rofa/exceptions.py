class SimulationNotFinished(Exception):
    def __init__(self, keyword):
        super().__init__(
            "Run simulation first before using `{}` method.".format(keyword)
        )


class DailyReturnsNotRegistered(Exception):
    def __init__(self):
        super().__init__("Daily returns data have not yet registered.")
