from functools import wraps
from .exceptions import SimulationNotFinished


def check_simulation_finished(keyword):
    """decorator for checking if simuatlion runned"""

    def wrapper(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            self = args[0]
            if not self.finished:
                raise SimulationNotFinished(keyword)
            return func(*args, **kwargs)

        return wrapped

    return wrapper
