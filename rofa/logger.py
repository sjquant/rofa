import sys
import logging
import logging.config

LOGGING_CONFIG = dict(
    version=1,
    loggers={"rofa": {"level": "INFO", "handlers": ["console"]}},
    handlers={
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "generic",
            "stream": sys.stdout,
        }
    },
    formatters={
        "generic": {
            "format": "%(asctime)s [%(process)d] [%(levelname)s] %(message)s",
            "datefmt": "[%Y-%m-%d %H:%M:%S %z]",
            "class": "logging.Formatter",
        }
    },
)

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("rofa")
