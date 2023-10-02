import logging
from enum import Enum


class LogLevel(Enum):
    CRITICAL = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4


def setup_logging():
    logging.config.fileConfig("logging.conf")


def log(msg: any, level: LogLevel = LogLevel.DEBUG) -> None:
    """
    日志级别：CRITICAL(0) > ERROR(1) > WARNING(2) > INFO(3) > DEBUG(4) 如果未指定日志级别，将直接打印到终端
    """
    msg = str(msg)
    if not isinstance(level, LogLevel):
        level = LogLevel(level)

    logger = logging.getLogger()

    if level == LogLevel.DEBUG:
        logger.debug(msg)
    elif level == LogLevel.INFO:
        logger.info(msg)
    elif level == LogLevel.WARNING:
        logger.warning(msg)
    elif level == LogLevel.ERROR:
        logger.error(msg)
    elif level == LogLevel.CRITICAL:
        logger.critical(msg)
    else:
        print(msg)
