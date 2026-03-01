import logging
import sys


def setup_logger():
    logger = logging.getLogger("app_logger")

    if logger.handlers:  # защита от дублей
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # INFO и ниже -> info.log
    fh_info = logging.FileHandler("/var/log/optimizer_fastapi_info.log", encoding="utf-8")
    fh_info.setLevel(logging.INFO)
    fh_info.setFormatter(fmt)

    class _InfoOnly(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.levelno <= logging.INFO

    fh_info.addFilter(_InfoOnly())

    # WARNING и выше -> err.log
    fh_err = logging.FileHandler("/var/log/optimizer_fastapi_err.log", encoding="utf-8")
    fh_err.setLevel(logging.WARNING)
    fh_err.setFormatter(fmt)

    # INFO и ниже -> stdout
    h_out = logging.StreamHandler(sys.stdout)
    h_out.setLevel(logging.INFO)
    h_out.setFormatter(fmt)
    h_out.addFilter(_InfoOnly())

    # WARNING и выше -> stderr
    h_err = logging.StreamHandler(sys.stderr)
    h_err.setLevel(logging.WARNING)
    h_err.setFormatter(fmt)

    # записываем в файлы
    logger.addHandler(fh_info)
    logger.addHandler(fh_err)
    #пойдут в journal
    logger.addHandler(h_out)
    logger.addHandler(h_err)
    return logger

logger = setup_logger()