import logging

from tqdm import tqdm


class SmartTqdmHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if hasattr(tqdm, "_instances") and len(tqdm._instances) > 0:
                tqdm.write(msg)
            else:
                super().emit(record)
        except Exception:
            self.handleError(record)


def setup_logging(level: str = "INFO") -> None:
    logger = logging.getLogger()
    logger.setLevel(level.upper())

    handler = SmartTqdmHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)
