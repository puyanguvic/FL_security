from __future__ import annotations

import logging


_NOISE_PATTERNS = (
    "abort_signal_triggered",
    "ask to abort job",
    "items remained in in_events",
    "ready-to-end-run",
    "get_status returns None in SimEnv",
)


class NvflareNoiseFilter(logging.Filter):
    _patterns = _NOISE_PATTERNS

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._patterns)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def silence_nvflare(level: int = logging.ERROR) -> None:
    if not getattr(logging, "_flbench_noise_filter_installed", False):
        orig_handle = logging.Logger.handle

        def _handle(self: logging.Logger, record: logging.LogRecord):  # type: ignore[override]
            try:
                msg = record.getMessage()
            except Exception:
                return orig_handle(self, record)
            if any(p in msg for p in _NOISE_PATTERNS):
                return None
            return orig_handle(self, record)

        logging.Logger.handle = _handle  # type: ignore[assignment]
        logging._flbench_noise_filter_installed = True  # type: ignore[attr-defined]

    if not getattr(logging, "_flbench_noise_handler_installed", False):
        orig_handler_handle = logging.Handler.handle

        def _handler_handle(self: logging.Handler, record: logging.LogRecord):  # type: ignore[override]
            try:
                msg = record.getMessage()
            except Exception:
                return orig_handler_handle(self, record)
            if any(p in msg for p in _NOISE_PATTERNS):
                return False
            return orig_handler_handle(self, record)

        logging.Handler.handle = _handler_handle  # type: ignore[assignment]
        logging._flbench_noise_handler_installed = True  # type: ignore[attr-defined]

    def _install(logger: logging.Logger) -> None:
        if not any(isinstance(f, NvflareNoiseFilter) for f in logger.filters):
            logger.addFilter(NvflareNoiseFilter())

    nv_logger = logging.getLogger("nvflare")
    nv_logger.setLevel(level)
    _install(nv_logger)
    _install(logging.getLogger())
