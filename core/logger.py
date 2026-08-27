import logging
import config
from core.logging_config import setup_logging, get_logger as _get_logger

def get_logger(name):
    # Reconfigure root if not already set up.
    if not logging.getLogger().handlers:
        setup_logging(
            level=logging.DEBUG if config.DEBUG_VERBOSE else logging.INFO,
            json_logging=getattr(config, "ENABLE_JSON_LOGGING", False)
        )
    return _get_logger(name)
