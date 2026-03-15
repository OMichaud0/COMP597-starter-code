#!/usr/bin/env python3
import sys
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class Logger:
    """Simple logger with timestamps and module prefixes."""

    def __init__(self, module: str, level: LogLevel = LogLevel.INFO):
        """
        Initialize logger.

        Args:
            module: Module/prefix name (e.g., 'PROBE', 'MAIN', 'AGGREGATE')
            level: Minimum logging level to display (default: INFO)
        """
        self.module = module
        self.level = level

    def _format_message(self, level_name: str, message: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [{self.module}] [{level_name}] {message}"

    def debug(self, message: str) -> None:
        if self.level.value <= LogLevel.DEBUG.value:
            print(self._format_message("DEBUG", message), flush=True)

    def info(self, message: str) -> None:
        if self.level.value <= LogLevel.INFO.value:
            print(self._format_message("INFO", message), flush=True)

    def warning(self, message: str) -> None:
        if self.level.value <= LogLevel.WARNING.value:
            print(self._format_message("WARNING", message), flush=True, file=sys.stderr)

    def error(self, message: str) -> None:
        if self.level.value <= LogLevel.ERROR.value:
            print(self._format_message("ERROR", message), flush=True, file=sys.stderr)
