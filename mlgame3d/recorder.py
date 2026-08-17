"""
Recorder Module

This module collects the warnings and errors that occur while running a game and
writes them to ``warning.json`` / ``error.json`` in the record folder, using the
same file format as MLGame's ProgressLogExecutor.
"""

import datetime
import os
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from mlgame3d.utils.io import write_json
from mlgame3d.utils.logger import logger


class ErrorEnum(str, Enum):
    AI_INIT_ERROR = "AI_INIT_ERROR"
    AI_EXEC_ERROR = "AI_EXEC_ERROR"
    GAME_EXEC_ERROR = "GAME_EXEC_ERROR"
    COMMAND_ERROR = "COMMAND_ERROR"


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


@dataclass
class GameWarning:
    """Data structure for game warnings (e.g. MLPlay timeout)"""

    message: str
    step: int
    episode: int
    time_stamp: datetime.datetime = field(default_factory=_utc_now)


@dataclass
class GameError:
    """Data structure for game errors"""

    error_type: ErrorEnum
    message: str
    step: int
    episode: int
    time_stamp: datetime.datetime = field(default_factory=_utc_now)


class GameRecorder:
    """
    Collects warnings and errors raised while running a game.

    Every warning/error is logged to the console. When a record folder is set,
    it is also appended to ``warning.json`` / ``error.json`` in that folder
    (the whole list is rewritten on each update, like MLGame's recorder).

    ``episode`` and ``step`` are updated by the game runner so that each
    record carries the position in the game where it happened. ``step`` is
    the number of ``env.step()`` calls made so far in the current episode
    (i.e. the decision index, not the Unity frame count).
    """

    def __init__(self) -> None:
        self.record_folder: Optional[str] = None
        self.episode: int = 0
        self.step: int = 0
        self._warnings: List[GameWarning] = []
        self._errors: List[GameError] = []

    def set_record_folder(self, folder: Optional[str]) -> None:
        """
        Set the folder where warning.json / error.json are written.
        Pass None to disable file recording (console logging is unaffected).
        """
        self.record_folder = folder

    @property
    def warnings(self) -> List[GameWarning]:
        return self._warnings

    @property
    def errors(self) -> List[GameError]:
        return self._errors

    def warning(self, message: str) -> None:
        """Log a warning and record it in warning.json."""
        logger.opt(depth=1).warning(message)
        self._warnings.append(GameWarning(message=message, step=self.step, episode=self.episode))
        self._write("warning.json", self._warnings)

    def error(self, error_type: ErrorEnum, message: str) -> None:
        """Log an error and record it in error.json."""
        logger.opt(depth=1).error(message)
        self._append_error(error_type, message)

    def exception(self, error_type: ErrorEnum, message: str) -> None:
        """
        Log the exception currently being handled (with traceback) and record it
        in error.json. Must be called from within an ``except`` block.
        """
        logger.opt(depth=1, exception=True).error(message)
        self._append_error(error_type, f"{message}\n{traceback.format_exc()}")

    def _append_error(self, error_type: ErrorEnum, message: str) -> None:
        self._errors.append(GameError(error_type=error_type, message=message, step=self.step, episode=self.episode))
        self._write("error.json", self._errors)

    def _write(self, filename: str, data) -> None:
        if self.record_folder is None:
            return
        path = os.path.join(self.record_folder, filename)
        try:
            write_json(path, data)
        except OSError as e:
            # Do not go through self.error() here to avoid recursing on a broken record folder
            logger.error(f"Failed to write {path}: {e}")


recorder = GameRecorder()
