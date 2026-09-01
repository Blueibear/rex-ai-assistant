"""Cross-platform single-instance lock for the Rex background supervisor."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import BinaryIO


class AlreadyRunningError(RuntimeError):
    """Raised when another process already owns the runtime lock."""


class SingleInstanceLock:
    """Hold an advisory one-process lock for the lifetime of this object."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: BinaryIO | None = None
        self._locked = False

    def __enter__(self) -> "SingleInstanceLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def acquire(self) -> None:
        """Acquire the lock non-blockingly or raise ``AlreadyRunningError``."""

        if self._locked:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            self._lock_handle(handle)
        except BaseException:
            handle.close()
            raise

        self._handle = handle
        self._locked = True

    def close(self) -> None:
        """Release the advisory lock and close its owning file handle."""

        handle = self._handle
        if handle is None:
            return

        try:
            if self._locked:
                self._unlock_handle(handle)
        finally:
            self._locked = False
            self._handle = None
            handle.close()

    @staticmethod
    def _lock_handle(handle: BinaryIO) -> None:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise AlreadyRunningError("Rex background runtime is already running") from exc
                raise
            return

        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise AlreadyRunningError("Rex background runtime is already running") from exc
            raise

    @staticmethod
    def _unlock_handle(handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
