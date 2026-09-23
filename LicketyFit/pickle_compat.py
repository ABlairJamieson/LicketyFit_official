"""Compatibility helpers for NumPy-containing calibration pickles."""

from __future__ import annotations

import pickle
from typing import Any, BinaryIO


class _NumPyCompatibleUnpickler(pickle.Unpickler):
    """Map NumPy 2's private ``numpy._core`` name for NumPy 1 runtimes."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def load_numpy_pickle(stream: BinaryIO) -> Any:
    """Load a pickle written by NumPy 1.x or 2.x.

    NumPy 2 serializes array reconstruction functions under the private module
    name ``numpy._core``. NumPy 1.x exposes the equivalent implementation as
    ``numpy.core``, so ordinary ``pickle.load`` fails before the array can be
    reconstructed. The public array representation is otherwise compatible.
    """

    return _NumPyCompatibleUnpickler(stream).load()
