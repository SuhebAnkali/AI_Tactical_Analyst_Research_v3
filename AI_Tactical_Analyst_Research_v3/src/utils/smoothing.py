"""Temporal smoothing utilities."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Generic, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class MajorityVoteSmoother(Generic[T]):
    """Temporal majority vote smoother."""

    window_size: int
    history: Deque[T] = field(default_factory=deque)

    def update(self, value: T) -> T:
        self.history.append(value)
        while len(self.history) > self.window_size:
            self.history.popleft()
        counts = Counter(self.history)
        return counts.most_common(1)[0][0]


@dataclass
class ExponentialSmoother:
    """Scalar/vector exponential smoothing."""

    alpha: float = 0.2
    state: np.ndarray | None = None

    def update(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        if self.state is None:
            self.state = value.copy()
        else:
            self.state = self.alpha * value + (1.0 - self.alpha) * self.state
        return self.state.copy()
