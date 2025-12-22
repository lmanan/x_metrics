"""Base adapter interface for metrics."""

from abc import ABC, abstractmethod
from typing import Any


class BaseMetricAdapter(ABC):
    """Abstract base class for metric adapters."""

    @abstractmethod
    def compute(self) -> dict[str, Any]:
        """Compute the metric and return results as a dictionary."""
        pass
