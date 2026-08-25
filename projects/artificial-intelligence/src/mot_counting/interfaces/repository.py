"""Event repository interface (§4.2, §4.5, §6, §10.6).

Defines the ``IEventRepository`` contract — the only component allowed to
persist crossing events to storage (CSV in v1).
"""

from abc import ABC, abstractmethod

from mot_counting.types import CrossingEvent


class IEventRepository(ABC):
    """Abstract interface for persisting crossing events."""

    @abstractmethod
    def save(self, event: CrossingEvent) -> None:
        """Persist a single crossing event.

        Args:
            event: A validated :class:`~mot_counting.types.CrossingEvent` to
                store.  Implementations may buffer events internally and
                write them on :meth:`flush`.
        """

    @abstractmethod
    def flush(self) -> None:
        """Write any buffered events to the underlying storage."""

    @abstractmethod
    def close(self) -> None:
        """Flush remaining data and release all storage resources."""
