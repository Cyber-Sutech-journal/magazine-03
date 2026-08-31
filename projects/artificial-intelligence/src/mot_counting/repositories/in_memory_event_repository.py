from mot_counting.interfaces.repository import IEventRepository
from mot_counting.types import CrossingEvent


class InMemoryEventRepository(IEventRepository):
    """
    In-memory implementation of the event repository.
    Used primarily for testing and transient data storage.
    """

    def __init__(self) -> None:
        self.events: list[CrossingEvent] = []

    def save(self, event: CrossingEvent) -> None:
        self.events.append(event)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
