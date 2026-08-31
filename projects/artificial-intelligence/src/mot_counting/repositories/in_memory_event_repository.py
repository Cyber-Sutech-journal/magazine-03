from mot_counting.interfaces.repository import IEventRepository
from mot_counting.types import CrossingEvent


class InMemoryEventRepository(IEventRepository):
    def __init__(self):
        self.events = []

    def save(self, event: CrossingEvent) -> None:
        self.events.append(event)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
