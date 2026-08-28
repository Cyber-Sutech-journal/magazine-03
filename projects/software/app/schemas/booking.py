from enum import Enum
from pydantic import BaseModel
class SeatStatus(str, Enum):
    available = "available"
    booked = "booked"


class SeatOut(BaseModel):

    id: int
    seat_number: str
    status: SeatStatus

    class Config:
        from_attributes = True


class BookingCreate(BaseModel):
    showtime_id: int
    seat_id: int


class BookingOut(BaseModel):

    id: int
    user_id: int
    showtime_id: int
    seat_id: int
    status: str

    class Config:
        from_attributes = True
