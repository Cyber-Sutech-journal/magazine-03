from pydantic import BaseModel
from datetime import datetime

class MovieCreate(BaseModel):
    title: str
    duration: int

class MovieResponse(BaseModel):
    id: int
    title: str
    duration: int 

    class Config:
        from_attributes = True


class ShowtimeCreate(BaseModel):
    movie_id: int
    hall_id: int
    start_time: datetime

class ShowtimeResponse(BaseModel):
    id: int
    movie_id: int
    hall_id: int
    start_time: datetime

    class Config:
        from_attributes = True



class MovieOut(BaseModel):
    id: int
    title: str
    duration: int 
