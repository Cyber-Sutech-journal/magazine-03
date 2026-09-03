from pydantic import BaseModel

class MovieCreate(BaseModel):
    title: str
    duration: int

class MovieOut(BaseModel):
    id: int
    title: str
    duration: int 
