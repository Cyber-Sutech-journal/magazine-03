from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from app.database import Base

class Movie(Base):
    __tablename__ = "movies"

    id=Column(Integer,primary_key=True,index=True)
    title=Column(String,nullable=False)
    duration=Column(Integer,nullable=False)

class Hall(Base):
    __tablename__ = "halls"

    id=Column(Integer,primary_key=True,index=True)
    name=Column(String,nullable=False)
    capacity=Column(Integer,nullable=False)

class Showtime(Base):
     __tablename__ = "showtime"

     id=Column(Integer,primary_key=True,index=True)
     movie_id=Column(Integer,ForeignKey("movies.id"),nullable=False)
     hall_id=Column(Integer,ForeignKey("halls.id"),nullable=False)
     start_time=Column(DateTime,nullable=False)
