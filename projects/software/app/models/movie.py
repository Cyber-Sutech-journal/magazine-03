from sqlalchemy import Column,Integer,String,ForeignKey,DateTime
from app.database import Base

class Movie(Base):
    tablename="movies"

    id=Column(Integer,primary_key=True,index=True)
    title=Column(String,nullable=False)
    duration=Column(Integer,nullable=False)

class Hall(Base):
    tablename="halls"

    id=Column(Integer,primary_key=True,index=True)
    name=Column(String,nullable=False)
    capacity=Column(Integer,nullable=False)

class Showtime(Base):

    id=Column(Integer,primary_key=True,index=True)
    movie_id=Column(Integer,ForeignKey("movie_id"),nullable=False)
    hall_id=Column(Integer,ForeignKey("halls_id"),nullable=False)
    start_time=Column(DateTime,nullable=False)
