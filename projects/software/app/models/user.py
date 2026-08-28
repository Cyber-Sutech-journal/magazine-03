from sqlalchemy import Column, Integer, String
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id=Column(Integer,primary_key=True,index=True)
    email=Column(String,unique=True,nullable=False,index=True)
    hashed_password=Column(String,nullable=False)
    role=Column(String,default="user")
