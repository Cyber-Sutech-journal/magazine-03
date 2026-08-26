from sqlalchemy import Column,Integer,String
from app.database import Base

class Account(Base):
    tablename ="users"

    id=Column(Integer,primary_key=True,index=True)
    email=Column(String,unique=True,nullable=False,index=True)
    hashed_passwoard=Column(String,nullable=False)
    role=Column(String,default="user")
