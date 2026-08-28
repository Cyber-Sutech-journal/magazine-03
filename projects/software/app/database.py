from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import declarative_base

engine=create_engine("sqlite:///./cinema.db")
SessionLocal=sessionmaker(bind=engine)
Base=declarative_base()

def get_db():
    session = SessionLocal()
    try:
      yield session
    finally:
       session.close()
    