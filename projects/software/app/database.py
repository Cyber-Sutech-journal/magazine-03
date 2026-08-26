from sqlalchemy import create_engine
engine=create_engine("sqlite:///./cinema.db")
from sqlalchemy.orm import sessionmaker
sessionlocal=sessionmaker(bind=engine)
from sqlalchemy import declarative_base
Base=declarative_base()

def get_db():
    session=sessionlocal()
    try:
      yield session
    finally:
       session.close()
    