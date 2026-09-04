from fastapi import FastAPI

from app.database import Base, engine
from app.routers import auth, movies, bookings

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Cinema Booking API")

app.include_router(auth.router)
app.include_router(movies.router)
app.include_router(bookings.router)


@app.get("/")
def root():
    return {"message": "Cinema Booking API is running"}