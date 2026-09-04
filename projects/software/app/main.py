from fastapi import FastAPI
from app.routers import auth, movies


app = FastAPI()

app.include_router(auth.router)
app.include_router(movies.router)