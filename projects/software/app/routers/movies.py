from fastapi import APIRouter, Depends, HTTPEException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User

from app.schemas.movie import (
    MovieCreate,
    MovieResponse,
    ShowtimeCreate,
    ShowtimeResponse
)

from app.auth.jwt_handler import get_current_user