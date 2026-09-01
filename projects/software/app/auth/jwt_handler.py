from datetime import datetime, timedelta, timezone
import jwt 
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash
from sqlalchemy.orm import  Session

from app.database import get_db
import app.models.user 
import User 

SECRET_KEY = "change_this_to_a_long_random_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

password_hash = PasswordHash.recommended()
oauth2_scheme = OAuthPasswordBearer(tokenUrl="/auth/login")