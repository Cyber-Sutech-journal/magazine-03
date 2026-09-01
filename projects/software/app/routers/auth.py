from fastapi import APIRouter, Depends, HTTPException, status
from app.auth.jwt_handler import (hash_password, verify_password, create_access_token)
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserLogin 

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

@router.post("/register" , status_code=status.HTTP_201_CREATED)
def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    existing_user = db.query(User).filter(
        User.email == user_data.email
    ).first()

if existing_user:
    raise HTTPException(
        status_code = status.HTTP_400_BAD_REQUEST,
        detail = "Email already registered"
    )

hashed_password = hash_password(user_data.password)
new_user = User(
    email=user_data.email, 
    hashed_password = hashed_password,
    role="user"
)

db.add(new_user)
db.commit()
db.refresh(new_user)

return {
     "id" : new_user.id,
     "email" : new_user.email,
     "role" : new_user.role
}

@router.post("/login")
def login(
   user_data : UserLogin,
   db : Session = Depends(get_db)
):
   user = db.query(User).filter(
      User.email == user_data.email
   ).first()

if not user or not verify_password(
      user_data.password,
      user.hashed_password
   ):
      raise HTTPException(
         status_code = status.HTTP_401_UNAUTHORIZED,
         detail = "Invalid email or password"
      )

access_token = create_access_token(
   data={"sub" : user.email}
)

return {
   "access_token : access_token",
   "token_type" : "bearer"
}