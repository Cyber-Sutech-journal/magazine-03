from pydantic import BaseModel

class UserCreate(BaseModel):
    email=str
    passwoard=str

class Userout(BaseModel):
    id=int
    email=str
    role=str

class Token(BaseModel):
    access_token=str
    token_type=str