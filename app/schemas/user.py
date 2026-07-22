from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import Optional


class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    role: Optional[str] = None


class UserRead(BaseModel):
    id: int
    email: EmailStr
    username: str
    role: str
    is_active: bool
    created_at: datetime
    status: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
