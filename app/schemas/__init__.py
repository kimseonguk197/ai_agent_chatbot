from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr
from app.models import RoleEnum


# Member
class MemberCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None
    age: Optional[int] = None
    role: RoleEnum = RoleEnum.user


class MemberResponse(BaseModel):
    id: int
    email: str
    name: Optional[str] = None
    age: Optional[int] = None
    role: RoleEnum
    created_at: datetime

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str


# Product
class ProductCreate(BaseModel):
    name: str
    category: str
    price: float
    stock: int


class ProductResponse(BaseModel):
    id: int
    name: str
    category: str
    price: float
    stock: int
    member_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Order
class OrderCreate(BaseModel):
    product_id: int
    quantity: int


class OrderResponse(BaseModel):
    id: int
    member_id: int
    product_id: int
    quantity: int
    created_at: datetime

    class Config:
        from_attributes = True


# Chat
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    id: int
    member_id: int
    request: str
    response: str
    created_at: datetime

    class Config:
        from_attributes = True
