from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import models, schemas
from app.services.auth import hash_password, verify_password, create_access_token
from app.dependencies import get_db, get_current_member

router = APIRouter(prefix="/members", tags=["members"])


@router.post("/signup", response_model=schemas.MemberResponse, status_code=status.HTTP_201_CREATED)
def signup(body: schemas.MemberCreate, db: Session = Depends(get_db)):
    if db.query(models.Member).filter(models.Member.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    member = models.Member(
        email=body.email,
        password=hash_password(body.password),
        name=body.name,
        age=body.age,
    )
    db.add(member)
    db.commit()
    db.refresh(member)
    return member


@router.post("/login", response_model=schemas.TokenResponse)
def login(body: schemas.LoginRequest, db: Session = Depends(get_db)):
    member = db.query(models.Member).filter(models.Member.email == body.email).first()
    if not member or not verify_password(body.password, member.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": str(member.id)})
    return {"access_token": token}


@router.get("/me", response_model=schemas.MemberResponse)
def my_page(current_member: models.Member = Depends(get_current_member)):
    return current_member
