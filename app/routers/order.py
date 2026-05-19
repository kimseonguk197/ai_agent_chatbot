from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import models, schemas
from app.dependencies import get_db, get_current_member
from app.ai.rag.semantic_cache import semantic_cache

router = APIRouter(prefix="/orders", tags=["orders"])


@router.post("", response_model=schemas.OrderResponse, status_code=status.HTTP_201_CREATED)
def create_order(
    body: schemas.OrderCreate,
    db: Session = Depends(get_db),
    current_member: models.Member = Depends(get_current_member),
):
    product = db.query(models.Product).filter(models.Product.id == body.product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    if product.stock < body.quantity:
        raise HTTPException(status_code=400, detail="Insufficient stock")

    product.stock -= body.quantity
    order = models.Order(member_id=current_member.id, product_id=body.product_id, quantity=body.quantity)
    db.add(order)
    db.commit()
    db.refresh(order)

    # 주문 발생 시 해당 사용자의 캐시만 삭제
    semantic_cache.flush_by_member(current_member.id)

    return order


@router.get("/me", response_model=List[schemas.OrderResponse])
def my_orders(
    db: Session = Depends(get_db),
    current_member: models.Member = Depends(get_current_member),
):
    return db.query(models.Order).filter(models.Order.member_id == current_member.id).all()
