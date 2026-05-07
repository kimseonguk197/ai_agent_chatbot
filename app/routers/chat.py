from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app import models, schemas
from app.models import RoleEnum
from app.dependencies import get_db, get_current_member
from app.ai import classify_message

router = APIRouter(prefix="/chats", tags=["chat"])


@router.post("", response_model=schemas.ChatResponse, status_code=status.HTTP_201_CREATED)
def create_chat(
    body: schemas.ChatRequest,
    db: Session = Depends(get_db),
    current_member: models.Member = Depends(get_current_member),
):
    action = classify_message(body.message)

    if action == "make_document" and current_member.role != RoleEnum.employee:
        action = "요청하실수 없는 작업입니다."

    chat_record = models.Chat(
        member_id=current_member.id,
        request=body.message,
        response=action,
    )
    db.add(chat_record)
    db.commit()
    db.refresh(chat_record)
    return chat_record
