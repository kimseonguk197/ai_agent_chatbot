from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app import models, schemas
from app.models import RoleEnum
from app.dependencies import get_db, get_current_member
from app.ai.classification.llm_calling import classify_message, generate_response
from app.ai.rag.retriever import search_policy
from app.ai.rag.llm_calling_langchain import generate_response_with_memory
from app.ai.rag.memory import load_chat_history
from app.routers.order import my_orders

from app.ai.rag.semantic_cache import semantic_cache

router = APIRouter(prefix="/chats", tags=["chat"])


def _format_orders(orders: list) -> str:
    if not orders:
        return "주문 내역이 없습니다."
    lines = [
        f"- 주문번호: {o.id} / 상품ID: {o.product_id} / 수량: {o.quantity} / 주문일: {o.created_at.strftime('%Y-%m-%d')}"
        for o in orders
    ]
    return "\n".join(lines)


@router.post("", response_model=schemas.ChatResponse, status_code=status.HTTP_201_CREATED)
def create_chat(
    body: schemas.ChatRequest,
    db: Session = Depends(get_db),
    current_member: models.Member = Depends(get_current_member),
):
    # Redis에서 유사한 질문의 응답을 검색 (LLM 호출 없이 즉시 반환)
    # 히트 시: 즉시 응답 및 ttl 갱신
    # 미스 시: 아래 분기 처리로 진행

    cached_response = semantic_cache.search(body.message)

    if cached_response:
        response_text = cached_response

    else:
        action = classify_message(body.message)

        if action == "get_my_orders":
            # 개인화 데이터: 캐시 저장 안 함 (사용자마다 다른 응답)
            orders = my_orders(db=db, current_member=current_member)
            data = _format_orders(orders)
            response_text = generate_response(body.message, data)

        elif action == "get_policy":
            context = search_policy(body.message)
            if context is None:
                response_text = "요청하실수 없는 작업입니다."
            else:
                # Window Memory: 최근 5턴 대화 기록을 함께 전달
                history = load_chat_history(current_member.id, db)
                response_text = generate_response_with_memory(body.message, context, history)

                # 정책 응답 Semantic Cache에 저장
                semantic_cache.store(body.message, response_text)

        elif action == "make_document" and current_member.role != RoleEnum.employee:
            response_text = "요청하실수 없는 작업입니다."

        else:
            response_text = action

    chat_record = models.Chat(
        member_id=current_member.id,
        request=body.message,
        response=response_text,
    )
    db.add(chat_record)
    db.commit()
    db.refresh(chat_record)
    return chat_record
