from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy.orm import Session

from app.models import Chat

# Window Memory 크기: 최근 N턴의 대화만 유지
# 너무 크면 토큰 비용 증가, 너무 작으면 문맥 손실
MEMORY_WINDOW = 5


def load_chat_history(member_id: int, db: Session, limit: int = MEMORY_WINDOW) -> list:

    # 테이블에서 특정 사용자의 최근 대화 기록을 LangChain 메시지 형식으로 반환
    # 최신순으로 limit개 가져온 뒤, 역순 정렬하여 시간 오름차순으로 변환
    recent_chats = (
        db.query(Chat)
        .filter(Chat.member_id == member_id)
        .order_by(Chat.created_at.desc())
        .limit(limit)
        .all()
    )

    messages = []
    # 반환값: [HumanMessage, AIMessage, HumanMessage, AIMessage] 형태의 리스트로 변환
    for chat in reversed(recent_chats):  # 앞의 5개 중 오래된 것부터
        # 명확한 응답을 위해 두 주체의 대화의 구분이 필요
        # HumanMessage : 사용자가 보낸 메시지, AIMessage : AI(LLM)가 응답했던 메시지
        messages.append(HumanMessage(content=chat.request))
        messages.append(AIMessage(content=chat.response))

    return messages
