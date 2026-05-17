from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ai.rag.retriever import invalidate_bm25_cache

from app import schemas
from app.dependencies import get_db
from app.ai.rag.vector_store import vector_store

router = APIRouter(prefix="/documents", tags=["documents"])

# 청킹 설정
# chunk_size: 청크 당 최대 글자 수
# chunk_overlap: 청크 간 겹치는 글자 수. 청크 경계에서 문맥이 잘리는 것을 방지
# separators = ["\n\n", "\n", ". ", " ", ""] 이런 매커니즘을 우선순위로 문장 split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)


@router.post("", response_model=schemas.DocumentResponse, status_code=status.HTTP_201_CREATED)
def add_documents(body: schemas.DocumentCreate, db: Session = Depends(get_db)):
    # 입력된 텍스트들을 청크 단위로 분할
    # RecursiveCharacterTextSplitter는 문단 → 문장 → 단어 순으로 자연스럽게 분할
    # chunks = splitter.create_documents(body.texts)
    # chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_texts = [chunk for chunk in body.texts]

    # LangChain vector store에 저장
    # add_texts()는 내부적으로 임베딩 생성 + langchain_pg_embedding 테이블 저장을 함께 처리
    vector_store.add_texts(chunk_texts)

    # 하이브리드 검색을 위한 코드. 새 문서 추가 시 BM25 캐시 무효화
    invalidate_bm25_cache()  
    return {"added": len(body.texts)}

