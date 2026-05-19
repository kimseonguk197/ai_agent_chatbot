from sqlalchemy import text
from langchain_community.retrievers import BM25Retriever

from app.database import engine
from .vector_store import vector_store

# 1에 가까울수록 엄격한 필터링
SIMILARITY_THRESHOLD = 0.4

def search_policy(query: str) -> str | None:
    # 방법1. Dense Search (의미 기반 벡터 검색)
    # query를 임베딩으로 변환한 뒤 pgvector에서 유사한 문서를 검색. 최대 3개까지만 가져옴(top_k)
    # 장점: 의미적으로 유사한 문서를 찾음. "반품 어떻게 해?"로 검색해도 "환불 정책"을 찾아냄
    # 단점: 정확한 키워드가 있어도 벡터 거리가 멀면 놓칠 수 있음
    results = vector_store.similarity_search_with_relevance_scores(query, k=3)
    print(results)
    for doc, score in results:
        print(f"{score:.4f} | {doc.page_content}")
    relevant = [doc.page_content for doc, score in results if score >= SIMILARITY_THRESHOLD]
    print(relevant)
    if not relevant:
        return None
    return "\n".join(relevant)

    # 방법2. Sparse Search - BM25 (키워드 기반)
    # "환불 30일" 검색 → "환불", "30일" 단어로 문서를 TF-IDF 점수로 랭킹
    # 장점: 정확한 단어가 있을 때 정확도 향상. 고유명사, 날짜, 상품코드 등에 강함
    # 단점: "반품"으로 검색하면 "환불"이 있는 문서를 못 찾음

    # # 방법3. Hybrid Search (Dense + Sparse 결합)
    # # RRF(Reciprocal Rank Fusion) 알고리즘으로 두 방식의 순위를 결합
    # # 장점: 의미 기반 + 키워드 기반의 단점을 상호 보완. 프로덕션 환경에서 가장 권장
    # # 단점: 구현 복잡도가 올라가고 BM25 인메모리 유지 비용 상승

    # # BM25용으로 DB에서 전체 문서 텍스트 조회
    # all_texts = _get_all_texts()
    # if not all_texts:
    #     return None

    # # Sparse retriever (키워드 기반)
    # # "환불은 언제 가능해?"라는 질문을 ["환불", "언제", "가능"] 등으로 분해
    # # DB에 저장된 "환불은 구매 후 30일 이내에 가능합니다." 문장을 ["환불", "구매", "30일", "가능"] 등으로 분해
    # bm25_retriever = _get_bm25_retriever()
    # if bm25_retriever is None:
    #     return None
    # bm25_results = bm25_retriever.invoke(query)

    # # Dense retriever (벡터 기반) - 유사도 기반 문장 3개 추출 후 임계값 이상만 필터링
    # dense_results_with_scores = vector_store.similarity_search_with_relevance_scores(query, k=3)
    # dense_results = [doc for doc, score in dense_results_with_scores if score >= SIMILARITY_THRESHOLD]

    # # RRF로 BM25(최대3개) + Dense(최대3개) 결과를 합산 후 상위 3~6개 반환
    # fused = _reciprocal_rank_fusion([bm25_results, dense_results])
    # print(fused)
    # relevant = [item["doc"].page_content for item in fused[:3]]

    # if not relevant:
    #     return None
    # return "\n".join(relevant)



def _reciprocal_rank_fusion(results_lists: list, k: int = 60) -> list:
    # RRF 알고리즘: 각 retriever의 순위를 1/(rank+k) 점수로 변환 후 합산
    # 예를 들어)dense를 통한 top3가 문서A,D,C BM25를 통한 문서순위가 B,A,C 이렇다면,
    
    # k=60은 RRF 논문에서 권장하는 기본값 (상위 랭크에 과도한 가중치 방지)
    scores: dict = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content
            if key not in scores:
                scores[key] = {"score": 0.0, "doc": doc}
            scores[key]["score"] += 1 / (rank + k)
    return sorted(scores.values(), key=lambda x: x["score"], reverse=True)


def _get_all_texts() -> list[str]:
    # BM25를 위해 langchain_pg_embedding 테이블에서 전체 문서 텍스트 조회
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT document FROM langchain_pg_embedding"))
        # ["환불 정책 텍스트...", "배송 정책 텍스트...", ...] 같은 문자열 리스트를 반환
        return [row[0] for row in rows]


# BM25 알고리즘 최적화를 위한 코드
_bm25_cache: BM25Retriever | None = None
def _get_bm25_retriever() -> BM25Retriever | None:
    # 전체 문서 로드가 반복되지 않도록, 최초 1회만 빌드되도록 설정
    global _bm25_cache
    if _bm25_cache is None:
        all_texts = _get_all_texts()
        if not all_texts:
            return None
        _bm25_cache = BM25Retriever.from_texts(all_texts)
        _bm25_cache.k = 3
    return _bm25_cache


def invalidate_bm25_cache():
    # /documents 엔드포인트에서 새 문서 추가 BM25 재빌드
    global _bm25_cache
    _bm25_cache = None

