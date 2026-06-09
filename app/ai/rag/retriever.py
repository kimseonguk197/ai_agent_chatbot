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
    for doc, score in results:
        print(f"{score:.4f} | {doc.page_content}")
    relevant = [doc.page_content for doc, score in results if score >= SIMILARITY_THRESHOLD]
    print(relevant)
    if not relevant:
        return None
    return "\n".join(relevant)

    # 방법2. Sparse Search - BM25 (키워드 기반)
    # 예시)"환불은 언제 가능해?"라는 질문을 ["환불", "언제", "가능"] 등으로 분해
    # DB에 저장된 "환불은 구매 후 30일 이내에 가능합니다." 문장을 ["환불", "구매", "30일", "가능"] 등으로 분해하여 단어마나 빈도수 점수화
    # 장점: 정확한 단어가 있을 때 정확도 향상. 고유명사, 날짜, 상품코드 등에 강함
    # 단점: "반품"으로 검색하면 "환불"이 있는 문서를 못 찾음

    # # 방법3. Hybrid Search (Dense + Sparse 결합)
    # # RRF(Reciprocal Rank Fusion) 알고리즘으로 두 방식의 순위를 결합
    # # 장점: 의미 기반 + 키워드 기반의 단점을 상호 보완. 프로덕션 환경(대규모코퍼스)에서 권장
    # # 단점: 구현 복잡도

    # 방법3-1. BM25 - 키워드 기반 문장 3개 추출
    # bm25_results = _get_bm25_results(query)

    # 방법3-2. Dense retriever  - 유사도 기반 문장 3개 추출 후 임계값 이상만 필터링
    # dense_results_with_scores = vector_store.similarity_search_with_relevance_scores(query, k=3)
    # dense_results = [doc for doc, score in dense_results_with_scores if score >= SIMILARITY_THRESHOLD]

    # 방법3-3. RRF로 BM25(최대3개) + Dense(최대3개) 결과를 합산 후 rerank
    # fused = _reciprocal_rank_fusion([bm25_results, dense_results])
    # print(fused)
    # relevant = [item["doc"].page_content for item in fused[:3]]

    # if not relevant:
    #     return None
    # return "\n".join(relevant)


# BM25 알고리즘
def _get_bm25_results(query: str) -> list:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT document FROM langchain_pg_embedding"))
        all_texts = [row[0] for row in rows]
    if not all_texts:
        return []
    retriever = BM25Retriever.from_texts(all_texts)
    retriever.k = 3
    return retriever.invoke(query)



def _reciprocal_rank_fusion(results_lists: list, k: int = 60) -> list:
    # RRF 알고리즘: 각 retriever의 순위를 1/(rank+k) 점수로 변환 후 합산
    # k=60은 RRF 논문에서 권장하는 기본값
    scores: dict = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content
            if key not in scores:
                scores[key] = {"score": 0.0, "doc": doc}
            scores[key]["score"] += 1 / (rank + k)
    return sorted(scores.values(), key=lambda x: x["score"], reverse=True)


