import os
import uuid

import redis
import numpy as np
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 기본 TTL: 24시간. 캐시 히트 시마다 갱신
CACHE_TTL = int(os.getenv("SEMANTIC_CACHE_TTL", 86400))

# 정책 Q&A는 잘못된 캐시 반환 방지를 위해 RAG(0.4)보다 훨씬 높게 설정
CACHE_THRESHOLD = 0.9

# Redis 키 네임스페이스: 모든 캐시 항목은 이 prefix로 저장되고, 그 뒤에 적절한 UUID부여
# 예시) semantic_cache:abc-123 : { question: "배송 정책이 뭐야?", response: "...", question_embedding: [0.1, 0.2, ...] }
CACHE_PREFIX = "semantic_cache:"

# RediSearch 벡터 인덱스 이름
# 예시)아래 데이터들을 idx:semantic_cache라는 이름의 인덱스(카테고리)로 묶어 더 빠르게 벡터 검색
# semantic_cache:abc-123 의 { ..., question_embedding: [0.1, 0.2, ...] }
# semantic_cache:def-456 의 { ..., question_embedding: [0.1, 0.2, ...] }
# semantic_cache:ghi-789 의 { ..., question_embedding: [0.1, 0.2, ...] }
INDEX_NAME = "idx:semantic_cache"

# text-embedding-3-large 차원 (vector_store.py와 동일 모델 사용)
VECTOR_DIM = 3072

# Redis Stack 기반 Semantic Cache 기능
class SemanticCache:
    # 순서
    # 1. get_policy 관련 질문을 임베딩 벡터로 변환
    # 2. Redis에서 유사한 질문 벡터 검색
    # 3. 유사도 >= CACHE_THRESHOLD 이면 저장된 응답 반환(캐시 HIT)
    # 4. 캐시 히트 시 TTL 갱신 (Sliding TTL)

    def __init__(self):
        self._available = False
        try:
            self.r = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=False,
            )
            self.r.ping()  # 연결확인
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",  # vector_store.py와 동일 모델
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            self._ensure_index()
            self._available = True
            print("[SemanticCache] Redis 연결 성공")
        except Exception as e:
            # Redis 미실행 시에도 서버가 정상 동작하도록 graceful degradation
            print(f"[SemanticCache] Redis 연결 실패 → 캐시 없이 동작: {e}")

    def search(self, question: str) -> str | None:
        # redis 연결 실패시 종료
        if not self._available:
            return None

        try:
            query_embedding = self._embed(question)

            # 아래 쿼리는 Redis Stack(RediSearch)의 고유 문법
            q = (
                Query("*=>[KNN 1 @question_embedding $vec AS score]")
                .sort_by("score")
                .return_fields("response", "score")
                .dialect(2) #Redis Query 언어의 버전 지정
            )
            results = self.r.ft(INDEX_NAME).search(
                q, query_params={"vec": query_embedding}
            )

            if not results.docs:
                return None

            doc = results.docs[0]
            # Redis COSINE distance는 거리를 의미하므로, similarity의 반대의미
            distance = float(doc.score)
            similarity = 1 - distance

            print(f"[SemanticCache] 유사도: {similarity:.4f}")

            if similarity < CACHE_THRESHOLD:
                print("[SemanticCache] 미스")
                return None

            # ttl갱신
            self.r.expire(doc.id, CACHE_TTL)
            response = doc.response
            # 문자열로 변환
            if isinstance(response, bytes):
                response = response.decode("utf-8")
            return response

        except Exception as e:
            print(f"[SemanticCache] 검색 오류: {e}")
            return None

    def _embed(self, text: str) -> bytes:
        # 질문 텍스트 → 임베딩 벡터, self.embeddings : init에서 생성한 변수
        vector = self.embeddings.embed_query(text)
        # numpy 배열로 변환후 바이트로 리턴
        return np.array(vector, dtype=np.float32).tobytes()
    
    def store(self, question: str, response: str) -> None:
        if not self._available:
            return
        try:
            key = f"{CACHE_PREFIX}{uuid.uuid4()}"
            embedding = self._embed(question)

            self.r.hset(
                key,
                mapping={
                    "question": question.encode("utf-8"),
                    "response": response.encode("utf-8"),
                    "question_embedding": embedding,
                },
            )
            self.r.expire(key, CACHE_TTL)
            print(f"[SemanticCache] 저장 완료 (TTL: {CACHE_TTL}초): {question[:40]}...")

        except Exception as e:
            print(f"[SemanticCache] 저장 오류: {e}")


    def _ensure_index(self):
        # 벡터 검색 인덱스 생성 (이미 존재하면 스킵)
        try:
            self.r.ft(INDEX_NAME).info()
        except Exception:
            self.r.ft(INDEX_NAME).create_index(
                fields=[
                    TextField("question"),
                    TextField("response"),
                    # FLAT: 소규모 데이터에 적합한 전수 검색 방식
                    VectorField(
                        "question_embedding",
                        "FLAT",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": VECTOR_DIM,
                            "DISTANCE_METRIC": "COSINE",
                        },
                    ),
                ],
                # 아래 CACHE_PREFIX 이 프레픽스로 저장이 발생하면 현재 index사용
                definition=IndexDefinition(
                    prefix=[CACHE_PREFIX],
                    index_type=IndexType.HASH,
                ),
            )
            print(f"[SemanticCache] 인덱스 생성 완료: {INDEX_NAME}")


# 모듈 로드 시 1회 초기화
semantic_cache = SemanticCache()
