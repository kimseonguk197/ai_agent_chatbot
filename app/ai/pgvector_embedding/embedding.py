import os
from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector

# 0) 준비: 환경변수 및 연결 설정
# OpenAI API Key 확인
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다.")

client = OpenAI(api_key=openai_api_key)

# PostgreSQL 연결 정보 (로컬 환경에 맞게 수정하여 사용하세요)
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "your_database_name"
DB_USER = "your_username"
DB_PASSWORD = "your_password"

# 1) 임베딩 모델 설정 (OpenAI 1536차원 모델)
EMBEDDING_MODEL = "text-embedding-3-small"

# 2) 쇼핑몰 정책 문서
POLICY_TEXTS = [
    # refund
    "이 쇼핑몰의 환불 가능 기간은 상품 수령 후 7일 이내이다.",
    "환불 책임 부서는 고객지원팀이며 최종 환불 승인 책임자는 CS 운영 매니저이다.",
    "전자제품은 개봉 또는 사용 흔적이 있는 경우 환불이 제한될 수 있다.",
    "고객 책임으로 상품이 훼손되거나 포장이 심하게 손상된 경우 환불이 불가능하다.",
    "환불 검수 완료 후 결제 수단 기준으로 3영업일 이내 환불이 처리된다.",

    # exchange
    "교환은 상품 수령 후 7일 이내 신청 가능하다.",
    "사이즈 교환 시 왕복 배송비는 고객이 부담한다.",

    # shipping
    "기본 배송비는 3,000원이며 5만원 이상 구매 시 무료배송이다.",
    "도서산간 지역은 추가 배송비가 발생할 수 있다.",

    # coupon
    "쿠폰은 유효기간 내에만 사용할 수 있으며 중복 사용은 불가능하다.",

    # points
    "적립금은 구매 확정 후 3일 이내 지급된다.",

    # member
    "회원 등급은 최근 3개월 구매 금액을 기준으로 매월 1일 갱신된다.",
]

# 3) DB 테이블 초기화 및 데이터 삽입 함수
def init_db_and_insert():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cur = conn.cursor()
    
    # pgvector 확장 활성화 (이미 되어있다면 생략 가능)
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # 기존 테이블이 있다면 삭제하고 재생성 (실행 시 초기화 목적)
    cur.execute("DROP TABLE IF EXISTS shopping_policies;")
    cur.execute("""
        CREATE TABLE shopping_policies (
            id SERIAL PRIMARY KEY,
            topic TEXT,
            content TEXT,
            embedding vector(1536)
        );
    """)
    conn.commit()

    # 문서 임베딩 생성 및 저장
    # 텍스트를 파악하여 임의의 토픽(refund, exchange, shipping 등)을 할당하는 로직
    for text in POLICY_TEXTS:
        # 간단히 텍스트 내용에 따른 주제 분류 (또는 전체를 하나의 카테고리로 묶을 수도 있습니다)
        if "환불" in text:
            topic = "refund"
        elif "교환" in text:
            topic = "exchange"
        elif "배송" in text:
            topic = "shipping"
        elif "쿠폰" in text:
            topic = "coupon"
        elif "적립금" in text:
            topic = "points"
        else:
            topic = "member"

        # OpenAI 임베딩 API 호출
        response = client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding

        cur.execute(
            "INSERT INTO shopping_policies (topic, content, embedding) VALUES (%s, %s, %s)",
            (topic, text, embedding)
        )
    
    conn.commit()
    cur.close()
    conn.close()
    print("DB 초기화 및 데이터 입력이 완료되었습니다.")

# 4) Query 함수
def run_query(question: str, top_k: int = 3, topic_filter: str | None = None):
    # 질문에 대한 임베딩 생성
    response = client.embeddings.create(
        input=[question],
        model=EMBEDDING_MODEL
    )
    q_embedding = response.data[0].embedding

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    register_vector(conn)
    cur = conn.cursor()

    # 조건에 따른 쿼리 구성
    if topic_filter:
        cur.execute(
            """
            SELECT id, topic, content, 1 - (embedding <=> %s) AS cosine_similarity 
            FROM shopping_policies 
            WHERE topic = %s 
            ORDER BY embedding <=> %s 
            LIMIT %s;
            """,
            (q_embedding, topic_filter, q_embedding, top_k)
        )
    else:
        cur.execute(
            """
            SELECT id, topic, content, 1 - (embedding <=> %s) AS cosine_similarity 
            FROM shopping_policies 
            ORDER BY embedding <=> %s 
            LIMIT %s;
            """,
            (q_embedding, q_embedding, top_k)
        )
    
    results = cur.fetchall()
    cur.close()
    conn.close()

    return results

# 실행 및 테스트
if __name__ == "__main__":
    # 1. DB 초기화 및 데이터 적재 실행
    init_db_and_insert()

    # 2. 대조 질문 테스트
    questions = [
        "환불을 받으려면 어떻게 해야 하나요?",
        "배송비 규정이 어떻게 되나요?",
    ]

    for q in questions:
        print(f"\n== Query: {q} ==")
        results = run_query(q, top_k=3)
        for row in results:
            policy_id, topic, content, score = row
            print(f"- id={policy_id} score={score:.4f} topic={topic} text={content}")

    # 3. 필터 테스트 (대조 실험)
    print("\n== Query '환불을 받으려면 어떻게 해야 하나요?' with topic=shipping filter ==")
    results = run_query("환불을 받으려면 어떻게 해야 하나요?", top_k=3, topic_filter="shipping")
    for row in results:
        policy_id, topic, content, score = row
        print(f"- id={policy_id} score={score:.4f} topic={topic} text={content}")