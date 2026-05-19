import os
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

# langchain_postgres는 psycopg3 드라이버 사용 (postgresql+psycopg://)
_DATABASE_URL = os.getenv("DATABASE_URL", "").replace(
    "postgresql://", "postgresql+psycopg://"
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# PGVector객체가 만들어지는 시점에 아래 테이블2개가 자동으로 생성
# langchain_pg_collection(컬렉션 목록 테이블), langchain_pg_embedding(실제 데이터 테이블)  
# langchain_pg_embedding : id, collection_id, embedding(vector), document(text) 컬럼으로 구성
    
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="documents",
    connection=_DATABASE_URL,
    use_jsonb=True,
)
