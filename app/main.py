from fastapi import FastAPI
from app.database import engine, Base
from app.routers import member, product, order, chat
from app import models

Base.metadata.drop_all(bind=engine)   # 기존 테이블 전부 삭제
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Agent API")

app.include_router(member.router)
app.include_router(product.router)
app.include_router(order.router)
app.include_router(chat.router)
