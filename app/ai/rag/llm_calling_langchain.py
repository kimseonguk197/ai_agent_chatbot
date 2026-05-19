import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from app.ai.classification.classification_list import TOOLS

llm_classify = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

llm_response = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3  
)

# langchain에서는 아래와 같이 모델만 변경하면 쉽게 코드 리팩토링 가능 
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
# llm = ChatAnthropic(model="claude-opus-4-5")
# 모델이 바뀌어도 bind_tools(), invoke(), StrOutputParser() 등 나머지 코드는 그대로 사용 가능
llm_with_tools = llm_classify.bind_tools(TOOLS)


def classify_message_langchain(message: str) -> str:
    response = llm_with_tools.invoke(message)
    tool_calls = response.tool_calls
    if not tool_calls:
        return "답변이 어려운 질문입니다."
    return tool_calls[0]["name"]


def generate_response_langchain(user_message: str, data: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "사용자의 질문에 대해 아래 참고 데이터를 바탕으로 사용자의 질문에 답변해. 만약 참고데이터에 적절한 내용이 없으면 응답불가합니다 라고 답변해. \n\n[참고 데이터]\n{data}"),
        ("user", "{user_message}"),
    ])
    #  LCEL(LangChain Expression Language) 
    #  파이프 연산자로 컴포넌트를 연결하는 방식
    chain = prompt | llm_response | StrOutputParser()
    return chain.invoke({"data": data, "user_message": user_message})


# history(메모리)와 함께 llm응답 생성
def generate_response_langchain_memory(user_message: str, data: str, history: list = None) -> str:
    print(f"generate_response memory back data {history}")
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "사용자의 질문에 대해 아래 참고 데이터를 바탕으로 사용자의 질문에 핵심만 짧게 답변해. "
            "이전 대화에서 답할 수 있는 내용이 있으면 그것을 우선으로 사용해."
            "참고 데이터와 이전 대화 모두에서 관련 내용을 찾을 수 없는 경우에만 응답불가합니다 라고 답변해.\n\n"
            "[참고 데이터]\n{data}"
        ),
        # history : 이전 대화 기록을 system과 현재 질문 사이에 삽입
        # load_chat_history()로 가져온 HumanMessage/AIMessage 리스트 데이터
        MessagesPlaceholder(variable_name="history"),
        ("user", "{user_message}"),
    ])
    chain = prompt | llm_response | StrOutputParser()
    return chain.invoke({
        "data": data or "",  # None이면 빈 문자열로 폴백
        "user_message": user_message,
        "history": history or [],  # None이면 빈 리스트로 폴백
    })


