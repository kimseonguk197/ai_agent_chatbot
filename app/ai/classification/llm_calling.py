import os
from dotenv import load_dotenv
from openai import OpenAI
from .classification_list import TOOLS

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def classify_message(message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": message}],
        tools=TOOLS,
        tool_choice="auto",
    )
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        return "응답할수 없는 질문입니다."
    return tool_calls[0].function.name


def generate_response(user_message: str, data: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "사용자의 질문에 대해 아래 참고 데이터를 바탕으로 사용자의 질문에 핵심만 짧게 답변해. 만약 참고데이터에 적절한 내용이 없으면 응답불가합니다 라고 답변해.",
            },
            {
                "role": "user",
                "content": f"{data}",
            },
        ],
    )
    return response.choices[0].message.content.strip()

