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
