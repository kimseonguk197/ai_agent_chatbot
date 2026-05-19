import requests


def generate_response_sllm(user_message: str, data: str) -> str:
    prompt = (
        "사용자의 질문에 대해 아래 참고 데이터를 바탕으로 사용자의 질문에 답변해. "
        "만약 참고데이터에 적절한 내용이 없으면 응답불가합니다 라고 답변해.\n\n"
        f"[참고 데이터]\n{data}\n\n"
        f"[사용자 질문]\n{user_message}"
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2:3b", "prompt": prompt, "stream": False},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"]
