import requests

VALID_LABELS = ["get_my_orders", "get_my_profile", "get_policy"]


def sllm_classifier(user_message: str) -> str:
    prompt = (
        "아래 질문을 읽고 반드시 다음 네 가지 중 하나만 출력해. 다른 말은 절대 하지 마.\n"
        "- get_my_orders\n"
        "- get_my_profile\n"
        "- get_policy\n"
        "- 응답불가합니다\n\n"
        f"질문: {user_message}\n"
        "출력:"
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2:3b", "prompt": prompt, "stream": False},
        timeout=300,
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()

    # 후처리: 유효한 레이블 포함 여부 확인
    for label in VALID_LABELS:
        if label in raw:
            return label

    return "응답불가합니다"
