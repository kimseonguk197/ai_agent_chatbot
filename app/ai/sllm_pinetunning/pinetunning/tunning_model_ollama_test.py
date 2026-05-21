import requests

VALID_LABELS = ["get_my_orders", "get_my_profile", "get_policy"]
OLLAMA_MODEL_NAME = "llama3-2-classifier"  # ollama create로 등록한 모델명


def tunnig_classifier(user_message: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL_NAME, "prompt": user_message, "stream": False},
        timeout=300,
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()

    for label in VALID_LABELS:
        if label in raw:
            return label

    return "응답불가합니다"
