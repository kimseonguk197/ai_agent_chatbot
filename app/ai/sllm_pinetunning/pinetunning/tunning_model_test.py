import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

VALID_LABELS = ["get_my_orders", "get_my_profile", "get_policy"]
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MERGED_REPO_ID = "bradkim198/llama32-3b-style-merged"

# GPU/CPU 분기
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {DEVICE}")

# 모델 로드 (Merged Model)
print(f"로컬 Merged 모델 로딩 중 ({MERGED_REPO_ID})")
tokenizer = AutoTokenizer.from_pretrained(MERGED_REPO_ID)
model = AutoModelForCausalLM.from_pretrained(
    MERGED_REPO_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
    device_map={"": DEVICE},
)
model.eval()

# 병합(lora) 모델 답변 생성하기
def tunnig_classifier(user_message: str) -> str:
    prompt = (
        "아래 질문을 읽고 반드시 다음 네 가지 중 하나만 출력해. 다른 말은 절대 하지 마.\n"
        "- get_my_orders\n"
        "- get_my_profile\n"
        "- get_policy\n"
        "- 응답불가합니다\n\n"
        f"질문: {user_message}\n출력:"
    )
    messages = [{"role": "user", "content": prompt}]
    
    encoded = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(DEVICE)
    
    # 만약 encoded가 딕셔너리(BatchEncoding)라면 .input_ids를 쓰고, 아니면 그대로 사용
    input_tensor = encoded.input_ids if hasattr(encoded, "input_ids") else encoded

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_tensor, 
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    raw = tokenizer.decode(outputs[0][input_tensor.shape[-1]:], skip_special_tokens=True).strip()

    for label in VALID_LABELS:
        if label in raw:
            return label

    return "응답불가합니다"
