import os
import gc
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# 0. 기본 환경 설정
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DATA_PATH = "train_classification.jsonl"

ADAPTER_DIR = "./llama32-3b-style-lora"
MERGED_DIR = "./llama32-3b-style-merged"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("[WARN] GPU를 찾을 수 없거나 드라이버 문제로 인해 CPU 모드로 실행합니다.")
# 모델 가중치를 메모리에 저장할 때 쓰는 데이터타입. 
# GPU는 bfloat16 연산을 하드웨어에서 직접 지원하기 때문에 빠르고 메모리도 절약
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
# CPU환경일때, 스레드 수 제한
# CPU가 1코어면 min(8, 1) → 1이 되는 거고, 만약 16코어면 min(8, 16) → 8로 제한
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


# 1. HF 로그인
def hf_login():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("[OK] Hugging Face login successful.")
    else:
        print("[WARN] failed")

# 2. 모델 / 토크나이저 로드 : 텍스트를 모델이 이해할 수 있는 숫자(토큰 ID)로 변환하는 도구
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # pad_token 미설정 방지
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


# 3. 기본 모델 불러오기
def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        low_cpu_mem_usage=True
    )
    model.to(DEVICE)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    return model

# 4. 데이터 전처리
def load_and_prepare_dataset(tokenizer):
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]

    # 원본 컬럼(instruction, response) 제거, text 컬럼만 유지
    dataset = dataset.map(
        lambda x: build_text(x, tokenizer),
        remove_columns=dataset.column_names
    )
    return dataset
# 4. 데이터 전처리
def build_text(example, tokenizer):
    # 기존 instruction / response 형식에  새로운 text 컬럼을 추가
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}



# 5. LoRA 학습
def train_lora():
    print("[INFO] Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("[INFO] Loading base model...")
    model = load_base_model()

    print("[INFO] Loading dataset...")
    dataset = load_and_prepare_dataset(tokenizer)

    # 어디(레이어)를 학습시킬지에 대한 설정
    peft_config = LoraConfig(
        r=32,               
        lora_alpha=128,     
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 얼마나, 어떤 방식으로 학습할지
    sft_config = SFTConfig(
        output_dir=ADAPTER_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_length=512, #입력데이터 최대 길이
        fp16=False,
        bf16=True,   # A4500 (Ampere 아키텍처)은 bf16 지원
        dataloader_num_workers=4,   # 리눅스/맥, 윈도우는 보통 0
        packing=False,              # 대규모 데이터 & pad가 있을땐 true 설정 
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="no",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print(" Training started...")
    trainer.train()

    print(" Saving LoRA adapter...")
    # LoRA 어댑터 가중치 저장
    trainer.model.save_pretrained(ADAPTER_DIR)
    # 토크나이저 설정 저장 : 어뎁터를 다시 불러다 쓸때 같은 토크나이저 필요
    tokenizer.save_pretrained(ADAPTER_DIR)

    print(f"[OK] adapter saved to: {ADAPTER_DIR}")

    # 메모리 정리
    del trainer
    del model
    gc.collect()


# 6. LoRA 병합
def merge_lora_to_base():
    print("Loading base model for merge...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        low_cpu_mem_usage=True
    )
    base_model.to(DEVICE)

    print("Loading trained adapter...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR
    )

    print("Merging adapter into base model...")
    merged_model = peft_model.merge_and_unload()

    os.makedirs(MERGED_DIR, exist_ok=True)

    print("Saving merged model...")
    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)

    # 병합 폴더 안에 tokenizer도 저장. 이후 해당 모델 실행시킬때 같은 토크나이저 사용
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(MERGED_DIR)

    # 메모리 정리
    del base_model
    del peft_model
    del merged_model
    gc.collect()


# 6. 간단 테스트
def test_merged_model(prompt: str):
    print("[INFO] Loading merged model for inference...")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_DIR,
        dtype=DTYPE,
        low_cpu_mem_usage=True
    )
    model.to(DEVICE)
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n[TEST OUTPUT]")
    print(result)


# 0. 메인
def main():
    hf_login()
    train_lora()
    merge_lora_to_base()
    test_merged_model("배송비 얼마야?")

if __name__ == "__main__":
    main()