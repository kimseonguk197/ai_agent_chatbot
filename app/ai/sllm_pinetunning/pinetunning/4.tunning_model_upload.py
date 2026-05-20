from huggingface_hub import HfApi
import os

api = HfApi()
HF_ID = "bradkim198"  # 허깅페이스 ID

# 1. LoRA 어댑터(Adapter) 업로드 - 백업 및 재학습용
lora_repo = f"{HF_ID}/llama32-3b-style-lora"
api.create_repo(repo_id=lora_repo, exist_ok=True)

print(f"[INFO] LoRA 어댑터 업로드 중: {lora_repo}")
api.upload_folder(
    folder_path="./llama32-3b-style-lora",
    repo_id=lora_repo,
    repo_type="model",
    ignore_patterns=["checkpoint-*", "runs/*"] # 불필요한 파일 제외
)

# 2. 병합 모델(Full Model) 업로드 - 로컬 CPU PC 테스트용
merged_repo = f"{HF_ID}/llama32-3b-style-merged"
api.create_repo(repo_id=merged_repo, exist_ok=True)

print(f"[INFO] 병합 모델 업로드 중: {merged_repo}")
api.upload_folder(
    folder_path="./llama32-3b-style-merged",
    repo_id=merged_repo,
    repo_type="model",
)

print("\n" + "="*30)
print(f"모든 업로드가 완료")