import os
import json
import glob
import torch

# === 1. 强制设置环境变量 (解决 RTX 4090 NCCL 报错) ===
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer

# === 配置 ===
MODEL_ID = "/root/gpufree-data/hf/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
DATA_DIR = "/root/gpufree-data/lapped-lora/data" 
OUTPUT_DIR = "/root/gpufree-data/lapped-lora/model"

def merge_subsets_to_global():
    """将所有子数据集合并为一个全量数据集"""
    print("Merging subsets into global dataset...")
    all_texts = set()
    files = glob.glob(f"{DATA_DIR}/subset_*.json")
    
    # 排除已经生成的 global 文件，防止死循环
    files = [f for f in files if "global" not in f]
    
    if not files:
        raise FileNotFoundError(f"No subset_*.json files found in {DATA_DIR}")

    for fpath in files:
        with open(fpath, 'r') as f:
            data = json.load(f)
            for item in data:
                all_texts.add(item['text']) # 使用 set 自动去重
    
    merged_data = [{"text": t} for t in all_texts]
    output_path = f"{DATA_DIR}/subset_global.json"
    
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"Global dataset created at {output_path} with {len(merged_data)} unique samples.")
    return output_path

def train_global():
    # 1. 准备数据
    global_data_path = merge_subsets_to_global()
    
    print(f"\n=== Training GLOBAL LoRA ===")
    
    # 2. 量化配置 (BF16)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": 0}, 
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token 

    # 4. LoRA 配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"] 
    )
    model = get_peft_model(model, peft_config)

    # 5. 加载数据
    dataset = load_dataset("json", data_files=global_data_path, split="train")

    # 6. 训练参数 (BF16=True, FP16=False)
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/lora_global",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_strategy="no",
        fp16=False,   # 必须关闭
        bf16=True,    # 开启 BF16
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    def formatting_prompts_func(example):
        return example['text']

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
        args=training_args,
    )

    trainer.train()
    
    model.save_pretrained(f"{OUTPUT_DIR}/lora_global")
    print("Global LoRA saved successfully.")

if __name__ == "__main__":
    train_global()