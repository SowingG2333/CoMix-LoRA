import os
import json
import random
from datasets import load_dataset

# 配置
NUM_SUBSETS = 3
SAMPLES_PER_SUBSET = 1000  # 每个LoRA训练1000条
OVERLAP_RATIO = 0.5        # 50% 的重叠率
OUTPUT_DIR = "/root/gpufree-data/lapped-lora/data"

def prepare_data():
    # 1. 下载 GSM8K
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    all_data = list(dataset)
    random.shuffle(all_data) # 打乱顺序

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 构造重叠切分
    # 逻辑：滑动窗口。
    # Subset 0: [0 : 1000]
    # Subset 1: [500 : 1500] (前500条与Subset 0重复)
    # Subset 2: [1000 : 2000] (前500条与Subset 1重复)
    
    step = int(SAMPLES_PER_SUBSET * (1 - OVERLAP_RATIO))
    
    for i in range(NUM_SUBSETS):
        start_idx = i * step
        end_idx = start_idx + SAMPLES_PER_SUBSET
        
        subset_data = all_data[start_idx:end_idx]
        
        # 3. 格式化为 Qwen Base 喜欢的 SFT 格式，因为是 Base Model，我们不需要 Chat Template，直接用补全格式
        formatted_data = []
        for item in subset_data:
            # 简单的 Question-Answer 格式
            text = f"Question: {item['question']}\nAnswer: {item['answer']}<|endoftext|>"
            formatted_data.append({"text": text})
            
        # 保存
        filename = f"{OUTPUT_DIR}/subset_{i}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created {filename} with {len(formatted_data)} samples. (Indices: {start_idx}-{end_idx})")

if __name__ == "__main__":
    prepare_data()