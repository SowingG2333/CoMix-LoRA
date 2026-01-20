import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# === 1. 导入你的自定义模型 ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_router import CoMixModel

# === 2. 配置路径与参数 ===
BASE_MODEL_PATH = "/root/gpufree-data/hf/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
EXPERT_DIR = "/root/gpufree-data/OverlappedLoRA/model"
ROUTER_CHECKPOINT = "/root/gpufree-data/OverlappedLoRA/scripts/output/comix_router/checkpoint-100/routers.pt"

TEST_PROMPT = "Question: Janet has 3 times as many marbles as Arnold. If Arnold has 12 marbles, how many marbles do they have together?\nAnswer:"
TARGET_TEXT = " 48"
NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
N_SAMPLES = 20
NUM_EXPERTS = 8

# === 3. 辅助类：强制路由 ===
class ForcedRouter(torch.nn.Module):
    """一个假的 Router，永远只返回固定的 One-Hot 分布"""
    def __init__(self, expert_idx, num_experts):
        super().__init__()
        self.expert_idx = expert_idx
        self.num_experts = num_experts

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        probs = torch.zeros(batch_size, seq_len, self.num_experts, 
                          device=hidden_states.device, dtype=hidden_states.dtype)
        probs[:, :, self.expert_idx] = 1.0
        return probs

def measure_performance(model, tokenizer, input_ids, target_id, n_samples):
    logit_sum = 0
    rank_sum = 0
    
    with torch.no_grad():
        _ = model(input_ids) # Warmup

    for _ in range(n_samples):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            
            target_logit = logits[target_id].item()
            logit_sum += target_logit
            
            # Rank 1 is best
            rank = (logits > target_logit).sum().item() + 1
            rank_sum += rank

    return logit_sum / n_samples, rank_sum / n_samples

def main():
    print(">>> [1/4] Loading Model & Experts (BF16)...")
    # 初始化模型，调大 Alpha 以增强信号
    model = CoMixModel(
        base_model_path=BASE_MODEL_PATH,
        num_experts=NUM_EXPERTS,
        r=16, 
        alpha=64, 
        device="cuda"
    )
    model.load_experts(EXPERT_DIR)
    
    # === [关键设置] 显式激活 DP 裁剪 ===
    model.clip_threshold = 1.0 
    print(f">>> [DP Config] Global Clip Threshold set to {model.clip_threshold}")

    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    input_ids = tokenizer(TEST_PROMPT, return_tensors="pt")["input_ids"].to("cuda")
    target_id = tokenizer.encode(TARGET_TEXT, add_special_tokens=False)[0]
    
    # 加载训练好的 Router
    print(f">>> Loading Trained Router from {ROUTER_CHECKPOINT}...")
    if os.path.exists(ROUTER_CHECKPOINT):
        trained_router_state = torch.load(ROUTER_CHECKPOINT, map_location="cuda")
        model.routers.load_state_dict(trained_router_state)
    else:
        print("!!! Warning: Router checkpoint not found.")

    # === [关键修复] 保存原始 Router 对象的引用，而不是它的副本 ===
    # 我们只要保证不去修改 real_routers 里面的内容即可
    real_routers = model.routers 
    
    results = {
        "noise": NOISE_LEVELS,
        "single_logit": [], "single_rank": [],
        "comix_logit": [], "comix_rank": []
    }

    print("\n" + "="*85)
    print(f"{'Sigma':<6} | {'Mode':<15} | {'Logit (High Better)':<22} | {'Rank (Low Better)':<20}")
    print("="*85)

    for sigma in NOISE_LEVELS:
        model.noise_sigma = sigma
        
        # === 1. Single Expert Baseline ===
        expert_logits = []
        expert_ranks = []
        
        for i in range(NUM_EXPERTS):
            # === [关键修复] 创建一个新的 ModuleList，而不是原地修改 ===
            # 这样 real_routers 就不会被破坏
            forced_routers_list = [
                ForcedRouter(i, NUM_EXPERTS).to(model.device).to(torch.bfloat16)
                for _ in range(len(real_routers))
            ]
            model.routers = torch.nn.ModuleList(forced_routers_list)

            # 测量
            l, r = measure_performance(model, tokenizer, input_ids, target_id, n_samples=5)
            expert_logits.append(l)
            expert_ranks.append(r)
        
        # === [关键修复] 恢复原始 Router 对象 ===
        # 直接把指针指回 real_routers，不需要 load_state_dict
        model.routers = real_routers

        avg_single_logit = sum(expert_logits) / len(expert_logits)
        avg_single_rank = sum(expert_ranks) / len(expert_ranks)
        
        results["single_logit"].append(avg_single_logit)
        results["single_rank"].append(avg_single_rank)
        print(f"{sigma:<6.1f} | {'Avg Single':<15} | {avg_single_logit:<22.4f} | {avg_single_rank:<20.2f}")

        # === 2. CoMix (Geometric Defense) ===
        comix_logit, comix_rank = measure_performance(model, tokenizer, input_ids, target_id, N_SAMPLES)
        
        results["comix_logit"].append(comix_logit)
        results["comix_rank"].append(comix_rank)
        print(f"{sigma:<6.1f} | {'CoMix (Ours)':<15} | {comix_logit:<22.4f} | {comix_rank:<20.2f}")
        print("-" * 85)

    plot_results(results)

def plot_results(results):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results["noise"], results["single_logit"], 'o--', label="Avg Single Expert", color="gray", alpha=0.7)
    plt.plot(results["noise"], results["comix_logit"], 'o-', label="CoMix (Geometric Defense)", color="red", linewidth=2)
    plt.xlabel("Noise Sigma")
    plt.ylabel("Target Token Logit")
    plt.title("Robustness: Logit Stability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(results["noise"], results["single_rank"], 'o--', label="Avg Single Expert", color="gray", alpha=0.7)
    plt.plot(results["noise"], results["comix_rank"], 'o-', label="CoMix (Geometric Defense)", color="blue", linewidth=2)
    plt.xlabel("Noise Sigma")
    plt.ylabel("Target Token Rank (Lower is Better)")
    plt.title("Robustness: Prediction Rank")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    
    output_path = "privacy_comparison.png"
    plt.savefig(output_path)
    print(f"\n>>> Comparison plot saved to {output_path}")

if __name__ == "__main__":
    main()