import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# 你的 checkpoint 路径
CHECKPOINT_PATH = "/root/gpufree-data/OverlappedLoRA/model/router/routers.pt"

def check_status():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"等待 Checkpoint 生成... {CHECKPOINT_PATH} 不存在")
        return

    print(f">>> Loading Checkpoint: {CHECKPOINT_PATH}")
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    print(f">>> Found {len(state_dict)} keys in state_dict.")
    
    # 随机抽查第 0 层和 第 15 层（如果有的话）
    layers_to_check = [0, len(state_dict)//2]
    
    for i in layers_to_check:
        key = f"{i}.gate.weight"
        if key not in state_dict: continue
        
        weight = state_dict[key] # [num_experts, hidden_size]
        
        # 1. 检查是否还是全 0
        if torch.all(weight == 0):
            print(f"\n[Layer {i}] ⚠️ 警告: 权重全是 0！Router 根本没动！")
            print("可能原因：学习率太低 / 梯度被切断 / 代码有 Bug")
        else:
            print(f"\n[Layer {i}] ✅ 正常: 权重已更新")
            print(f"   Max: {weight.max().item():.6f}")
            print(f"   Min: {weight.min().item():.6f}")
            print(f"   Mean: {weight.mean().item():.6f}")
            print(f"   Std:  {weight.std().item():.6f} (标准差越大，说明学到的特征越明显)")

    print("\n>>> 结论判断:")
    first_w = state_dict["0.gate.weight"]
    if first_w.std() > 1e-4:
        print("🎉 恭喜！Router 正在学习差异化特征。Loss 震荡是正常的微调现象。")
    else:
        print("💀 完蛋。Router 基本没动。请检查 learning_rate 是否太小 (建议 1e-3) 或 remove_unused_columns 是否设为 False。")

if __name__ == "__main__":
    check_status()