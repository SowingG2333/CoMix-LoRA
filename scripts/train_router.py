import os
import sys
import torch
import logging
from dataclasses import dataclass, field
from typing import Dict
from safetensors.torch import load_file 

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training
from datasets import load_dataset

# === 路径设置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
mixlora_path = os.path.join(root_dir, "MixLoRA")
sys.path.append(mixlora_path)

from mixlora import MixLoraConfig
import mixlora.model
mixlora.model._compatible_model_types["qwen3"] = "_llama_forward"
from mixlora.model import inject_adapter_in_model

logger = logging.getLogger(__name__)

@dataclass
class Arguments:
    base_model: str = "/root/gpufree-data/hf/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
    expert_dir: str = "/root/gpufree-data/OverlappedLoRA/model"
    router_data: str = "/root/gpufree-data/OverlappedLoRA/data/router_train.json"
    num_experts: int = 8
    noise_sigma: float = 4.0
    clip_threshold: float = 1.0

def patch_mixlora_model(model):
    """
    核心修复函数：
    1. 遍历模型找到 mixlora_moes 字典
    2. 将 Router Gate 转换为 Parameter
    3. 将 python dict 转换为 nn.ModuleDict 以便 PyTorch 识别
    4. 确保所有 LoRA 权重在正确设备上
    """
    print(">>> Starting Deep Patching for MixLoRA...")
    patched_routers = 0
    patched_experts = 0
    
    # 获取需要 Patch 的模块列表
    modules_to_patch = []
    for name, module in model.named_modules():
        if hasattr(module, "mixlora_moes") and isinstance(module.mixlora_moes, dict):
            modules_to_patch.append(module)
            
    for module in modules_to_patch:
        # 确定目标设备 (通常跟随 down_proj)
        if hasattr(module, "down_proj"):
            target_device = module.down_proj.weight.device
        else:
            target_device = list(module.parameters())[0].device
            
        # 创建 ModuleDict 容器
        safe_moes = torch.nn.ModuleDict()
        
        for adapter_name, moe_layer in module.mixlora_moes.items():
            # === 1. 处理 Router Gate ===
            if hasattr(moe_layer, "gate_"):
                # 转换为 Parameter
                if not isinstance(moe_layer.gate_, torch.nn.Parameter):
                    p = torch.nn.Parameter(moe_layer.gate_.to(device=target_device, dtype=torch.bfloat16))
                    moe_layer.gate_ = p
                else:
                    # 已经在 Parameter 里了，确保设备正确
                    moe_layer.gate_.data = moe_layer.gate_.data.to(device=target_device, dtype=torch.bfloat16)
                
                moe_layer.gate_.requires_grad = True
                patched_routers += 1
            
            # === 2. 处理 Experts 容器 (dict -> ModuleDict) ===
            # MixLoRA 默认用 dict 存 experts，导致 PyTorch 找不到它们
            if hasattr(moe_layer, "experts_") and isinstance(moe_layer.experts_, dict):
                safe_experts = torch.nn.ModuleDict()
                for exp_name, exp_layer in moe_layer.experts_.items():
                    # 移动 LoRA 权重到 GPU
                    exp_layer.lora_A.to(target_device)
                    exp_layer.lora_B.to(target_device)
                    safe_experts[exp_name] = exp_layer
                moe_layer.experts_ = safe_experts
                patched_experts += len(safe_experts)

            # === 3. 处理循环引用 ===
            if "base_layer_" in moe_layer._modules:
                base = moe_layer._modules["base_layer_"]
                del moe_layer._modules["base_layer_"]
                moe_layer.__dict__["base_layer_"] = base
            
            safe_moes[adapter_name] = moe_layer
            
        # 替换原有的 dict
        module.mixlora_moes = safe_moes
        
    print(f">>> Patched {patched_routers} Router Gates and {patched_experts} Expert Containers.")
    
    # 额外检查 Shared Attention LoRA
    for name, module in model.named_modules():
        if module.__class__.__name__ == "LoraLinear":
            if hasattr(module, "base_layer_"):
                target_device = module.base_layer_.weight.device
                module.lora_A.to(target_device)
                module.lora_B.to(target_device)

def train():
    parser = transformers.HfArgumentParser((Arguments, TrainingArguments))
    if len(sys.argv) == 1:
        print(">>> No args provided, using default arguments...")
        args, training_args = parser.parse_args_into_dataclasses(args=[
            "--output_dir", "/root/gpufree-data/OverlappedLoRA/model/router",
            "--num_train_epochs", "3",
            "--per_device_train_batch_size", "2",
            "--gradient_accumulation_steps", "8",
            "--learning_rate", "1e-4",
            "--logging_steps", "5",
            "--save_steps", "100",
            "--save_total_limit", "2",
            "--overwrite_output_dir",
            "--gradient_checkpointing", "True",
            "--bf16", "True"
        ])
    else:
        args, training_args = parser.parse_args_into_dataclasses()

    # 1. Config
    print(">>> [1/6] Initializing Config...")
    config_dict = {
        "base_model_name_or_path": args.base_model,
        "task_type": "CAUSAL_LM",
        "peft_type": "MIXLORA",
        "routing_strategy": "mixlora",
        "num_experts": args.num_experts,
        "top_k": 2,
        "noise_sigma": args.noise_sigma,
        "clip_threshold": args.clip_threshold,
        "router_loss": True,
        "router_aux_loss_coef": 0.01,
        "router_init_range": 0.02,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "act_fn": "silu"
    }
    config = MixLoraConfig.from_config(config_dict)
    config.dtype_ = torch.bfloat16
    config.adapter_name_ = "default"

    # 2. Base Model (with 4-bit Quantization)
    print(f">>> [2/6] Loading Base Model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 3. Stitching
    print(">>> [3/6] Stitching Experts in Memory...")
    mixlora_weights = {}
    attn_accum = {} 
    
    for i in range(args.num_experts):
        expert_base_path = os.path.join(args.expert_dir, f"expert_{i}")
        safetensors_path = os.path.join(expert_base_path, "adapter_model.safetensors")
        bin_path = os.path.join(expert_base_path, "adapter_model.bin")

        if os.path.exists(safetensors_path):
            expert_state = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            expert_state = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"Expert {i} not found")
        
        for k, v in expert_state.items():
            if "lora_" not in k: continue
            parts = k.split(".")
            try:
                if "layers" in parts:
                    idx = parts.index("layers")
                    layer_idx = parts[idx+1]
                    proj_part = parts[idx+3]
                    suffix = ".".join(parts[idx+4:])
                else:
                    continue
            except IndexError:
                continue

            v = v.to(dtype=torch.bfloat16)

            if proj_part in ["gate_proj", "up_proj", "down_proj"]:
                new_key = f"mixlora.layers.{layer_idx}.mlp.{proj_part}.experts.{i}.{suffix}"
                mixlora_weights[new_key] = v
            elif proj_part in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                attn_key = f"mixlora.layers.{layer_idx}.self_attn.{proj_part}.{suffix}"
                if attn_key not in attn_accum:
                    attn_accum[attn_key] = v.float()
                else:
                    attn_accum[attn_key] += v.float()
    
    for k, v_sum in attn_accum.items():
        mixlora_weights[k] = (v_sum / args.num_experts).to(dtype=torch.bfloat16)

    # 4. Router Init
    print(">>> [4/6] Initializing Router Gates...")
    hidden_size = model.config.hidden_size
    for layer_idx in range(model.config.num_hidden_layers):
        gate_key = f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"
        gate_weight = torch.randn(args.num_experts, hidden_size, dtype=torch.bfloat16) * 0.02
        mixlora_weights[gate_key] = gate_weight

    # 5. Inject
    print(">>> [5/6] Injecting Adapter...")
    inject_adapter_in_model(model, config, mixlora_weights)

    # === [核心修复] 调用 Patch 函数 ===
    patch_mixlora_model(model)
    # ===============================

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # Freeze / Unfreeze
    trainable_count = 0
    print(">>> Checking parameters for unfreezing:")
    for name, param in model.named_parameters():
        # [Strict Filter] 只训练包含 gate 关键字 且 是浮点类型的参数
        # 这样可以完美避开 gate_proj (uint8/Params4bit)
        if ("gate" in name.lower()) and (param.dtype in [torch.bfloat16, torch.float32, torch.float16]):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
    
    print(f">>> [SAFETY CHECK] Total Trainable Parameters: {trainable_count}")
    
    if trainable_count == 0:
        print("!!! DEBUG: Printing all parameter names !!!")
        for n, p in model.named_parameters():
            if "gate" in n: print(f"{n} | {p.dtype} | {p.device}")
        raise ValueError("!!! No trainable parameters found! Please check the logs above. !!!")

    # 6. Training
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("json", data_files=args.router_data, split="train")
    
    def preprocess(examples):
        inputs = [f"Below is an instruction that describes a task.\n\n### Instruction:\n{i}\n\n### Response:\n{o}{tokenizer.eos_token}" 
                  for i, o in zip(examples['instruction'], examples['output'])]
        model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
        
        labels = model_inputs["input_ids"].copy()
        labels = [
            [(l if l != pad_token_id else -100) for l in label] 
            for label in labels
        ]
        model_inputs["labels"] = labels
        return model_inputs

    train_ds = dataset.map(preprocess, batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )

    print(">>> Start Training...")
    trainer.train()
    
    model.save_pretrained(training_args.output_dir)
    print(f">>> Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    train()