import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from safetensors.torch import load_file
import os

class CoMixRouter(nn.Module):
    """
    CoMix 路由模块
    初始化为全 0，保证初始状态下概率均匀 (1/N)。
    """
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts = num_experts
        
        # 初始化为全 0 -> Softmax 后为均匀分布
        nn.init.zeros_(self.gate.weight)
        
        # 缓存概率用于 Loss 计算
        self.last_probs = None

    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        probs = F.softmax(logits, dim=-1) 
        self.last_probs = probs 
        return probs

class CoMixLoRALayer(nn.Module):
    """
    支持几何防御的混合 LoRA 层
    严谨 DP 逻辑：Expert Output -> Clip (Sensitivity) -> Add Noise -> Weighted Sum
    """
    def __init__(self, base_layer: nn.Linear, num_experts: int, r: int, alpha: int, dropout: float = 0.05):
        super().__init__()
        self.base_layer = base_layer 
        self.num_experts = num_experts
        self.r = r
        self.scaling = alpha / r
        
        # 运行时属性
        self.current_probs = None
        self.noise_sigma = 0.0
        self.clip_threshold = -1.0 
        
        # 初始化专家权重
        self.lora_A = nn.ModuleList([
            nn.Linear(base_layer.in_features, r, bias=False) for _ in range(num_experts)
        ])
        self.lora_B = nn.ModuleList([
            nn.Linear(r, base_layer.out_features, bias=False) for _ in range(num_experts)
        ])
        self.dropout = nn.Dropout(dropout)
        
        # Kaiming 初始化
        for i in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_A[i].weight, a=5**0.5)
            nn.init.zeros_(self.lora_B[i].weight)

    def forward(self, x, *args, **kwargs):
        # 1. 如果没有 Router 介入，仅运行 Base Layer
        if self.current_probs is None:
            return self.base_layer(x)

        # 2. 准备计算
        router_probs = self.current_probs # [batch, seq, num_experts]
        batch_size, seq_len, _ = x.shape
        
        # 确保输出容器 dtype 与输入一致 (BFloat16)
        final_lora_out = torch.zeros(
            batch_size, seq_len, self.base_layer.out_features, 
            device=x.device, dtype=x.dtype
        )
        
        # 3. 遍历所有专家
        for e in range(self.num_experts):
            weight = router_probs[:, :, e].unsqueeze(-1)

            # 计算 LoRA 输出 (BFloat16)
            lora_out = self.lora_B[e](self.lora_A[e](self.dropout(x))) * self.scaling
            
            # === DP 步骤 1: 裁剪 (Clipping) ===
            # 必须在加噪之前进行，确保护每个样本的 L2 灵敏度被限制在 clip_threshold 以内
            if self.clip_threshold > 0:
                # 对 hidden_dim 维度计算范数 (Per-sample clipping)
                norm = torch.norm(lora_out, p=2, dim=-1, keepdim=True)
                # 缩放系数: min(1, C / norm)
                scale = torch.clamp(self.clip_threshold / (norm + 1e-6), max=1.0)
                lora_out = lora_out * scale
            
            # === DP 步骤 2: 加噪 (Noise Injection) ===
            # 噪声必须加在被裁剪过的信号上
            if self.noise_sigma > 0.0:
                # 计算噪声标准差
                # 如果定义 sigma 为 "Noise Multiplier" (标准 DP 定义)，则 std = sigma * C
                noise_std = self.noise_sigma
                if self.clip_threshold > 0:
                    noise_std *= self.clip_threshold
                
                expert_noise = torch.randn_like(lora_out) * noise_std
                lora_out = lora_out + expert_noise

            # 加权累加
            final_lora_out += lora_out * weight

        # 注意：这里不再需要最后的裁剪了，因为每个部分都已经裁剪过了
            
        return self.base_layer(x) + final_lora_out

class CoMixModel(nn.Module):
    def __init__(self, base_model_path, num_experts=8, r=16, alpha=32, device="cuda"):
        super().__init__()
        print(f">>> [CoMix] Loading base model from {base_model_path}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 # 基座计算类型
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        self.num_experts = num_experts
        self.r = r
        self.alpha = alpha
        self.device = device
        
        # 隐私参数
        self.noise_sigma = 0.0 
        self.clip_threshold = -1.0

        # Hack Trainer
        self.is_loaded_in_4bit = False
        self.quantization_config = None 
        
        self.replace_modules()
        
        # 初始化 Router 并转为 BFloat16
        self.routers = nn.ModuleList([
            CoMixRouter(self.model.config.hidden_size, num_experts).to(device).to(torch.bfloat16)
            for _ in range(self.model.config.num_hidden_layers)
        ])
        
        self.register_hooks()

    def replace_modules(self):
        print(">>> [CoMix] Replacing Linear layers and casting to BFloat16...")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            modules_to_replace = []
            for name, module in layer.named_modules():
                if any(name.endswith(t) for t in target_modules):
                    path = name.split('.')
                    leaf_name = path[-1]
                    parent_mod = layer
                    for p in path[:-1]:
                        parent_mod = getattr(parent_mod, p)
                    modules_to_replace.append((parent_mod, leaf_name, module))
            
            for parent, leaf_name, original_linear in modules_to_replace:
                if isinstance(original_linear, CoMixLoRALayer): continue
                
                # 创建层
                comix_layer = CoMixLoRALayer(original_linear, self.num_experts, self.r, self.alpha)
                
                # === [关键修复] 强制转换为 BFloat16 并移动到正确设备 ===
                comix_layer = comix_layer.to(dtype=torch.bfloat16, device=self.device)
                
                setattr(parent, leaf_name, comix_layer)

    def load_experts(self, expert_dir):
        print(f">>> [CoMix] Loading Experts from {expert_dir}...")
        for i in range(self.num_experts):
            path_safe = os.path.join(expert_dir, f"expert_{i}", "adapter_model.safetensors")
            path_bin = os.path.join(expert_dir, f"expert_{i}", "adapter_model.bin")
            
            target_path = path_safe if os.path.exists(path_safe) else (path_bin if os.path.exists(path_bin) else None)
            if not target_path:
                print(f"!!! Warning: Expert {i} not found.")
                continue
                
            state_dict = load_file(target_path) if target_path.endswith('.safetensors') else torch.load(target_path, map_location="cpu")
            
            # 加载逻辑
            for k, v in state_dict.items():
                if "lora_" not in k: continue
                parts = k.split('.')
                try:
                    if 'layers' in parts: l_idx = int(parts[parts.index('layers') + 1])
                    else: continue 
                    
                    proj_name = None
                    for t in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                        if t in parts: proj_name = t; break
                    if not proj_name: continue

                    is_A = "lora_A" in k
                    block = self.model.model.layers[l_idx]
                    
                    # 递归查找模块
                    for m_name, module in block.named_modules():
                        if m_name.endswith(proj_name) and isinstance(module, CoMixLoRALayer):
                            target = module.lora_A[i] if is_A else module.lora_B[i]
                            # 确保加载的权重也是 BFloat16
                            target.weight.data = v.to(target.weight.device).to(torch.bfloat16)
                            break
                except Exception as e:
                    pass

    def register_hooks(self):
        def router_hook(module, args, layer_idx):
            hidden_states = args[0]
            # 确保 hidden_states 也是 BFloat16
            router = self.routers[layer_idx]
            probs = router(hidden_states)
            
            for m in module.modules():
                if isinstance(m, CoMixLoRALayer):
                    m.current_probs = probs
                    m.noise_sigma = self.noise_sigma
                    m.clip_threshold = self.clip_threshold
            return None 

        for idx, layer in enumerate(self.model.model.layers):
            layer.register_forward_pre_hook(lambda m, a, i=idx: router_hook(m, a, i))

    def get_aux_loss(self):
        """计算负载均衡 Loss"""
        total_loss = 0
        valid_routers = 0
        for router in self.routers:
            if not hasattr(router, 'last_probs') or router.last_probs is None: continue
            probs = router.last_probs
            mean_probs = probs.mean(dim=(0, 1))
            target = torch.full_like(mean_probs, 1.0 / self.num_experts)
            total_loss += F.mse_loss(mean_probs, target)
            valid_routers += 1
        if valid_routers == 0: return torch.tensor(0.0, device=self.device, requires_grad=True)
        return total_loss / valid_routers

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.routers.state_dict(), os.path.join(save_directory, "routers.pt"))
    
    def __getattr__(self, name):
        try: return super().__getattr__(name)
        except AttributeError: return getattr(self.model, name)