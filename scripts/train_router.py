import os
import sys
import torch
import logging
import transformers
from dataclasses import dataclass, field
from typing import Optional
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer
from datasets import load_dataset

# 导入 Custom Router 模型
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_router import CoMixModel

logger = logging.getLogger(__name__)

@dataclass
class Arguments:
    """
    超参数配置类
    可以通过命令行参数覆盖默认值，例如: --noise_sigma 0.0
    """
    base_model: str = field(
        default="/root/gpufree-data/hf/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
        metadata={"help": "Base model path"}
    )
    expert_dir: str = field(
        default="/root/gpufree-data/OverlappedLoRA/model",
        metadata={"help": "Path to directory containing expert_* folders"}
    )
    router_data: str = field(
        default="/root/gpufree-data/OverlappedLoRA/data/router_train.json",
        metadata={"help": "Path to training data json"}
    )
    num_experts: int = field(default=8, metadata={"help": "Number of experts"})
    
    # 隐私与噪声参数
    noise_sigma: float = field(
        default=0.0, 
        metadata={"help": "Noise sigma for DP. MUST be 0.0 during training for router to learn."}
    )
    clip_threshold: float = field(default=1.0, metadata={"help": "Gradient clipping threshold"})
    
    # LoRA 参数 (需与 expert 训练时一致)
    r: int = field(default=16)
    alpha: int = field(default=32)
    
    # Router 训练专用参数 (如果不通过 TrainingArguments 传，就用这里的默认值)
    router_lr: float = field(default=1e-3, metadata={"help": "Learning rate for router"})
    aux_loss_coef: float = field(default=0.05, metadata={"help": "Coefficient for load balancing loss"})

class RouterTrainer(Trainer):
    def __init__(self, aux_loss_coef=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_loss_coef = aux_loss_coef

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. 主任务 Loss
        if num_items_in_batch is not None:
             output = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        else:
             output = super().compute_loss(model, inputs, return_outputs=True)
        
        loss, outputs = output
        
        # 2. 负载均衡 Loss
        if hasattr(model, "module"):
            aux_loss = model.module.get_aux_loss()
        else:
            aux_loss = model.get_aux_loss()
            
        # 3. 总 Loss
        total_loss = loss + self.aux_loss_coef * aux_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None: output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f">>> [Trainer] Saving ONLY Router weights to {output_dir}...")
        
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_to_save, "routers"):
            torch.save(model_to_save.routers.state_dict(), os.path.join(output_dir, "routers.pt"))

def recursive_clear_quantization_flags(target_obj):
    flags = ["is_loaded_in_4bit", "is_loaded_in_8bit", "is_quantized"]
    for f in flags:
        if hasattr(target_obj, f): setattr(target_obj, f, False)
    if hasattr(target_obj, "config"):
        for f in flags:
            if hasattr(target_obj.config, f): setattr(target_obj.config, f, False)

def train():
    parser = transformers.HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

    if len(sys.argv) == 1:
        print(">>> No args provided, using DEFAULT arguments...")
        args, training_args = parser.parse_args_into_dataclasses(args=[
            "--output_dir", "/root/gpufree-data/OverlappedLoRA/model/router",  # 输出路径
            "--num_train_epochs", "3",                # 训练轮数
            "--per_device_train_batch_size", "8",     # Batch Size
            "--gradient_accumulation_steps", "2",     # 梯度累积步数
            "--learning_rate", "1e-3",                # Router 学习率
            "--logging_steps", "5",
            "--save_steps", "100",
            "--save_total_limit", "2",
            "--remove_unused_columns", "False",       # 防止 Trainer 删数据
            "--gradient_checkpointing", "True",       # 开启梯度检查点，节省显存
            "--bf16", "True",                         # 开启 BF16
            "--report_to", "none"                     # 不上报 wandb
        ])
    else:
        # 如果命令行传入了参数，则正常解析
        args, training_args = parser.parse_args_into_dataclasses()

    # 强制 Trainer 保留所有列，不要自动删除 input_ids 等
    training_args.remove_unused_columns = False 

    print(">>> [1/4] Initializing CoMix Model...")
    device_id = f"cuda:{training_args.local_rank}" if training_args.local_rank != -1 else "cuda"
    
    model = CoMixModel(
        base_model_path=args.base_model,
        num_experts=args.num_experts,
        r=args.r,
        alpha=args.alpha,
        device=device_id
    )
    
    # 设置参数
    model.noise_sigma = args.noise_sigma
    model.clip_threshold = args.clip_threshold
    
    # 加载专家
    model.load_experts(args.expert_dir)

    # 冻结参数
    print(">>> [2/4] Freezing Base Model & Experts...")
    if training_args.gradient_checkpointing:
        model.model.enable_input_require_grads()

    trainable_params = 0
    for name, param in model.named_parameters():
        if "routers" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    print(f">>> Trainable Parameters (Routers): {trainable_params}")

    # 数据处理
    print(">>> [3/4] Loading Dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("json", data_files=args.router_data, split="train")
    
    def preprocess(examples):
        inputs = [f"Below is an instruction that describes a task.\n\n### Instruction:\n{i}\n\n### Response:\n{o}{tokenizer.eos_token}" 
                  for i, o in zip(examples['instruction'], examples['output'])]
        model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
        
        labels = model_inputs["input_ids"].copy()
        labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
        model_inputs["labels"] = labels
        return model_inputs

    train_ds = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    # Hack Quantization Flags
    recursive_clear_quantization_flags(model)
    if hasattr(model, "model"): recursive_clear_quantization_flags(model.model)

    print(">>> [4/4] Starting Training...")
    trainer = RouterTrainer(
        aux_loss_coef=args.aux_loss_coef, # 传入 aux loss 系数
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    print(">>> Training Finished.")

if __name__ == "__main__":
    train()