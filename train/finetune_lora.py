import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # 比TRITON_DISABLE更底层的开关
os.environ["TRITON_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 禁用异步执行，便于调试
import torch

torch.backends.cuda.enable_flash_sdp = False  # 必须关闭Flash Attention
torch.backends.cuda.enable_mem_efficient_sdp = False
torch.backends.cuda.enable_math_sdp = True  # 仅保留最基础数学实现

from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments,DataCollatorForSeq2Seq,Trainer

import json

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"  # 禁用Windows不支持的特性
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Optional set GPU device ID

#模型加载
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./models/Qwen2.5-1.5B",
    max_seq_length = 3000,
    dtype = None,
    load_in_4bit = True,
    local_files_only = True,
    use_gradient_checkpointing="standard",
    rope_scaling={"type": "dynamic", "factor": 2.0},  # 增强位置外推

)

tokenizer.pad_token = tokenizer.eos_token  #0602+

print(f"pad_token_id: {tokenizer.pad_token_id}")
print(f"eos_token_id: {tokenizer.eos_token_id}")



# 确认分词器类名
print("Tokenizer class:", type(tokenizer).__name__)  # 例如 Qwen2Tokenizer


#读取训练文件
class JsonFolderDataset:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    def __len__(self):
        return len(self.file_list)

    def __iter__(self):
        for file_name in self.file_list:
            with open(os.path.join(self.folder_path, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
            yield {
                "matrix": data["matrix"],
                "answer": data["answer"],
                'instruction':data['instruction']
            }

tokenizer.padding_side = "left"  # 确保填充在左侧，答案在右侧固定位置

#对应的预处理代码
def format_data(examples):
    #inputs = []
    #labels = []

    instruction = examples["instruction"]
    matrix = examples["matrix"]
    answer = examples["answer"]
    full_text = instruction + matrix + answer  # 直接拼接完整文本

    # 统一对完整文本进行 Tokenize
    tokenized = tokenizer(full_text, add_special_tokens=False)   #原都为False
    input_ids = tokenized["input_ids"]

    # 计算 input_text 的长度（instruction + matrix）
    input_text = instruction + matrix
    input_tokenized = tokenizer(input_text, add_special_tokens=False)["input_ids"]
    answer_start_idx = len(input_tokenized)

    # 生成标签：input 部分设为 -100，answer 部分保留 token
    label = [-100] * len(input_ids)
    label[answer_start_idx:] = input_ids[answer_start_idx:]  # 注意：如果分词不一致需调整

    return {"input_ids": input_ids, "labels": label}


# 加载训练数据（修改部分）
def create_dataset(folder_path):
    dataset = Dataset.from_generator(
        JsonFolderDataset,
        gen_kwargs={"folder_path": folder_path}
    )
    return dataset


Raw_data = create_dataset("./data/dsl/final")  # 替换为你的训练文件夹路径
train_dataset = Raw_data.map(format_data)




# 设置训练参数（保持不变）
args = TrainingArguments(
    output_dir = "./masked_lm",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps=2,
    num_train_epochs = 60,
    learning_rate = 2e-5,
    warmup_ratio=0.1,             # 10%步数用于学习率预热
    max_grad_norm=1.0,            # 梯度裁剪阈值
    warmup_steps=30,  # 对于小数据集更重要
    weight_decay=0.01,
    logging_dir="logs",  #可视化
    logging_steps = 20,
    save_strategy = "no",  # 改为按步保存（原为"epoch"）---"steps"
    #save_steps = 100,         # 每100步保存一次检查点
    #save_total_limit = 2,     # 最多保留2个检查点（避免磁盘爆炸）
    #fp16 = True,
    torch_compile=False,
    lr_scheduler_type="cosine",
    # 余弦退火更适合小数据
    #load_best_model_at_end = True,  # 禁用此选项以支持续传
)

# 应用 LoRA 配置（保持不变）
model = FastLanguageModel.get_peft_model(
    model,
    r=4,    # LoRA 秩
    target_modules=["q_proj", "k_proj", "v_proj"],  #"o_proj", #"gate_proj","up_proj","down_proj"],
    lora_alpha=8,
    lora_dropout=0,   #默认0有优化，原本0.1
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
    #task_type="CAUSAL_LM",
)

# 启动训练器（修改文本字段）
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    # 使用标准的数据整理器
    data_collator=DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,  # 按需启用
    label_pad_token_id=-100,
    return_tensors="pt",


    )
)

trainer.train()
'''
# 在 trainer.train() 前添加检查点检测
last_checkpoint = None
if os.path.exists(args.output_dir):
    checkpoints = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
'''
# 开始训练并保存（保持不变）
#trainer.train(resume_from_checkpoint=True)
model.save_pretrained("matrix_lora_output")

