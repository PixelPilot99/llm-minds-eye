from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


# 加载原始基础模型（如 Qwen1.5-1.8B）
model_name = "./models/Qwen2.5-1.5B"  # 替换为你的原始模型
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载原始模型（配置需与训练时一致）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,           # 保持与训练相同的量化配置
    device_map="auto",           # 自动分配设备
    torch_dtype=torch.float16,   # 数据类型
    #trust_remote_code=True     # 如果需要（如Qwen系列可能需要）
)

# 加载LoRA适配器
lora_path = "./matrix_lora_output"
model = PeftModel.from_pretrained(model, lora_path)


tokenizer.pad_token = tokenizer.eos_token  # 用结束符作为填充
print(f"Pad token set to: {tokenizer.pad_token}")
# 关键设置 - 解决乱码
tokenizer.padding_side = "left"  # 重要！

forced_token_id=tokenizer.encode("<|assistant|>")[0]  # 强制从答案部分开始生成

# 回答问题的函数
def answer_medical_question(instruction,user,matrix):
    # Qwen2.5的对话格式
    prompt = f"{instruction}+<|im_start|>user\n{user}{matrix}<|im_end|><|im_start|>assistant\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=3000
    ).to("cuda")



    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.9,
        top_k=50,
        top_p =0.92,
        do_sample=True,
        repetition_penalty=1.8,
        no_repeat_ngram_size=3,
        #pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        forced_bos_token_id=forced_token_id,

    )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.split("<|im_end|>")[0].strip()


# 测试问题
instruction = "<|im_start|>system\n这里展示了一个矩阵图像，根据结构描述画面内容，不需要输出矩阵。<|im_end|>"
user = "以自然语言描述这个物体"
matrix = '''R1:32☆|R2:32☆|R3:32☆|R4:16☆1★15☆|R5:15☆3★14☆|R6:14☆4★14☆|R7:14☆5★13☆|R8:13☆7★12☆|R9:13☆7★12☆|R10:12☆9★11☆|R11:12☆9★11☆|R12:11☆11★10☆|R13:11☆11★10☆|R14:10☆13★9☆|R15:10☆14★8☆|R16:9☆15★8☆|R17:9☆15★8☆|R18:9☆4★1☆5★1☆4★8☆|R19:9☆3★2☆5★2☆3★8☆|R20:9☆2★3☆5★3☆2★8☆|R21:9☆1★4☆5★4☆1★8☆|R22:14☆5★13☆|R23:14☆5★13☆|R24:14☆5★13☆|R25:14☆5★13☆|R26:14☆5★13☆|R27:14☆5★13☆|R28:14☆5★13☆|R29:14☆5★13☆|R30:14☆5★13☆|R31:32☆|R32:32☆'''
#question = f"<|im_start|>user\n{user}<|im_end|><|vision_start|>{matrix}<|vision_end|>"
answer = answer_medical_question(instruction=instruction,user=user,matrix=matrix)
print(answer)

