import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 基座模型路径
base_model_path = "/root/autodl-tmp/qwen25_7b"

# 2. 你训练好的 LoRA 路径 (请确认这个路径下有 adapter_model.safetensors)
# 建议选你刚才觉得效果最好的那个 checkpoint，比如 checkpoint-500 或 600
lora_path = "/root/autodl-tmp/LLaMA-Factory/saves/Qwen2-7B/lora/august_fast_v4/checkpoint-1098" 

# 3. 测试文件和结果文件
input_file = "/root/autodl-tmp/test_cases.json"
output_file = "/root/autodl-tmp/test_results.json"

# ================= 加载模型 =================
print("正在加载基座模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16, # 3090用 fp16 或 bf16
    trust_remote_code=True
)

print(f"正在加载 LoRA 权重: {lora_path} ...")
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

# ================= 推理函数 =================
def generate_response(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # 使用 Qwen 的对话模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,      # 允许生成的最大长度
            temperature=0.75,         # 温度：0.7-0.8 比较稳，0.9 比较浪
            top_p=0.9,
            repetition_penalty=1.05,  # 关键：防复读参数
            do_sample=True
        )
    
    # 提取生成的回复（去掉输入的 prompt 部分）
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# ================= 批量执行 =================
print("开始批量测试...")

# 读取测试题
with open(input_file, 'r', encoding='utf-8') as f:
    test_cases = json.load(f)

results = []

# 遍历每一道题
for case in tqdm(test_cases):
    print(f"\n正在生成第 {case['id']} 题: {case['role']} ...")
    
    response = generate_response(case['system'], case['prompt'])
    
    # 打印预览一下
    print("-" * 30)
    print(f"Prompt: {case['prompt'][:30]}...")
    print(f"Response: {response[:50]}...")
    print("-" * 30)
    
    # 保存结果
    results.append({
        "id": case["id"],
        "role": case["role"],
        "prompt": case["prompt"],
        "model_response": response
    })

# 写入结果文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n全部完成！结果已保存到: {output_file}")