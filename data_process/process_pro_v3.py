import json
import os
import re
import random

# 配置
input_dir = '/root/autodl-tmp/raw_data'  # 你的txt文件夹
output_file = '/root/autodl-tmp/LLaMA-Factory/data/august_pro.json'
min_output_len = 400  # 输出至少要有400字，逼它写长
max_seq_len = 2000    # 总长度控制，防止显存爆炸

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = text.replace('　', '').strip()
    return text

dataset = []
prompts_style = [
    "请模仿八月长安的文风，续写这段情节：",
    "基于《最好的我们》或《暗恋》的世界观，继续描写：",
    "保持细腻、清冷的笔触，描写接下来的心理活动和环境：",
    "你是八月长安，请继续创作这段小说："
]

print("开始处理数据，模式：长文本强化...")

for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        print(f"正在深度解析: {filename} ...")
        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
            content = clean_text(f.read())
            
            # 按段落切分，而不是按字切分
            paragraphs = content.split('\n')
            paragraphs = [p for p in paragraphs if len(p) > 20] # 过滤掉太短的废话
            
            i = 0
            while i < len(paragraphs) - 10:
                # 策略：Input 是 1-2 个段落，Output 是后面紧接着的 5-10 个段落
                # 这样模型就学会了：给我一点点线索，我要回忆起一大段故事
                
                input_text = "\n".join(paragraphs[i : i+2])
                
                # 动态决定输出长度，让模型适应不同长度的输出
                num_output_paras = random.randint(5, 12)
                output_text = "\n".join(paragraphs[i+2 : i+2+num_output_paras])
                
                # 只有当 Input+Output 长度合适，且 Output 足够长时才录入
                if len(output_text) > min_output_len and len(input_text) + len(output_text) < max_seq_len:
                    
                    # 偶尔注入角色扮演指令，增强人物记忆
                    instruction = random.choice(prompts_style)
                    if "洛枳" in input_text:
                        instruction = "你现在是洛枳，请用你的日记体口吻续写此刻的心情："
                    elif "余淮" in input_text:
                        instruction = "你现在是余淮，请表现出你少年的意气风发或无奈，续写对话："
                    
                    dataset.append({
                        "instruction": instruction,
                        "input": input_text,
                        "output": output_text
                    })
                
                # 滑动窗口：步长小一点，增加数据密度
                i += 2

# 保存
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"数据构建完成！共生成 {len(dataset)} 条高质量长文数据。")
print(f"请务必在 dataset_info.json 中注册 'august_pro'。")