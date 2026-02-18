# Zhenhua-LLM (振华大模型)

> **Towards Stylistic Nuance: Fine-tuning Large Language Models for Literary Style Replication of August Changan**

<p align="center">
  <!-- GitHub Stars -->
  <a href="https://github.com/arce-star/Zhenhua-LLM/stargazers" target="_black">
    <img src="https://img.shields.io/github/stars/arce-star/Zhenhua-LLM?style=flat" alt="GitHub Stars"></a>
  <!-- Hugging Face 模型页 -->
  <a href="https://huggingface.co/NQworker/Zhenhua-7B-August-Style" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=flat" alt="Hugging Face"></a>
  <!-- 开源协议 -->
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat" alt="License">
  <!-- Python 版本 -->
  <img src="https://img.shields.io/badge/Python-3.11+-green.svg?style=flat" alt="Python">
</p>

<p align="center">
  <a href="./README.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="./README_zh.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
</p>

## 📖 Abstract / 摘要

**Zhenhua-LLM** 是一个专注于**文学风格迁移**（Text Style Transfer）与**特定角色人格锚定**（Character Personality Anchoring）的开源研究项目。本项目基于通义千问 `Qwen2.5-7B-Instruct` 基座，通过收集并深度处理中国著名作家八月长安的“振华三部曲”（《最好的我们》、《你好，旧时光》、《暗恋·橘生淮南》）全量正文语料，利用 **LoRA (Low-Rank Adaptation)** 轻量化微调技术，成功复刻了其细腻、清冷且富含哲理的文学笔触。

本项目不仅实现了文学意象的精准还原（如“红榜”、“穿堂风”、“侧脸”等），还针对生成模型常见的“逻辑碎片化”和“短文本偏见”进行了专项优化，能够稳定产出具有高度叙事连贯性的长篇文学段落。

---

## 🛠️ Methodology / 核心技术路线

### 1. 数据工程 (Data Engineering)
本项目摒弃了传统的文本均匀切分法，采用了**段落级滑动窗口 (Paragraph-level Sliding Window)** 与 **长文强化引导 (Length-Bias Training)** 策略：

*   **段落级连贯性 (Paragraph-level Continuity)**：采用跨度为 2 个自然段的滑动步进（Stride），确保训练数据中包含完整的文学叙事单元。相比于硬性的 Token 切分，这种方法能更好地保留八月长安作品中特有的转场逻辑与叙事节奏。
*   **长文输出强化 (Long-form Reinforcement)**：针对性构造 **1-2 个自然段输入 -> 5-12 个自然段高质量输出** 的映射对。通过强制设置输出内容必须超过 400 字，从数据层级解决了模型生成“挤牙膏”的问题，显著提升了文学创作的细腻程度。
*   **动态人设锚定 (Dynamic Persona Anchoring)**：脚本在指令层级（Instruction）内置了逻辑判断，当上下文检测到“洛枳”、“余淮”等关键人物时，会自动切换至专为人设定制的**锚定指令（Persona Prompt）**，实现文风复刻与角色特质的深度耦合。

### 2. 超参数规范 (Hyperparameter Paradigm)
训练基于 `LLaMA-Factory` 框架，在 NVIDIA RTX 3090 (24GB) 算力平台上完成，核心参数配置如下：

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Base Model** | Qwen2.5-7B-Instruct | SOTA Chinese open-source LLM |
| **Method** | LoRA | Low-Rank Adaptation (all linear layers) |
| **Rank / Alpha** | 32 / 64 | Balanced capacity for style and knowledge |
| **Learning Rate** | 1e-4 | Cosine decaying scheduler |
| **Cutoff Length** | 2048 | Crucial for modeling long literary passages |
| **Epochs** | 3.0 - 5.0 | Optimized to prevent catastrophic forgetting |

---

## 📈 实验评估 / Evaluation

### 定性分析 (Qualitative Analysis)
实验结果显示，Zhenhua-LLM 在以下维度显著优于通用指令模型：
1.  **风格忠实度 (Stylistic Fidelity)**：能够自如运用八月长安标志性的“第二人称心理描写”和“通感比喻”。
2.  **情感共鸣 (Affective Resonance)**：显著增强了心理描写占比，能细腻处理“卑微与骄傲并存”的复杂暗恋情感。
3.  **叙事连贯性 (Narrative Cohesion)**：在长文生成指令下，模型输出的平均 Token 长度提升了 300%，且逻辑中轴稳定。

> *Case Study: [View Detailed Samples Here](./samples/luozhi.md)*

---

## 🚀 Quick Start / 快速开始

### 1. Installation
```bash
git clone https://github.com/arce-star/Zhenhua-LLM.git
cd Zhenhua-LLM
pip install -r requirements.txt
```

### 2. Model Inference
本地推理建议调用本仓库提供的 `batch_infer.py` 脚本。该脚本已内置优化的采样参数：
- **Repetition Penalty**: 1.05 (有效防止长文生成的循环冗余)
- **Temperature**: 0.85 (保持文学创作所需的随机性灵感)
- **Top-p**: 0.9

```bash
python batch_infer.py --model_path /path/to/your/merged_model
```

---

## 📅 Roadmap / 开发路线
- [x] **v1.0**: 风格化 SFT 完成，初步复刻“振华味”语感。
- [x] **v2.0**: 引入滑动窗口训练，解决长文本生成逻辑崩坏。
- [ ] **v3.0 (In Progress)**: 角色扮演专项增强（RP-Enhanced），实现多人物人格不混淆。
- [ ] **v4.0**: 引入 RAG (Retrieval-Augmented Generation)，实现对“振华宇宙”万余个细节的精准知识检索。

---

## ⚠️ Disclaimer / 免责声明
1.  **版权声明**：本项目所使用的训练语料版权归原作者八月长安及其所属出版机构所有。
2.  **用途限制**：本项目仅供自然语言处理（NLP）学术研究与技术交流使用，**严禁任何形式的商业用途、大规模传播或出版行为**。
3.  **风险提示**：模型输出基于概率生成，可能包含不可控的幻觉（Hallucination），不代表项目作者立场，亦不代表原作者的真实创作意图。

---

## 🤝 Acknowledgement / 致谢
- 感谢 **八月长安** 创作了温暖无数人的振华故事。
- 感谢 **Alibaba Qwen Team** 提供强大的基座模型支持。
- 感谢 **LLaMA-Factory** 提供的微调框架支持。
