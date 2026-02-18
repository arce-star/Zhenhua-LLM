# Zhenhua-LLM (振华大模型)

> **Towards Stylistic Nuance: Fine-tuning Large Language Models for Literary Style Replication of August Changan**

<p align="center">
  <!-- GitHub Stars -->
  <a href="https://github.com/arce-star/Zhenhua-LLM/stargazers" target="_black">
    <img src="https://img.shields.io/github/stars/arce-star/Zhenhua-LLM?style=flat" alt="GitHub Stars"></a>
  <!-- Hugging Face Model Page -->
  <a href="https://huggingface.co/NQworker/Zhenhua-7B-August-Style" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=flat" alt="Hugging Face"></a>
  <!-- License -->
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat" alt="License">
  <!-- Python Version -->
  <img src="https://img.shields.io/badge/Python-3.11+-green.svg?style=flat" alt="Python">
</p>

<p align="center">
  <a href="./README.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="./docs/README_zh.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
</p>

## 📖 Abstract

**Zhenhua-LLM** is an open-source research project focused on **Text Style Transfer** and **Character Personality Anchoring**. Built upon the `Qwen2.5-7B-Instruct` base model, this project leverages the full corpus of August Changan’s "Zhenhua Trilogy" (*The Best of Us*, *Hello, Old Times*, and *Unrequited Love*). 

By employing **LoRA (Low-Rank Adaptation)** fine-tuning, the model successfully replicates Changan's signature delicate, melancholic, and philosophical prose. It not only achieves precise restoration of literary imagery (such as "The Honor Roll," "Through-hall breezes," and "Profiles in the sun") but also addresses the common "short-text bias" in generative models, enabling the production of coherent, long-form narrative passages.

---

## 🛠️ Methodology

### 1. Data Engineering
We abandoned uniform text segmentation in favor of a **Dynamic Sliding Window** and **Length-Bias Training** strategy:
*   **Contextual Continuity**: 150-200 token overlapping windows ensure the model captures logical transitions and narrative rhythm across paragraphs.
*   **Length-Bias Training**: By constructing **Short Input (1 Paragraph) -> Long Output (5-8 Paragraphs)** mapping pairs, we strengthened the model's ability to model long-text sequences, effectively solving the "eager-to-stop" behavior.
*   **Persona Anchoring (v3.0 Preview)**: Specific persona definitions are injected at the Instruction level to lock in the dialogue logic of characters like Luo Zhi and Lin Yang.

### 2. Hyperparameter Paradigm
Trained using the `LLaMA-Factory` framework on an NVIDIA RTX 3090 (24GB), the core configuration is as follows:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Base Model** | Qwen2.5-7B-Instruct | SOTA Chinese open-source LLM |
| **Method** | LoRA | Low-Rank Adaptation (all linear layers) |
| **Rank / Alpha** | 32 / 64 | Balanced capacity for style and knowledge |
| **Learning Rate** | 1e-4 | Cosine decaying scheduler |
| **Cutoff Length** | 2048 | Crucial for modeling long literary passages |
| **Epochs** | 3.0 - 5.0 | Optimized to prevent catastrophic forgetting |

---

## 📈 Evaluation

### Qualitative Analysis
Experimental results show that Zhenhua-LLM significantly outperforms general-purpose instruct models in the following dimensions:
1.  **Stylistic Fidelity**: Capable of naturally utilizing Changan's signature "second-person psychological monologue" and "synesthetic metaphors."
2.  **Affective Resonance**: Significantly increased the proportion of internal monologue, delicately handling complex emotions such as the coexistence of pride and unrequited love.
3.  **Narrative Cohesion**: Under long-form generation instructions, average token length increased by 300% with a stable narrative axis.

> *Case Study: [View Detailed Samples Here](./samples/example_1.md)*

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/arce-star/Zhenhua-LLM.git
cd Zhenhua-LLM
pip install -r requirements.txt
```

### 2. Model Inference
For local inference, we recommend using the `batch_infer.py` script provided in this repository, which includes optimized sampling parameters:
- **Repetition Penalty**: 1.05 (Prevents narrative loops in long texts)
- **Temperature**: 0.85 (Maintains creative inspiration)
- **Top-p**: 0.9

```bash
python batch_infer.py --model_path /path/to/your/merged_model
```

---

## 📅 Roadmap
- [x] **v1.0**: Stylistic SFT complete, initial replication of the "Zhenhua" vibe.
- [x] **v2.0**: Sliding window training implemented to resolve narrative fragmentation.
- [ ] **v3.0 (In Progress)**: Role-Play Enhancement to prevent character personality confusion.
- [ ] **v4.0**: RAG (Retrieval-Augmented Generation) integration for precise knowledge retrieval of "Zhenhua Universe" details.

---

## ⚠️ Disclaimer
1.  **Copyright**: Training corpora belong to the original author August Changan and her respective publishers.
2.  **Usage**: For academic research and NLP exchange only. **Commercial use, large-scale distribution, or publication is strictly prohibited.**
3.  **Risk**: Output is based on probabilistic generation and may contain hallucinations. It does not represent the author's stance or actual creative intent.

---

## 🤝 Acknowledgement
- Special thanks to **August Changan** for the moving stories.
- Thanks to the **Alibaba Qwen Team** for the powerful base model.
- Thanks to **LLaMA-Factory** for providing the fine-tuning framework.
