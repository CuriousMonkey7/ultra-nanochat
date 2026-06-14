# Ultra Nano Chat: The LLM Training Pipeline

> **⚠️ WORK IN PROGRESS ⚠️**
> This repository is actively under development. The code and concepts are being refined, so expect rough edges as this project evolves.

## The Motivation: Why This Exists

This project was born out of a desire to look under the hood of modern Large Language Models without getting lost in the engineering.

When looking at incredible projects, Andrej Karpathy's NanoChat, the sheer volume of files, folders, and concurrent processes can feel overwhelming. While training a GPT-2-class model for under $100 is an amazing challenge, that level of optimization often obscures the core educational concepts. Whether you are trying to break these concepts down for a classroom of students or just trying to wrap your own head around the pipeline, sprawling codebases make learning difficult.

Realistically, most of us don't have the hardware to run full-scale models anyway. But if we can gain **90% of the understanding** of how the whole system works, that is incredibly valuable.

The goal here isn't to produce a state-of-the-art model. The goal is to understand exactly how we go from nothing to a working GPT model, with every step written down in a single, readable script.

## Core Philosophy

* **Single-File Visibility:** Every step of the complete training pipeline is contained in one script. You can trace the logic top-to-bottom.
* **Education Over Optimization:** We don't care about the perfect AdamW implementation or building a state-of-the-art custom tokenizer. We just want to learn how the pieces fit together.
* **Tiny Datasets:** We use extremely small, simplified datasets. They are just large enough to demonstrate the concepts in action without requiring massive compute.


## The Three-Stage Pipeline

This script walks through the three foundational stages of modern LLM training, running them end-to-end:

1. **Pre-training:** The model learns general language patterns by predicting the next token using raw text.
2. **Supervised Fine-Tuning (SFT):** The base model is aligned to follow conversational patterns and instruction-response formats.
3. **Reinforcement Learning (RL):** The model is optimized against task-specific rewards (like verifying math problem answers) to encourage correct reasoning.

---

## Quick Start

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/CuriousMonkey7/ultra-nanochat.git
cd ultra-nanochat
uv sync
```

### Running the Pipeline

Because the entire pipeline is consolidated, you only need to execute the main script. The script will automatically sequence through Pre-training, SFT, and RL.

```bash
python ultra_nanochat.py

```
