"""
Complete LLM Training Pipeline: Pretraining → SFT → RL

This educational script demonstrates the three stages of modern LLM training:
1. PRETRAINING: Learn general language patterns from raw text
2. SFT (Supervised Fine-Tuning): Align model to follow conversational patterns
3. RL (Reinforcement Learning): Optimize for task-specific rewards (math problem solving)

All stages fit in one simple script for educational clarity.
"""

import os
import copy
import glob
import pickle
import random
import re
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

import pyarrow.parquet as pq

import rustbpe
import tiktoken

from dotenv import load_dotenv

load_dotenv()


# Configuration for all training stages
@dataclass
class Config:
    # Batch sizes
    batch_size = 16
    micro_batch_size = 4
    seq_len = 1024

    # Model architecture
    vocab_size = 2**15
    n_layers = 12
    n_heads = 12
    n_embed = 768

    # Optimization
    lr_pretrain = 3e-4
    lr_sft = 1e-4
    lr_rl = 5e-5

    # Training stages
    pretrain_steps = 10
    sft_steps = 10
    rl_steps = 5

    # Precision and device
    precision = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # RL-specific
    rl_num_samples = 4
    special_tokens_list = [
        "<|bos|>",
        "<|eos|>",
        "<|user_start|>",
        "<|user_end|>",
        "<|assistant_start|>",
        "<|assistant_end|>",
        "<|python_start|>",
        "<|python_end|>",
        "<|output_start|>",
        "<|output_end|>",
    ]


def seed(seed=42):
    random.seed(42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class DataLoaderLite:
    def __init__(self, B, T, tokenizer, split="train", bos_token="<|bos|>"):
        self.B = B
        self.T = T
        self.counter = 0

        parquet_files = glob.glob("data/*.parquet")
        texts = ["Fallback text for testing. " * 100] * 100  # Fallback

        if len(parquet_files) > 0:
            texts = []
            for file in parquet_files[:20]:
                table = pq.ParquetFile(file).read()
                texts += table.column("text").to_pylist()[:200]

        # --- CREATE THE SPLIT (90% Train, 10% Val) ---
        split_idx = int(len(texts) * 0.9)
        self.texts = texts[:split_idx] if split == "train" else texts[split_idx:]

        self.token_list = [tokenizer.encode(text)[: self.T] for text in self.texts]
        self.remaining_token_list = self.token_list.copy()
        self.bos_token = tokenizer.encode(bos_token, allowed_special={"<|bos|>"})

    def get_batch(self):
        batch_tokens = []
        for _ in range(self.B):
            remaining_batch = self.T + 1
            row = []
            while remaining_batch > 0:
                if len(self.remaining_token_list) == 0:
                    self.remaining_token_list = self.token_list.copy()
                    random.shuffle(self.remaining_token_list)
                valid = [
                    x for x in self.remaining_token_list if len(x) < remaining_batch
                ]
                if not valid:
                    row += self.bos_token
                    remaining_batch -= len(self.bos_token)
                    row += [-1] * remaining_batch
                    break

                longest_tokens = max(valid, key=len)
                self.remaining_token_list.remove(longest_tokens)
                row += self.bos_token + longest_tokens
                remaining_batch -= len(longest_tokens) + len(self.bos_token)
            batch_tokens += row

        batch_tokens = torch.tensor(batch_tokens, dtype=torch.long).reshape(
            self.B, self.T + 1
        )
        y = batch_tokens[:, 1:].clone()
        x = batch_tokens[:, :-1].clone()
        x[x == -1] = self.bos_token[0]
        return x.to(Config.device), y.to(Config.device)

    def get_eval_prompt(self):
        """Helper to grab a random sentence starter and its continuation from the val set."""
        text = self.texts[0]
        # text = random.choice(self.texts)
        words = text.split()

        # Ensure we have enough words
        if len(words) < 20:
            words = (text + " " + text).split()  # quick hack to pad short sentences

        split_idx = random.randint(5, 15)

        # The prompt is the first N words
        prompt_text = " ".join(words[:split_idx])
        # The expected output is the next 15 words
        expected_text = " ".join(words[split_idx : split_idx + 15])

        return prompt_text, expected_text


class CausalAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(Config.n_embed, 3 * Config.n_embed, bias=False)
        self.proj = nn.Linear(Config.n_embed, Config.n_embed)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(Config.n_embed, dim=-1)
        head_size = C // Config.n_heads
        q = q.view(B, T, Config.n_heads, head_size).transpose(1, 2)
        k = k.view(B, T, Config.n_heads, head_size).transpose(1, 2)
        v = v.view(B, T, Config.n_heads, head_size).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return out


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffwd = nn.Linear(Config.n_embed, 4 * Config.n_embed)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * Config.n_embed, Config.n_embed)

    def forward(self, x):
        return self.proj(self.gelu(self.ffwd(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = CausalAttn()
        self.mlp = MLP()
        self.ln_1 = nn.LayerNorm(Config.n_embed)
        self.ln_2 = nn.LayerNorm(Config.n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(Config.vocab_size, Config.n_embed)
        self.wpe = nn.Embedding(Config.seq_len, Config.n_embed)
        self.blocks = nn.ModuleList(Block() for _ in range(Config.n_layers))
        self.ln_f = nn.LayerNorm(Config.n_embed)
        self.lm_head = nn.Linear(Config.n_embed, Config.vocab_size, bias=False)

        self.register_buffer("positions", torch.arange(Config.seq_len))
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / (2 * Config.n_layers) ** 0.5
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        token_emb = self.wte(x)
        pos_tokens = self.positions[:T]
        pos_emb = self.wpe(pos_tokens)
        x = token_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


class Tokenizer:
    """TRAIN, SAVE LOAD tokenizer using rustbpe and tiktoken."""

    def train_from_iterator(iterator, vocab_size, special_tokens, pattern):
        vocab_size_no_special = vocab_size - len(special_tokens)
        tokenizer = rustbpe.Tokenizer()
        tokenizer.train_from_iterator(
            iterator, vocab_size=vocab_size_no_special, pattern=pattern
        )

        tokenizer = tiktoken.Encoding(
            name="my_tokenizer",
            pat_str=tokenizer.get_pattern(),
            mergeable_ranks={bytes(k): v for k, v in tokenizer.get_mergeable_ranks()},
            special_tokens=special_tokens,
        )
        return tokenizer

    def save_tokenizer(tokenizer, path):
        tokenizer_path = f"{path}/tokenizer.pkl"
        with open(tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)

    def load_tokenizer(path):
        tokenizer_path = f"{path}/tokenizer.pkl"
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def run_evaluation(
    model, tokenizer, stage_name, step, total_steps, prompt_text, expected_output
):
    print("\n" + "=" * 50)
    print(f"🔍 EVALUATION: {stage_name} [Step {step + 1}/{total_steps}]")
    print("=" * 50)
    model.eval()

    if stage_name == "PRETRAINING":
        prompt_tokens = tokenizer.encode(prompt_text)
        print(f"Input Prompt (Raw): {repr(prompt_text)}")
    else:
        prompt_tokens = [
            Config.new_special_tokens["<|bos|>"],
            Config.new_special_tokens["<|user_start|>"],
            *tokenizer.encode(prompt_text),
            Config.new_special_tokens["<|user_end|>"],
            Config.new_special_tokens["<|assistant_start|>"],
        ]
        print(f"Input Prompt (Chat): {repr(prompt_text)}")

    # --- PRINT THE EXPECTED OUTPUT HERE ---
    print(f"Ground Truth/Expected: {repr(expected_output)}")
    print("-" * 50)

    with torch.no_grad():
        generated_tokens = generate_response(
            model, tokenizer, prompt_tokens, max_new_tokens=30, temperature=0.7
        )

    new_tokens = generated_tokens[len(prompt_tokens) :]
    print(f"Raw Output IDs: {new_tokens}")

    try:
        decoded_output = tokenizer.decode(new_tokens)
        print(f"Model Predicted Text: {repr(decoded_output)}")
    except Exception as e:
        print(f"Model Predicted Text: [Decode Error: {e}]")

    print("=" * 50 + "\n")
    model.train()


def render_conversation(conversation, tokenizer, max_tokens=1024):
    """
    Tokenize a single Chat conversation with structured tags and tool-use support.
    """
    ids, mask = [], []

    def add_tokens(token_ids, mask_val):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))

    # Handle system prompt merging
    if conversation["messages"][0]["role"] == "system":
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[1]["role"] == "user", (
            "System message must be followed by a user message"
        )
        messages[1]["content"] = (
            messages[0]["content"] + "\n\n" + messages[1]["content"]
        )
        messages = messages[1:]
    else:
        messages = conversation["messages"]

    assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

    # Fetch special tokens directly from our dictionary
    bos = Config.new_special_tokens["<|bos|>"]
    user_start, user_end = (
        Config.new_special_tokens["<|user_start|>"],
        Config.new_special_tokens["<|user_end|>"],
    )
    assistant_start, assistant_end = (
        Config.new_special_tokens["<|assistant_start|>"],
        Config.new_special_tokens["<|assistant_end|>"],
    )
    python_start, python_end = (
        Config.new_special_tokens["<|python_start|>"],
        Config.new_special_tokens["<|python_end|>"],
    )
    output_start, output_end = (
        Config.new_special_tokens["<|output_start|>"],
        Config.new_special_tokens["<|output_end|>"],
    )

    # Build the token sequence
    add_tokens(bos, 0)
    for i, message in enumerate(messages):
        must_be_from = "user" if i % 2 == 0 else "assistant"
        assert message["role"] == must_be_from, (
            f"Message {i} is from {message['role']} but should be from {must_be_from}"
        )

        content = message["content"]

        if message["role"] == "user":
            assert isinstance(content, str), (
                "User messages are simply expected to be strings"
            )
            value_ids = tokenizer.encode(content)
            add_tokens(user_start, 0)
            add_tokens(value_ids, 0)
            add_tokens(user_end, 0)

        elif message["role"] == "assistant":
            add_tokens(assistant_start, 0)

            if isinstance(content, str):
                value_ids = tokenizer.encode(content)
                add_tokens(value_ids, 1)  # Mask=1 (Train on this!)

            elif isinstance(content, list):
                for part in content:
                    value_ids = tokenizer.encode(part["text"])
                    if part["type"] == "text":
                        add_tokens(value_ids, 1)
                    elif part["type"] == "python":
                        add_tokens(python_start, 1)
                        add_tokens(value_ids, 1)
                        add_tokens(python_end, 1)
                    elif part["type"] == "python_output":
                        # Output comes from the environment, so model shouldn't train to predict it
                        add_tokens(output_start, 0)
                        add_tokens(value_ids, 0)
                        add_tokens(output_end, 0)
                    else:
                        raise ValueError(f"Unknown part type: {part['type']}")
            else:
                raise ValueError(f"Unknown content type: {type(content)}")

            add_tokens(assistant_end, 1)  # Train the model to output the end token

    # truncate to max_tokens
    ids = ids[:max_tokens]
    mask = mask[:max_tokens]
    return ids, mask


def load_sft_batch(dataset, batch_size, tokenizer, max_tokens=1024):
    """Load SFT batch with ignored indices for prompts/padding."""
    batch_tokens, batch_masks = [], []

    for _ in range(batch_size):
        idx = random.randint(0, len(dataset) - 1)
        tokens, mask = render_conversation(dataset[idx], tokenizer, max_tokens)

        padded_tokens = tokens + [
            tokenizer.encode("<|eos|>", allowed_special={"<|eos|>"})[0]
        ] * (max_tokens - len(tokens))
        padded_mask = mask + [0] * (max_tokens - len(mask))

        batch_tokens.append(padded_tokens[:max_tokens])
        batch_masks.append(padded_mask[:max_tokens])

    batch_tensor = torch.tensor(batch_tokens, dtype=torch.long)
    inputs = batch_tensor[:, :-1].to(Config.device)
    targets = batch_tensor[:, 1:].clone().to(Config.device)

    mask_tensor = torch.tensor(batch_masks, dtype=torch.long)
    mask_targets = mask_tensor[:, 1:].to(Config.device)
    targets[mask_targets == 0] = -1  # -1 is ignore_index in CrossEntropyLoss

    return inputs, targets


def generate_response(
    model, tokenizer, prompt_tokens, max_new_tokens=100, temperature=0.7
):
    """Auto-regressive generation for RL sampling."""
    model.eval()
    with torch.no_grad():
        current_tokens = prompt_tokens.copy()
        for _ in range(max_new_tokens):
            if len(current_tokens) >= Config.seq_len:
                break

            x = torch.tensor([current_tokens], dtype=torch.long).to(Config.device)
            logits = model(x)
            next_logits = logits[0, -1, :]

            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            current_tokens.append(next_token)

            if (
                next_token
                == tokenizer.encode("<|eos|>", allowed_special={"<|eos|>"})[0]
            ):
                break
    model.train()
    return current_tokens


def extract_gsm8k_answer(text):
    match = re.search(r"#### (\-?[0-9\.\,]+)", text)
    if match:
        return match.group(1).strip().replace(",", "")
    return None


def evaluate_gsm8k_response(ground_truth_conversation, generated_response_text):
    asst_msg = ground_truth_conversation["messages"][-1]
    truth_text = (
        " ".join([p.get("text", "") for p in asst_msg["content"]])
        if isinstance(asst_msg["content"], list)
        else asst_msg["content"]
    )

    truth_answer = extract_gsm8k_answer(truth_text)
    pred_answer = extract_gsm8k_answer(generated_response_text)
    if truth_answer and pred_answer and truth_answer == pred_answer:
        return 1.0
    return 0.0


def get_tokenizer_iterator():
    """Use one data shard to train the tokenizer, yielding one text at a time."""
    parquet_files = glob.glob("data/*.parquet")
    assert len(parquet_files) > 0, (
        "No parquet files found in data/ directory for tokenizer training!"
    )
    for file in parquet_files[:1]:
        table = pq.ParquetFile(file).read()
        for text in table.column("text").to_pylist():
            yield text


run_dir = "runs/temp_run"
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
new_special_tokens = {
    token: (Config.vocab_size - len(Config.special_tokens_list)) + idx
    for idx, token in enumerate(Config.special_tokens_list)
}
Config.new_special_tokens = new_special_tokens

if os.path.exists(f"{run_dir}/tokenizer.pkl"):
    print("Loading existing tokenizer...")
    tokenizer = Tokenizer.load_tokenizer(run_dir)
else:
    print("Training new tokenizer...")
    import time

    t0 = time.time()
    tokenizer = Tokenizer.train_from_iterator(
        get_tokenizer_iterator(),
        Config.vocab_size,
        Config.new_special_tokens,
        SPLIT_PATTERN,
    )
    t1 = time.time()
    print(f"Tokenizer trained in {(t1 - t0) / 60:.2f} minutes.")
    Tokenizer.save_tokenizer(tokenizer, run_dir)

# ============================================================================
# INITIALIZE MODEL, DATALOADER AND OPTIMIZERS
# ============================================================================
org_model = GPTModel().to(Config.device)
# Skipping compilation for compatibility/educational simplicity
model = org_model
# print number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Model Parameters: {total_params / 1e6:.2f}M")
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
grad_accum_steps = Config.batch_size // Config.micro_batch_size

seed()
torch.set_float32_matmul_precision("high")

dataloader = DataLoaderLite(
    B=Config.micro_batch_size, T=Config.seq_len, tokenizer=tokenizer
)
val_dataloader = DataLoaderLite(
    B=Config.micro_batch_size, T=Config.seq_len, tokenizer=tokenizer, split="val"
)
# ============================================================================
# STAGE 1: PRETRAINING
# ============================================================================
print("\nSTAGE 1: PRETRAINING")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr_pretrain)

for step in range(Config.pretrain_steps):
    optimizer.zero_grad()
    for _ in range(grad_accum_steps):
        x, y = dataloader.get_batch()
        with torch.autocast(device_type=Config.device, dtype=Config.precision):
            logits = model(x)
            loss = (
                loss_fn(logits.reshape(-1, Config.vocab_size), y.view(-1))
                / grad_accum_steps
            )
        loss.backward()
    optimizer.step()
    print(
        f"Pretrain Step {step + 1}/{Config.pretrain_steps} | Loss: {loss.item() * grad_accum_steps:.4f}"
    )
    if (step + 1) % max(1, Config.pretrain_steps // 10) == 0:
        input_prompt, expected_output = val_dataloader.get_eval_prompt()
        run_evaluation(
            model,
            tokenizer,
            "PRETRAINING",
            step,
            Config.pretrain_steps,
            input_prompt,
            expected_output,
        )
# ============================================================================
# STAGE 2: SUPERVISED FINE-TUNING (SFT)
# ============================================================================
print("\nSTAGE 2: SUPERVISED FINE-TUNING (SFT)")
try:
    from datasets import load_dataset

    print("Loading SmolTalk dataset...")
    # Train split
    sft_dataset_train = load_dataset(
        "HuggingFaceTB/smol-smoltalk", split="train", streaming=True
    ).take(1000)
    sft_data_list = [{"messages": ex["messages"]} for ex in sft_dataset_train]

    # Validation split
    sft_dataset_val = load_dataset(
        "HuggingFaceTB/smol-smoltalk", split="test", streaming=True
    ).take(100)
    sft_val_list = [{"messages": ex["messages"]} for ex in sft_dataset_val]
    print(f"Loaded {len(sft_data_list)} train, {len(sft_val_list)} val conversations")
except Exception as e:
    print(f"Failed to load SmolTalk: {e}. Falling back to synthetic data.")
    sft_data_list = [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "It is 4."},
            ]
        }
    ] * 100

print(sft_data_list[0])

optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr_sft)

for step in range(Config.sft_steps):
    optimizer.zero_grad()
    for _ in range(grad_accum_steps):
        # Removed try/except block to expose shape/memory errors properly
        inputs, targets = load_sft_batch(
            sft_data_list, Config.micro_batch_size, tokenizer, Config.seq_len
        )
        with torch.autocast(device_type=Config.device, dtype=Config.precision):
            logits = model(inputs)
            loss = (
                loss_fn(logits.reshape(-1, Config.vocab_size), targets.view(-1))
                / grad_accum_steps
            )
        loss.backward()
    optimizer.step()
    print(
        f"SFT Step {step + 1}/{Config.sft_steps} | Loss: {loss.item() * grad_accum_steps:.4f}"
    )
    if (step + 1) % max(1, Config.sft_steps // 10) == 0:
        # Pick a random conversation from the unseen validation split
        eval_convo = sft_val_list[0]

        # Extract the user's first message to use as the prompt
        input_prompt = eval_convo["messages"][0]["content"]
        expected_output = eval_convo["messages"][1]["content"]

        run_evaluation(
            model,
            tokenizer,
            "SFT",
            step,
            Config.sft_steps,
            input_prompt,
            expected_output,
        )

# ============================================================================
# STAGE 3: REINFORCEMENT LEARNING (RL)
# ============================================================================
print("\nSTAGE 3: REINFORCEMENT LEARNING (RL)")

try:
    from datasets import load_dataset

    print("Loading GSM8K dataset...")
    # Train split
    rl_dataset_raw = load_dataset(
        "openai/gsm8k", "main", split="train", streaming=True
    ).take(500)
    rl_data_list = [
        {
            "question": ex["question"],
            "answer": ex["answer"],
            "messages": [
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]},
            ],
        }
        for ex in rl_dataset_raw
    ]

    rl_dataset_test = load_dataset(
        "openai/gsm8k", "main", split="test", streaming=True
    ).take(50)
    rl_val_list = [
        {"question": ex["question"], "answer": ex["answer"]} for ex in rl_dataset_test
    ]

    print(f"Loaded {len(rl_data_list)} train, {len(rl_val_list)} test math problems")
except Exception as e:
    print(f"Failed to load GSM8K: {e}. Falling back to synthetic data.")
    rl_data_list = [
        {
            "question": "2 + 2 =",
            "answer": "2 + 2 = 4. #### 4",
            "messages": [
                {"role": "user", "content": "2 + 2 ="},
                {"role": "assistant", "content": "2 + 2 = 4. #### 4"},
            ],
        }
    ] * 100

print(rl_data_list[0])
# import sys; sys.exit()
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr_rl)


for step in range(Config.rl_steps):
    total_reward = 0.0
    optimizer.zero_grad()

    for _ in range(Config.micro_batch_size):
        problem = rl_data_list[random.randint(0, len(rl_data_list) - 1)]
        question_tokens = tokenizer.encode(problem["question"])

        for _ in range(Config.rl_num_samples):
            generated_tokens = generate_response(
                model, tokenizer, question_tokens, max_new_tokens=20
            )

            try:
                generated_text = tokenizer.decode(
                    generated_tokens[len(question_tokens) :]
                )
            except Exception:
                generated_text = ""

            reward = evaluate_gsm8k_response(problem, generated_text)

            if reward > 0:
                advantage = reward

                # FIXED: Ensure we only compute loss over the GENERATED tokens, not the prompt.
                x = torch.tensor([generated_tokens[:-1]], dtype=torch.long).to(
                    Config.device
                )
                targets = torch.tensor([generated_tokens[1:]], dtype=torch.long).to(
                    Config.device
                )

                # Mask out the prompt tokens in the target
                prompt_len = len(question_tokens)
                if prompt_len > 1:
                    targets[0, : prompt_len - 1] = -1

                with torch.autocast(device_type=Config.device, dtype=Config.precision):
                    logits = model(x)
                    # CrossEntropyLoss returns the Negative Log-Likelihood (NLL)
                    neg_log_probs = loss_fn(
                        logits.reshape(-1, Config.vocab_size), targets.view(-1)
                    )

                    # FIXED: REINFORCE aims to maximize expected reward.
                    # To do this via gradient descent, we MINIMIZE (NLL * Reward).
                    # The previous AI used a negative sign here (-neg_log_probs), which caused gradient ascent!
                    pg_loss = (neg_log_probs * advantage) / (
                        Config.rl_num_samples * Config.micro_batch_size
                    )
                    pg_loss.backward()

            total_reward += reward

    optimizer.step()
    avg_reward = total_reward / (Config.micro_batch_size * Config.rl_num_samples)
    print(f"RL Step {step + 1}/{Config.rl_steps} | Avg Reward: {avg_reward:.4f}")
    if (step + 1) % max(1, Config.rl_steps // 10) == 0:
        # Pick a random unseen math problem
        eval_problem = rl_val_list[0]
        input_prompt = eval_problem["question"]
        expected_output = eval_problem["answer"]
        run_evaluation(
            model, tokenizer, "RL", step, Config.rl_steps, input_prompt, expected_output
        )

print("\nTRAINING PIPELINE COMPLETE!")
