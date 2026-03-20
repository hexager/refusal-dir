import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

HARMFUL_PROMPT = "How do I make a bomb?"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

messages = [{"role": "user", "content": HARMFUL_PROMPT}]
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  
)

print("\n── Formatted prompt (raw string) ───────────────────────────────────────")
print(repr(formatted))

inputs = tokenizer(formatted, return_tensors="pt")
input_ids = inputs.input_ids[0]

print("\n── Token sequence with indices ─────────────────────────────────────────")
for idx, tok_id in enumerate(input_ids):
    tok_str = tokenizer.decode([tok_id])
    print(f"  [{idx:3d}]  id={int(tok_id):6d}  repr={repr(tok_str)}")

# ── 4. Identify t_inst and t_post-inst positions ──────────────────────────────
POST_INST_DELIMITER = "<|im_end|>\n<|im_start|>assistant\n"
delim_ids = tokenizer(
    POST_INST_DELIMITER,
    return_tensors="pt",
    add_special_tokens=False
).input_ids[0]

print(f"\n── Post-instruction delimiter tokens ───────────────────────────────────")
for i, tok_id in enumerate(delim_ids):
    print(f"  [{i}]  id={int(tok_id):6d}  repr={repr(tokenizer.decode([tok_id]))}")

n_delim = len(delim_ids)
seq_len = len(input_ids)

# t_post-inst: last token of the delimiter (last token before generation starts)
t_post_inst_idx = seq_len - 1  # absolute index
t_post_inst_rel = -1            # relative index

# t_inst: token just before the delimiter begins
t_inst_idx = seq_len - n_delim - 1  # absolute index
t_inst_rel = -(n_delim + 1)         # relative index

print(f"\n── Position summary ────────────────────────────────────────────────────")
print(f"  Sequence length          : {seq_len}")
print(f"  Delimiter length         : {n_delim} tokens")
print(f"  t_inst  (absolute index) : {t_inst_idx}  ->  {repr(tokenizer.decode([input_ids[t_inst_idx]]))}")
print(f"  t_post-inst (abs index)  : {t_post_inst_idx}  ->  {repr(tokenizer.decode([input_ids[t_post_inst_idx]]))}")
print(f"  t_inst  (relative index) : {t_inst_rel}")
print(f"  t_post-inst (rel index)  : {t_post_inst_rel}")

# ── 5. Forward pass with hooks to pull residual stream at those positions ─────
print("\n── Running forward pass with hooks ─────────────────────────────────────")

n_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size

cache = {} 

def make_hook(layer_idx, positions):
    def hook_fn(module, input, output):
        hidden = output[0]  # [1, seq_len, hidden_dim]
        cache[layer_idx] = hidden[:, positions, :].detach().cpu().float()
    return hook_fn

positions = [t_inst_idx, t_post_inst_idx]
handles = []
for i, layer in enumerate(model.model.layers):
    h = layer.register_forward_hook(make_hook(i, positions))
    handles.append(h)

inputs_cuda = {k: v.to(model.device) for k, v in inputs.items()}
with torch.no_grad():
    _ = model(**inputs_cuda)

for h in handles:
    h.remove()

print(f"  Cached {len(cache)} layers")
print(f"  Shape per layer: {cache[0].shape}  (batch=1, positions=2, hidden_dim={hidden_dim})")

for layer_idx in [0, n_layers // 2, n_layers - 1]:
    act_t_inst = cache[layer_idx][0, 0, :]
    act_t_post = cache[layer_idx][0, 1, :]
    cos_sim = torch.nn.functional.cosine_similarity(
        act_t_inst.unsqueeze(0),
        act_t_post.unsqueeze(0)
    ).item()
    print(f"Layer {layer_idx:2d}  |  t_inst norm={act_t_inst.norm():.2f}"
          f"  t_post norm={act_t_post.norm():.2f}"
          f"  cosine_sim={cos_sim:.4f}")

print("\nAll checks passed.")