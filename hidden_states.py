import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
HF_DATASET          = "declare-lab/CategoricalHarmfulQA"
POST_INST_DELIMITER = "<|im_end|>\n<|im_start|>assistant"

# ── Formatting ────────────────────────────────────────────────────────────────
def format_no_system(instruction: str) -> str:
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant"


# ── Batched tokenization with LEFT padding ────────────────────────────────────
def tokenize_batch(instructions: list[str], tokenizer) -> dict:
    """
    Left padding performed
    """
    formatted = [format_no_system(inst) for inst in instructions]
    return tokenizer(
        formatted,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,  # template already adds special tokens
    )


# ── Hook-based activation extraction ─────────────────────────────────────────
def extract_activations(
    model,
    inputs: dict,
    positions: list[int],
) -> torch.Tensor:
    """
    Run a forward pass and collect residual stream at given relative positions
    for every layer.
    """
    n_layers   = model.config.num_hidden_layers
    batch_size = inputs["input_ids"].shape[0]
    seq_len    = inputs["input_ids"].shape[1]

    # Convert relative positions to absolute
    abs_positions = [seq_len + p for p in positions]

    cache = {} 

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden = output[0]  # [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            cache[layer_idx] = hidden[:, abs_positions, :].detach().cpu().half()
        return hook_fn

    handles = [
        layer.register_forward_hook(make_hook(i))
        for i, layer in enumerate(model.model.layers)
    ]

    inputs_cuda = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        model(**inputs_cuda)

    for h in handles:
        h.remove()

    # Stack: [n_layers, batch, n_positions, hidden_dim]
    stacked = torch.stack([cache[i] for i in range(n_layers)], dim=0)
    # Permute: [batch, n_layers, n_positions, hidden_dim]
    return stacked.permute(1, 0, 2, 3)


# ── Main extraction loop ──────────────────────────────────────────────────────
def extract_for_category(
    category: str,
    questions: list[str],
    model,
    tokenizer,
    n_delim: int,
    batch_size: int,
    output_dir: str,
):
    all_activations = []
    t_inst_rel      = -(n_delim + 1)  # last token of user instruction
    t_post_inst_rel = -1               # last token of post-instruction delimiter
    positions = [t_inst_rel, t_post_inst_rel]

    for i in tqdm(range(0, len(questions), batch_size), desc=category):
        batch_questions = questions[i : i + batch_size]
        inputs = tokenize_batch(batch_questions, tokenizer)
        if i == 0:
            seq_len = inputs["input_ids"].shape[1]
            abs_t_inst     = seq_len + t_inst_rel
            abs_t_post     = seq_len + t_post_inst_rel
            t_inst_token   = tokenizer.decode([inputs["input_ids"][0, abs_t_inst].item()])
            t_post_token   = tokenizer.decode([inputs["input_ids"][0, abs_t_post].item()])
            print(f"  [sanity] seq_len={seq_len}  "
                  f"t_inst={repr(t_inst_token)} (idx {abs_t_inst})  "
                  f"t_post-inst={repr(t_post_token)} (idx {abs_t_post})")

        acts = extract_activations(model, inputs, positions)
        all_activations.append(acts)
    all_activations = torch.cat(all_activations, dim=0)

    safe_name = category.replace(" ", "_").replace("/", "_")
    out_path  = os.path.join(output_dir, f"activations_{safe_name}.pt")
    torch.save({"harmful": all_activations, "questions": questions}, out_path)
    print(f"  Saved {all_activations.shape} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", default="activations/")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--dry_run",    action="store_true",
                        help="Run on first 10 examples of each category only")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"                 # critical — do not remove
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # Qwen needs this

    delim_ids = tokenizer(
        POST_INST_DELIMITER,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids[0]
    n_delim = len(delim_ids)
    print(f"Delimiter length: {n_delim} tokens")

    print(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"  n_layers={model.config.num_hidden_layers}  "
          f"hidden_dim={model.config.hidden_size}\n")

    print("Loading CategoricalHarmfulQA...")
    dataset = load_dataset(HF_DATASET, split="en")

    categories = {}
    for example in dataset:
        cat = example["Category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(example["Question"])

    print(f"Categories ({len(categories)} total):")
    for cat, qs in sorted(categories.items()):
        print(f"  {cat}: {len(qs)} examples")
    print()

    for category, questions in sorted(categories.items()):
        if args.dry_run:
            questions = questions[:10]
            print(f"\n[dry_run] {category} — using first {len(questions)} examples")
        else:
            print(f"\nExtracting: {category} ({len(questions)} examples)")

        extract_for_category(
            category    = category,
            questions   = questions,
            model       = model,
            tokenizer   = tokenizer,
            n_delim     = n_delim,
            batch_size  = args.batch_size,
            output_dir  = args.output_dir,
        )

    print("\nDone. All categories extracted.")


if __name__ == "__main__":
    main()