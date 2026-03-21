import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

CATQA_DATASET       = "declare-lab/CategoricalHarmfulQA"
ALPACA_DATASET      = "tatsu-lab/alpaca"
MODEL_ID            = "Qwen/Qwen2.5-1.5B-Instruct"
POST_INST_DELIMITER = "<|im_end|>\n<|im_start|>assistant"   # no trailing \n, matches Zhao et al. Table 4


def format_no_system(instruction: str) -> str:
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant"


def tokenize_batch(instructions: list[str], tokenizer) -> dict:
    formatted = [format_no_system(inst) for inst in instructions]
    return tokenizer(
        formatted,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )


def extract_activations(
    model,
    inputs: dict,
    positions: list[int],
) -> torch.Tensor:
    n_layers  = model.config.num_hidden_layers
    seq_len   = inputs["input_ids"].shape[1]

    abs_positions = [seq_len + p for p in positions]

    cache = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden = output
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
    stacked = torch.stack([cache[i] for i in range(n_layers)], dim=0)
    return stacked.permute(1, 0, 2, 3)

def extract_and_save(
    label: str,
    instructions: list[str],
    out_path: str,
    model,
    tokenizer,
    n_delim: int,
    batch_size: int,
):
    all_activations = []

    t_inst_rel      = -(n_delim + 1)
    t_post_inst_rel = -1
    positions       = [t_inst_rel, t_post_inst_rel]

    for i in tqdm(range(0, len(instructions), batch_size), desc=label):
        batch = instructions[i : i + batch_size]
        inputs = tokenize_batch(batch, tokenizer)

        if i == 0:
            seq_len      = inputs["input_ids"].shape[1]
            abs_t_inst   = seq_len + t_inst_rel
            abs_t_post   = seq_len + t_post_inst_rel
            t_inst_tok   = tokenizer.decode([inputs["input_ids"][0, abs_t_inst].item()])
            t_post_tok   = tokenizer.decode([inputs["input_ids"][0, abs_t_post].item()])
            print(f"  [sanity] seq_len={seq_len}  "
                  f"t_inst={repr(t_inst_tok)} (idx {abs_t_inst})  "
                  f"t_post-inst={repr(t_post_tok)} (idx {abs_t_post})")

        acts = extract_activations(model, inputs, positions)
        all_activations.append(acts)

    all_activations = torch.cat(all_activations, dim=0)
    print(f"  final shape={all_activations.shape}")

    torch.save({"activations": all_activations, "instructions": instructions}, out_path)
    print(f"  Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=MODEL_ID)
    parser.add_argument("--output_dir", default="activations/")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--harmless",   action="store_true",
                        help="Extract harmless baseline from ALPACA instead of CATQA")
    parser.add_argument("--n_harmless", default=550, type=int,
                        help="Number of ALPACA examples to use (default 550, matches total CATQA size)")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Run on first 10 examples only")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    delim_ids = tokenizer(
        POST_INST_DELIMITER, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]
    n_delim = len(delim_ids)
    print(f"Delimiter length: {n_delim} tokens")
    print(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print(f"  n_layers={model.config.num_hidden_layers}  hidden_dim={model.config.hidden_size}\n")
    if args.harmless:
        print(f"Loading ALPACA (harmless baseline)...")
        alpaca = load_dataset(ALPACA_DATASET, split="train")

        # Use 'instruction' field, skip examples with empty instructions
        instructions = [
            ex["instruction"] for ex in alpaca
            if ex["instruction"].strip()
        ][:args.n_harmless]

        if args.dry_run:
            instructions = instructions[:10]
            print(f"[dry_run] using first {len(instructions)} ALPACA examples")
        else:
            print(f"Using {len(instructions)} ALPACA examples")

        out_path = os.path.join(args.output_dir, "activations_harmless.pt")
        extract_and_save(
            label        = "harmless (ALPACA)",
            instructions = instructions,
            out_path     = out_path,
            model        = model,
            tokenizer    = tokenizer,
            n_delim      = n_delim,
            batch_size   = args.batch_size,
        )
    else:
        print("Loading CategoricalHarmfulQA...")
        dataset = load_dataset(CATQA_DATASET, split="en")

        categories = {}
        for ex in dataset:
            cat = ex["Category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(ex["Question"])

        print(f"Categories ({len(categories)} total):")
        for cat, qs in sorted(categories.items()):
            print(f"  {cat}: {len(qs)} examples")
        print()

        for category, questions in sorted(categories.items()):
            if args.dry_run:
                questions = questions[:10]
                print(f"\n[dry_run] {category} — {len(questions)} examples")
            else:
                print(f"\nExtracting: {category} ({len(questions)} examples)")

            safe_name = category.replace(" ", "_").replace("/", "_")
            out_path  = os.path.join(args.output_dir, f"activations_{safe_name}.pt")

            extract_and_save(
                label        = category,
                instructions = questions,
                out_path     = out_path,
                model        = model,
                tokenizer    = tokenizer,
                n_delim      = n_delim,
                batch_size   = args.batch_size,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()