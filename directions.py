import os
import argparse
import torch
import torch.nn.functional as F


CATEGORY_FILENAMES = [
    "Adult_Content",
    "Child_Abuse",
    "Economic_Harm",
    "Fraud_Deception",
    "Hate_Harass_Violence",
    "Illegal_Activity",
    "Malware_Viruses",
    "Physical_Harm",
    "Political_Campaigning",
    "Privacy_Violation_Activity",
    "Tailored_Financial_Advice",
]

CATEGORY_LABELS = [
    "Adult Content",
    "Child Abuse",
    "Economic Harm",
    "Fraud/Deception",
    "Hate/Harass/Violence",
    "Illegal Activity",
    "Malware Viruses",
    "Physical Harm",
    "Political Campaigning",
    "Privacy Violation",
    "Tailored Financial Advice",
]


def load_activations(path: str, position: int = 0) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")
    acts = data["activations"]          # [n_examples, n_layers, 2, hidden_dim]
    return acts[:, :, position, :].float()  # [n_examples, n_layers, hidden_dim]


def compute_dim(
    harmful: torch.Tensor,
    harmless: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_harmful  = harmful.mean(dim=0)   # [n_layers, hidden_dim]
    mean_harmless = harmless.mean(dim=0)  # [n_layers, hidden_dim]
    raw           = mean_harmful - mean_harmless
    normed        = F.normalize(raw, dim=-1)
    return raw, normed


def compute_cosine_similarity_matrix(
    directions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_categories, n_layers, hidden_dim = directions.shape
    layer_sim = torch.zeros(n_layers, n_categories, n_categories)

    for l in range(n_layers):
        d = directions[:, l, :]          
        layer_sim[l] = d @ d.T

    mean_sim = layer_sim.mean(dim=0)     
    return layer_sim, mean_sim


def compute_off_diagonal_means(sim_matrix: torch.Tensor) -> torch.Tensor:
    n = sim_matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    off_diag_means = torch.zeros(n)
    for i in range(n):
        off_diag_means[i] = sim_matrix[i][mask[i]].mean()
    return off_diag_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", default="activations/")
    parser.add_argument("--output_dir",      default="results/")
    parser.add_argument("--position",        default=0, type=int,
                        help="Token position to use: 0=t_inst (harmfulness), 1=t_post-inst (refusal)")
    parser.add_argument("--layer_start", default=9,  type=int,
                    help="Start layer for similarity computation (inclusive)")
    parser.add_argument("--layer_end",   default=20, type=int,
                    help="End layer for similarity computation (inclusive)")
    parser.add_argument("--refused", action="store_true",
                    help="Use refused-only filtered activations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    harmless_path = os.path.join(args.activations_dir, "activations_harmless.pt")
    print(f"Loading harmless baseline from {harmless_path}...")
    harmless = load_activations(harmless_path, position=args.position)
    print(f"  harmless shape: {harmless.shape}")

    n_layers    = harmless.shape[1]
    hidden_dim  = harmless.shape[2]
    n_categories = len(CATEGORY_FILENAMES)
    all_raw    = torch.zeros(n_categories, n_layers, hidden_dim)
    all_normed = torch.zeros(n_categories, n_layers, hidden_dim)

    print(f"\nComputing DIM directions at position {args.position} "
          f"({'t_inst' if args.position == 0 else 't_post-inst'})...\n")

    for i, (fname, label) in enumerate(zip(CATEGORY_FILENAMES, CATEGORY_LABELS)):
        path = os.path.join(args.activations_dir, f"activations_{fname}.pt")
        harmful = load_activations(path, position=args.position)
        print(f"  [{i+1:2d}/{n_categories}] {label:<30}  harmful={harmful.shape[0]} examples")

        raw, normed = compute_dim(harmful, harmless)
        all_raw[i]    = raw
        all_normed[i] = normed
    # Check if mean_harmful vectors are already similar across categories
    print("\n── Sanity: pairwise similarity of raw mean_harmful (before DIM) ────")
    raw_means = torch.zeros(n_categories, n_layers, hidden_dim)
    for i, fname in enumerate(CATEGORY_FILENAMES):
        prefix = "activations_refused_" if args.refused else "activations_"
        path = os.path.join(args.activations_dir, f"{prefix}{fname}.pt")
        harmful = load_activations(path, position=0)
        raw_means[i] = harmful.mean(dim=0)

    raw_means_normed = F.normalize(raw_means, dim=-1)
    # average cosine sim across layers for each pair
    sim = torch.einsum('ilh,jlh->ijl', raw_means_normed, raw_means_normed).mean(dim=-1)
    off_diag = sim[~torch.eye(n_categories, dtype=torch.bool)].mean()
    print(f"  Mean pairwise cosine sim of raw harmful means: {off_diag:.4f}")
    print(f"  (If this is >0.99, all harmful categories look identical at t_inst)")
    print("\nComputing cosine similarity matrix...")
    #layer_sim, mean_sim = compute_cosine_similarity_matrix(all_normed)
    directions_subset = all_normed[:, args.layer_start:args.layer_end+1, :]
    layer_sim, mean_sim = compute_cosine_similarity_matrix(directions_subset)
    off_diag_means      = compute_off_diagonal_means(mean_sim)
    print("\n── Inter-category cosine similarity (off-diagonal means) ───────────")
    print(f"  {'Category':<35}  {'Mean sim to others':>20}")
    print(f"  {'-'*35}  {'-'*20}")
    for i, label in enumerate(CATEGORY_LABELS):
        print(f"  {label:<35}  {off_diag_means[i].item():>20.4f}")
    print(f"\n  Overall mean (all off-diagonal): {off_diag_means.mean().item():.4f}")
    print(f"  Zhao et al. Qwen2 reference:     ~0.65 (range 0.60-0.73)")
    print(f"  {'PASS' if off_diag_means.mean().item() < 0.85 else 'CHECK'} "
          f"— directions are {'distinct' if off_diag_means.mean().item() < 0.85 else 'suspiciously similar'}")

    out_path = os.path.join(args.output_dir, "directions.pt")
    torch.save({
        "directions"       : all_normed,       
        "directions_raw"   : all_raw,           
        "categories"       : CATEGORY_LABELS,
        "layer_sim_matrix" : layer_sim,         
        "mean_sim_matrix"  : mean_sim,          
        "off_diag_means"   : off_diag_means,    
        "position"         : args.position,
    }, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()