"""
Compute logits from LAMMPS activation dump and print predicted next token.

Reads activations.dump (768 values from bufA) and wte.npy (token embeddings),
computes logits = activations @ wte.T, prints top-k predictions.

Usage:
    python compute_logits.py [--top-k 10]
"""

import argparse
import numpy as np


def read_activations(dump_file="activations.dump"):
    """Read x-coordinates from LAMMPS custom dump file."""
    activations = {}
    reading_atoms = False
    with open(dump_file) as f:
        for line in f:
            if line.startswith("ITEM: ATOMS"):
                reading_atoms = True
                continue
            if reading_atoms:
                parts = line.split()
                atom_id = int(parts[0])
                x_val = float(parts[1])
                activations[atom_id] = x_val

    # Sort by atom ID and return as array
    n = len(activations)
    result = np.zeros(n)
    for i in range(n):
        result[i] = activations[i + 1]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default="activations.dump")
    parser.add_argument("--wte", default="wte.npy")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ref-logits", default="logits_ref.npy")
    args = parser.parse_args()

    # Load activations
    x = read_activations(args.dump)
    print(f"Loaded {len(x)} activations from {args.dump}")
    print(f"  Range: [{x.min():.4f}, {x.max():.4f}]")

    # Load token embeddings
    wte = np.load(args.wte)
    print(f"Loaded wte: {wte.shape}")

    # Compute logits
    logits = x @ wte.T
    print(f"Logits: {logits.shape}, range [{logits.min():.2f}, {logits.max():.2f}]")

    # Top-k predictions
    top_k_ids = np.argsort(logits)[-args.top_k:][::-1]

    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print(f"\nTop-{args.top_k} predictions:")
        for i, tid in enumerate(top_k_ids):
            token_str = tokenizer.decode([tid])
            print(f"  {i+1}. token {tid:5d} = '{token_str}' (logit={logits[tid]:.4f})")
    except ImportError:
        print(f"\nTop-{args.top_k} token IDs: {list(top_k_ids)}")

    # Compare with reference if available
    try:
        logits_ref = np.load(args.ref_logits)
        max_diff = np.max(np.abs(logits - logits_ref))
        top1_match = np.argmax(logits) == np.argmax(logits_ref)
        print(f"\nReference comparison:")
        print(f"  Max logit diff: {max_diff:.2e}")
        print(f"  Top-1 match: {top1_match}")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
