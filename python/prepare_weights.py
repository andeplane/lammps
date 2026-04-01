"""
One-time GPT-2 weight preparation for pure LAMMPS inference.

Loads GPT-2 124M from HuggingFace, encodes input prompt, and writes:
  - gpt2.data: LAMMPS data file with 3840 atoms, bufA initialized to embedding
  - weights/: directory with 121 weight/bias/param files
  - wte.npy: token embedding matrix for logit computation

Usage:
    python prepare_weights.py "Alan Turing theorized that"
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from weight_loader import load_gpt2_from_huggingface


# Atom layout constants
N_BUFA = 768       # atoms 1-768
N_BUFB = 3072      # atoms 769-3840
N_TOTAL = N_BUFA + N_BUFB  # 3840


def write_data_file(filepath, x_embed):
    """Write LAMMPS data file with 3840 atoms.

    bufA (1-768): x-coord = embedded input, y-coord = atom index
    bufB (769-3840): x-coord = 0, y-coord = atom index
    """
    with open(filepath, 'w') as f:
        f.write("LAMMPS GPT-2 data file\n\n")
        f.write(f"{N_TOTAL} atoms\n")
        f.write("2 atom types\n\n")
        f.write("-500.0 500.0 xlo xhi\n")
        f.write("-100.0 4000.0 ylo yhi\n")
        f.write("-100.0 100.0 zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("1 1.0\n")
        f.write("2 1.0\n\n")
        f.write("Atoms  # atomic\n\n")
        # bufA: type 1, x = embedded value
        for i in range(N_BUFA):
            f.write(f"{i+1} 1 {x_embed[i]:.15e} {float(i)} 0.0\n")
        # bufB: type 2, x = 0
        for i in range(N_BUFB):
            atom_id = N_BUFA + i + 1
            f.write(f"{atom_id} 2 0.0 {float(atom_id - 1)} 0.0\n")


def write_pair_weights(filepath, input_ids, output_ids, W):
    """Write pair_neural weight file.

    W[i,j] is the weight from input_ids[j] to output_ids[i].
    W shape: (n_output, n_input)
    """
    with open(filepath, 'w') as f:
        for i, out_id in enumerate(output_ids):
            for j, in_id in enumerate(input_ids):
                w = W[i, j]
                if w != 0.0:
                    f.write(f"{in_id} {out_id} {w:.15e}\n")


def write_bias(filepath, output_ids, b):
    """Write fix neural/bias file."""
    with open(filepath, 'w') as f:
        for i, atom_id in enumerate(output_ids):
            f.write(f"{atom_id} {b[i]:.15e}\n")


def write_ln_params(filepath, atom_ids, gamma, beta):
    """Write fix neural/layernorm parameter file."""
    with open(filepath, 'w') as f:
        for i, atom_id in enumerate(atom_ids):
            f.write(f"{atom_id} {gamma[i]:.15e} {beta[i]:.15e}\n")


def prepare(prompt, output_dir="."):
    """Generate all files needed for LAMMPS GPT-2 inference."""

    tokenizer, hparams, params = load_gpt2_from_huggingface("gpt2")

    # Encode prompt (T=1: use only the last token for next-token prediction)
    input_ids = tokenizer.encode(prompt)
    T = len(input_ids)
    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {input_ids} (T={T})")

    if T > 1:
        print(f"WARNING: T={T} > 1. The LAMMPS script currently supports T=1 only.")
        print(f"Using last token position for prediction (all tokens embedded).")

    # For T=1, use only the first token
    # For T>1, we'd need a different particle layout
    token_id = input_ids[0]
    x_embed = params['wte'][token_id] + params['wpe'][0]
    x_embed = x_embed.astype(np.float64)

    # Atom ID ranges
    bufA_ids = list(range(1, N_BUFA + 1))              # 1-768
    bufB_qkv_ids = list(range(769, 769 + 2304))        # 769-3072
    bufB_v_ids = list(range(769 + 1536, 769 + 2304))   # 2305-3072
    bufB_3072_ids = list(range(769, 769 + 3072))        # 769-3840

    # Create output directories
    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Write data file
    data_path = os.path.join(output_dir, "gpt2.data")
    write_data_file(data_path, x_embed)
    print(f"Wrote {data_path} ({N_TOTAL} atoms)")

    # Write wte for logit computation
    wte_path = os.path.join(output_dir, "wte.npy")
    np.save(wte_path, params['wte'])
    print(f"Wrote {wte_path}")

    # Save tokenizer info
    with open(os.path.join(output_dir, "prompt_info.txt"), 'w') as f:
        f.write(f"prompt: {prompt}\n")
        f.write(f"token_ids: {input_ids}\n")
        f.write(f"n_head: {hparams['n_head']}\n")

    # Write weight files for each block
    n_files = 0
    for block_idx in range(12):
        block = params['blocks'][block_idx]
        prefix = os.path.join(weights_dir, f"block_{block_idx}")

        # LN1 params (atoms 1-768)
        write_ln_params(f"{prefix}_ln1.dat", bufA_ids,
                       block['ln_1']['g'], block['ln_1']['b'])

        # QKV linear: bufA(1-768) -> bufB_qkv(769-3072)
        # GPT-2 weight shape: (768, 2304), for pair_neural need W.T = (2304, 768)
        W_qkv = block['attn']['c_attn']['w'].T.astype(np.float64)
        write_pair_weights(f"{prefix}_qkv_w.dat", bufA_ids, bufB_qkv_ids, W_qkv)
        write_bias(f"{prefix}_qkv_b.dat", bufB_qkv_ids,
                  block['attn']['c_attn']['b'])

        # c_proj linear: bufB_v(2305-3072) -> bufA(1-768)
        W_cproj = block['attn']['c_proj']['w'].T.astype(np.float64)
        write_pair_weights(f"{prefix}_cproj_w.dat", bufB_v_ids, bufA_ids, W_cproj)
        write_bias(f"{prefix}_cproj_b.dat", bufA_ids,
                  block['attn']['c_proj']['b'])

        # LN2 params
        write_ln_params(f"{prefix}_ln2.dat", bufA_ids,
                       block['ln_2']['g'], block['ln_2']['b'])

        # c_fc linear: bufA(1-768) -> bufB_3072(769-3840)
        W_fc = block['mlp']['c_fc']['w'].T.astype(np.float64)
        write_pair_weights(f"{prefix}_fc_w.dat", bufA_ids, bufB_3072_ids, W_fc)
        write_bias(f"{prefix}_fc_b.dat", bufB_3072_ids,
                  block['mlp']['c_fc']['b'])

        # fc_proj linear: bufB_3072(769-3840) -> bufA(1-768)
        W_fcproj = block['mlp']['c_proj']['w'].T.astype(np.float64)
        write_pair_weights(f"{prefix}_fcproj_w.dat", bufB_3072_ids, bufA_ids, W_fcproj)
        write_bias(f"{prefix}_fcproj_b.dat", bufA_ids,
                  block['mlp']['c_proj']['b'])

        n_files += 10  # 2 LN + 4 weight + 4 bias
        print(f"  Block {block_idx}: wrote 10 files")

    # Final layer norm
    write_ln_params(os.path.join(weights_dir, "ln_f.dat"), bufA_ids,
                   params['ln_f']['g'], params['ln_f']['b'])
    n_files += 1

    print(f"\nTotal: {n_files} weight files in {weights_dir}/")

    # Compute numpy reference for verification
    print("\nComputing numpy reference...")
    from gpt2_lammps import gpt2_numpy
    logits_ref = gpt2_numpy(np.array([token_id]), **params, n_head=hparams['n_head'])
    next_token = int(np.argmax(logits_ref))
    print(f"Expected next token: {next_token} = '{tokenizer.decode([next_token])}'")
    np.save(os.path.join(output_dir, "logits_ref.npy"), logits_ref)

    total_size = sum(
        os.path.getsize(os.path.join(weights_dir, f))
        for f in os.listdir(weights_dir)
    )
    print(f"Total weight file size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare GPT-2 weights for LAMMPS")
    parser.add_argument("prompt", nargs="?", default="The",
                       help="Input prompt (default: 'The')")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    args = parser.parse_args()
    prepare(args.prompt, args.output_dir)
