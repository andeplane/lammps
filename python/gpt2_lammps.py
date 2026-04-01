"""
GPT-2 inference via LAMMPS molecular dynamics.

Every neural network operation — linear layers, GELU, softmax, layer norm —
is computed by setting up particles in LAMMPS and finding their equilibrium
positions through energy minimization. The x-coordinate of each particle
encodes an activation value.

Usage:
    python gpt2_lammps.py "Alan Turing theorized that" --n_tokens 20
"""

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lammps', 'python'))

from orchestrator import LammpsNeuralEngine
from weight_loader import load_gpt2_from_huggingface


def gpt2_lammps(inputs, wte, wpe, blocks, ln_f, n_head, engine):
    """Full GPT-2 forward pass via LAMMPS.

    inputs: list of token IDs
    Returns: logits array of shape (n_vocab,) for the last position
    """
    T = len(inputs)
    d_model = wte.shape[1]

    # Token + position embeddings (lookup, no LAMMPS needed)
    x = wte[inputs] + wpe[range(T)]  # (T, d_model)

    # Transformer blocks
    for i, block in enumerate(blocks):
        print(f"  Block {i+1}/{len(blocks)}...", end="", flush=True)
        t0 = time.time()
        x = engine.transformer_block(x, block, n_head)
        dt = time.time() - t0
        print(f" {dt:.1f}s")

    # Final layer norm
    for t in range(T):
        x[t] = engine.layer_norm(x[t], ln_f['g'], ln_f['b'])

    # Logits: x[-1] @ wte.T (project last position to vocab)
    # This is a large linear layer (768 -> 50257) with no bias.
    # For efficiency, compute in numpy since wte.T is the embedding matrix
    # and this is just a dot product (no physics needed for the final projection).
    logits = x[-1] @ wte.T  # (n_vocab,)

    return logits


def generate(inputs, params, n_head, n_tokens_to_generate, engine):
    """Generate tokens autoregressively via LAMMPS."""
    from tqdm import tqdm

    generated = []
    for i in tqdm(range(n_tokens_to_generate), desc="Generating"):
        logits = gpt2_lammps(inputs, **params, n_head=n_head, engine=engine)
        next_id = int(np.argmax(logits))
        inputs = np.append(inputs, [next_id])
        generated.append(next_id)

    return generated


def gpt2_numpy(inputs, wte, wpe, blocks, ln_f, n_head):
    """Reference numpy GPT-2 (from gpt2.py) for verification."""
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def layer_norm(x, g, b, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / np.sqrt(variance + eps) + b

    def linear(x, w, b):
        return x @ w + b

    def ffn(x, c_fc, c_proj):
        return linear(gelu(linear(x, **c_fc)), **c_proj)

    def attention(q, k, v, mask):
        return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

    def mha(x, c_attn, c_proj, n_head):
        x = linear(x, **c_attn)
        qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1)))
        causal_mask = (1 - np.tri(x.shape[0])) * -1e10
        out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
        x = linear(np.hstack(out_heads), **c_proj)
        return x

    def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
        x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
        x = x + ffn(layer_norm(x, **ln_2), **mlp)
        return x

    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    x = layer_norm(x, **ln_f)
    return x[-1] @ wte.T


def main(prompt="Alan Turing theorized that", n_tokens=5, verify=True):
    """Run GPT-2 inference via LAMMPS molecular dynamics."""

    # Load model
    tokenizer, hparams, params = load_gpt2_from_huggingface("gpt2")
    n_head = hparams["n_head"]

    # Encode prompt
    input_ids = np.array(tokenizer.encode(prompt))
    print(f"\nPrompt: '{prompt}'")
    print(f"Tokens: {list(input_ids)} (length {len(input_ids)})")

    if verify:
        # First, run numpy reference for one forward pass
        print("\n--- NumPy reference (one forward pass) ---")
        t0 = time.time()
        logits_np = gpt2_numpy(input_ids, **params, n_head=n_head)
        dt = time.time() - t0
        next_token_np = int(np.argmax(logits_np))
        print(f"Next token: {next_token_np} = '{tokenizer.decode([next_token_np])}'")
        print(f"Time: {dt:.2f}s")

    # Run LAMMPS version
    print("\n--- LAMMPS molecular dynamics (one forward pass) ---")
    engine = LammpsNeuralEngine()

    t0 = time.time()
    logits_lmp = gpt2_lammps(input_ids, **params, n_head=n_head, engine=engine)
    dt = time.time() - t0

    next_token_lmp = int(np.argmax(logits_lmp))
    print(f"Next token: {next_token_lmp} = '{tokenizer.decode([next_token_lmp])}'")
    print(f"Time: {dt:.1f}s")

    if verify:
        # Compare logits
        # Note: logits are computed slightly differently (LAMMPS uses -100 mask
        # vs numpy uses -1e10), so compare top-k predictions instead
        top_k = 10
        np_top = np.argsort(logits_np)[-top_k:][::-1]
        lmp_top = np.argsort(logits_lmp)[-top_k:][::-1]

        print(f"\nTop-{top_k} predictions:")
        print(f"  NumPy:  {[tokenizer.decode([t]) for t in np_top]}")
        print(f"  LAMMPS: {[tokenizer.decode([t]) for t in lmp_top]}")
        print(f"  Match:  {np_top[0] == lmp_top[0]} (top-1)")

        logit_diff = np.max(np.abs(logits_np - logits_lmp))
        print(f"  Max logit diff: {logit_diff:.2e}")

    if n_tokens > 0:
        print(f"\n--- Generating {n_tokens} tokens ---")
        generated = generate(input_ids, params, n_head, n_tokens, engine)
        output_text = tokenizer.decode(generated)
        print(f"\n{prompt}{output_text}")

    engine.cleanup()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPT-2 via LAMMPS molecular dynamics")
    parser.add_argument("prompt", nargs="?", default="Alan Turing theorized that",
                       help="Input prompt")
    parser.add_argument("--n_tokens", type=int, default=5,
                       help="Number of tokens to generate")
    parser.add_argument("--no-verify", action="store_true",
                       help="Skip numpy verification")
    args = parser.parse_args()

    main(prompt=args.prompt, n_tokens=args.n_tokens, verify=not args.no_verify)
