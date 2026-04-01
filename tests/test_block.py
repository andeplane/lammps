"""
Test: Verify a single transformer block via LAMMPS matches the numpy
reference implementation from gpt2.py.

Uses small dimensions (d_model=8, n_head=2) for speed, with random weights.
"""

import sys
import os
import numpy as np

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lammps', 'python'))

# Set library path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'lammps', 'build')
os.environ.setdefault('DYLD_LIBRARY_PATH', build_dir)

from orchestrator import LammpsNeuralEngine

# Reference implementations from gpt2.py
def gelu_ref(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax_ref(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm_ref(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear_ref(x, w, b):
    return x @ w + b

def ffn_ref(x, c_fc, c_proj):
    return linear_ref(gelu_ref(linear_ref(x, **c_fc)), **c_proj)

def attention_ref(q, k, v, mask):
    return softmax_ref(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

def mha_ref(x, c_attn, c_proj, n_head):
    x = linear_ref(x, **c_attn)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1)))
    causal_mask = (1 - np.tri(x.shape[0])) * -100.0
    out_heads = [attention_ref(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = linear_ref(np.hstack(out_heads), **c_proj)
    return x

def transformer_block_ref(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha_ref(layer_norm_ref(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn_ref(layer_norm_ref(x, **ln_2), **mlp)
    return x


def make_random_block_params(d_model, n_head, rng, scale=0.1):
    """Create random transformer block parameters."""
    d_ff = d_model * 4  # standard 4x expansion

    return {
        'ln_1': {
            'g': rng.randn(d_model) * 0.02 + 1.0,
            'b': rng.randn(d_model) * 0.02,
        },
        'ln_2': {
            'g': rng.randn(d_model) * 0.02 + 1.0,
            'b': rng.randn(d_model) * 0.02,
        },
        'attn': {
            'c_attn': {
                'w': rng.randn(d_model, 3 * d_model) * scale,
                'b': rng.randn(3 * d_model) * scale,
            },
            'c_proj': {
                'w': rng.randn(d_model, d_model) * scale,
                'b': rng.randn(d_model) * scale,
            },
        },
        'mlp': {
            'c_fc': {
                'w': rng.randn(d_model, d_ff) * scale,
                'b': rng.randn(d_ff) * scale,
            },
            'c_proj': {
                'w': rng.randn(d_ff, d_model) * scale,
                'b': rng.randn(d_model) * scale,
            },
        },
    }


def test_individual_ops():
    """Test each operation individually through the engine."""
    print("=== Individual Operations Test ===")
    engine = LammpsNeuralEngine()
    rng = np.random.RandomState(42)

    # Linear
    x = rng.randn(8)
    w = rng.randn(8, 16) * 0.1
    b = rng.randn(16) * 0.1
    y_ref = linear_ref(x, w, b)
    y_lmp = engine.linear(x, w, b)
    err = np.max(np.abs(y_ref - y_lmp))
    print(f"  Linear (8->16):   max_err = {err:.2e}", "PASS" if err < 1e-6 else "FAIL")

    # GELU
    x = rng.randn(16) * 2
    y_ref = gelu_ref(x)
    y_lmp = engine.gelu(x)
    err = np.max(np.abs(y_ref - y_lmp))
    print(f"  GELU (16):        max_err = {err:.2e}", "PASS" if err < 1e-6 else "FAIL")

    # Softmax
    x = rng.randn(8) * 3
    y_ref = softmax_ref(x)
    y_lmp = engine.softmax(x)
    err = np.max(np.abs(y_ref - y_lmp))
    print(f"  Softmax (8):      max_err = {err:.2e}", "PASS" if err < 1e-6 else "FAIL")

    # Layer norm
    x = rng.randn(8)
    g = rng.randn(8) * 0.02 + 1.0
    b = rng.randn(8) * 0.02
    y_ref = layer_norm_ref(x, g, b)
    y_lmp = engine.layer_norm(x, g, b)
    err = np.max(np.abs(y_ref - y_lmp))
    print(f"  LayerNorm (8):    max_err = {err:.2e}", "PASS" if err < 1e-6 else "FAIL")

    engine.cleanup()
    print()
    return err < 1e-6


def test_transformer_block_t1():
    """Test full transformer block with T=1, small dimensions."""
    print("=== Transformer Block (T=1, d=8, heads=2) ===")

    d_model = 8
    n_head = 2
    T = 1

    rng = np.random.RandomState(123)
    x = rng.randn(T, d_model) * 0.5
    params = make_random_block_params(d_model, n_head, rng, scale=0.1)

    # Numpy reference
    y_ref = transformer_block_ref(x, **params, n_head=n_head)

    # LAMMPS engine
    engine = LammpsNeuralEngine()
    y_lmp = engine.transformer_block(x, params, n_head)
    engine.cleanup()

    err = np.max(np.abs(y_ref - y_lmp))
    print(f"  Input:    {x[0][:4]}...")
    print(f"  Ref out:  {y_ref[0][:4]}...")
    print(f"  LAMMPS:   {y_lmp[0][:4]}...")
    print(f"  Max err:  {err:.2e}")

    tol = 1e-6
    passed = err < tol
    print(f"  {'PASS' if passed else 'FAIL'} (tol={tol})")
    print()
    return passed


def test_transformer_block_t3():
    """Test full transformer block with T=3 (exercises full attention)."""
    print("=== Transformer Block (T=3, d=8, heads=2) ===")

    d_model = 8
    n_head = 2
    T = 3

    rng = np.random.RandomState(456)
    x = rng.randn(T, d_model) * 0.5
    params = make_random_block_params(d_model, n_head, rng, scale=0.1)

    # Numpy reference
    y_ref = transformer_block_ref(x, **params, n_head=n_head)

    # LAMMPS engine
    engine = LammpsNeuralEngine()
    y_lmp = engine.transformer_block(x, params, n_head)
    engine.cleanup()

    err = np.max(np.abs(y_ref - y_lmp))
    print(f"  Max err:  {err:.2e}")

    tol = 1e-6
    passed = err < tol
    print(f"  {'PASS' if passed else 'FAIL'} (tol={tol})")
    print()
    return passed


if __name__ == "__main__":
    results = []
    results.append(test_individual_ops())
    results.append(test_transformer_block_t1())
    results.append(test_transformer_block_t3())

    print("=" * 50)
    if all(results):
        print("ALL TESTS PASSED")
    else:
        print(f"SOME TESTS FAILED")
        sys.exit(1)
