"""
Load GPT-2 weights from HuggingFace and convert to picoGPT format.

The picoGPT format uses nested dicts with numpy arrays:
  params = {
      'wte': np.array (vocab_size, d_model),
      'wpe': np.array (context_len, d_model),
      'blocks': [
          {
              'ln_1': {'g': np.array, 'b': np.array},
              'ln_2': {'g': np.array, 'b': np.array},
              'attn': {
                  'c_attn': {'w': np.array (d_model, 3*d_model), 'b': np.array},
                  'c_proj': {'w': np.array (d_model, d_model), 'b': np.array},
              },
              'mlp': {
                  'c_fc': {'w': np.array (d_model, 4*d_model), 'b': np.array},
                  'c_proj': {'w': np.array (4*d_model, d_model), 'b': np.array},
              },
          },
          ...  # n_layer blocks
      ],
      'ln_f': {'g': np.array, 'b': np.array},
  }
"""

import numpy as np


def load_gpt2_from_huggingface(model_name="gpt2"):
    """Load GPT-2 weights from HuggingFace, return (encoder, hparams, params).

    model_name: "gpt2" (124M), "gpt2-medium" (355M), etc.
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"Loading {model_name} from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    sd = model.state_dict()
    config = model.config

    hparams = {
        "n_vocab": config.vocab_size,
        "n_ctx": config.n_positions,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer,
    }

    def to_np(key):
        return sd[key].cpu().numpy()

    # Token and position embeddings
    wte = to_np("transformer.wte.weight")
    wpe = to_np("transformer.wpe.weight")

    # Transformer blocks
    blocks = []
    for i in range(config.n_layer):
        prefix = f"transformer.h.{i}"
        block = {
            "ln_1": {
                "g": to_np(f"{prefix}.ln_1.weight"),
                "b": to_np(f"{prefix}.ln_1.bias"),
            },
            "ln_2": {
                "g": to_np(f"{prefix}.ln_2.weight"),
                "b": to_np(f"{prefix}.ln_2.bias"),
            },
            "attn": {
                "c_attn": {
                    "w": to_np(f"{prefix}.attn.c_attn.weight"),  # Conv1D: (d, 3*d)
                    "b": to_np(f"{prefix}.attn.c_attn.bias"),
                },
                "c_proj": {
                    "w": to_np(f"{prefix}.attn.c_proj.weight"),
                    "b": to_np(f"{prefix}.attn.c_proj.bias"),
                },
            },
            "mlp": {
                "c_fc": {
                    "w": to_np(f"{prefix}.mlp.c_fc.weight"),
                    "b": to_np(f"{prefix}.mlp.c_fc.bias"),
                },
                "c_proj": {
                    "w": to_np(f"{prefix}.mlp.c_proj.weight"),
                    "b": to_np(f"{prefix}.mlp.c_proj.bias"),
                },
            },
        }
        blocks.append(block)

    # Final layer norm
    ln_f = {
        "g": to_np("transformer.ln_f.weight"),
        "b": to_np("transformer.ln_f.bias"),
    }

    params = {
        "wte": wte,
        "wpe": wpe,
        "blocks": blocks,
        "ln_f": ln_f,
    }

    print(f"Loaded: {config.n_layer} layers, d_model={config.n_embd}, "
          f"n_head={config.n_head}, vocab={config.vocab_size}")

    return tokenizer, hparams, params


def verify_weights(params, hparams):
    """Sanity-check the loaded weight shapes."""
    d = hparams["n_embd"]
    n_layer = hparams["n_layer"]

    assert params["wte"].shape == (hparams["n_vocab"], d)
    assert params["wpe"].shape == (hparams["n_ctx"], d)
    assert len(params["blocks"]) == n_layer

    for i, block in enumerate(params["blocks"]):
        assert block["ln_1"]["g"].shape == (d,), f"block {i} ln_1.g"
        assert block["attn"]["c_attn"]["w"].shape == (d, 3 * d), f"block {i} c_attn.w"
        assert block["attn"]["c_attn"]["b"].shape == (3 * d,), f"block {i} c_attn.b"
        assert block["attn"]["c_proj"]["w"].shape == (d, d), f"block {i} c_proj.w"
        assert block["mlp"]["c_fc"]["w"].shape == (d, 4 * d), f"block {i} c_fc.w"
        assert block["mlp"]["c_proj"]["w"].shape == (4 * d, d), f"block {i} c_proj.w"

    print("All weight shapes verified.")


if __name__ == "__main__":
    tokenizer, hparams, params = load_gpt2_from_huggingface("gpt2")
    verify_weights(params, hparams)

    # Quick test: encode/decode
    text = "Hello, world!"
    ids = tokenizer.encode(text)
    print(f"'{text}' -> {ids} -> '{tokenizer.decode(ids)}'")
