"""
LAMMPS Neural Engine: orchestrates GPT-2 neural network operations
as molecular dynamics energy minimizations.

Each operation (linear layer, GELU, softmax, layer norm) is executed
as a LAMMPS simulation where particles settle to equilibrium positions
that encode the computation result.

All heavy computation happens in C++ LAMMPS extensions:
  - pair_style neural: bilinear coupling for linear layers
  - fix neural/self: self-restoring spring
  - fix neural/bias: per-atom bias force
  - fix neural/gelu: GELU activation
  - fix neural/softmax: softmax normalization
  - fix neural/layernorm: layer normalization
"""

import os
import tempfile
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lammps', 'python'))
from lammps import lammps


class LammpsNeuralEngine:
    """Executes neural network operations via LAMMPS physics simulations."""

    def __init__(self, lammps_args=None):
        if lammps_args is None:
            lammps_args = ["-log", "none", "-screen", "none"]
        self.lammps_args = lammps_args
        self._tmpdir = tempfile.mkdtemp(prefix="lammps_neural_")
        self._call_counter = 0

    def _unique_path(self, prefix, ext="dat"):
        """Generate a unique file path per call to avoid any file reuse issues."""
        self._call_counter += 1
        return os.path.join(self._tmpdir, f"{prefix}_{self._call_counter}.{ext}")

    def cleanup(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_data_file(self, filepath, positions_by_type):
        """Write LAMMPS data file.

        positions_by_type: list of (type_id, positions_array) tuples.
        Each positions_array is shape (n, 3) with [x_activation, y_spatial, z_spatial].
        """
        all_atoms = []
        for type_id, pos in positions_by_type:
            for p in pos:
                all_atoms.append((type_id, p[0], p[1], p[2]))

        n_total = len(all_atoms)
        n_types = max(t for t, _, _, _ in all_atoms)

        # Compute bounding box
        xs = [a[1] for a in all_atoms]
        ys = [a[2] for a in all_atoms]
        zs = [a[3] for a in all_atoms]
        margin = 500.0
        x_lo, x_hi = min(xs) - margin, max(xs) + margin
        y_lo, y_hi = min(ys) - margin, max(ys) + margin
        z_lo, z_hi = min(zs) - margin, max(zs) + margin
        # Ensure minimum box dimensions
        x_lo, x_hi = min(x_lo, -margin), max(x_hi, margin)
        y_lo, y_hi = min(y_lo, -margin), max(y_hi, margin)
        z_lo, z_hi = min(z_lo, -margin), max(z_hi, margin)

        with open(filepath, 'w') as f:
            f.write("LAMMPS neural data\n\n")
            f.write(f"{n_total} atoms\n")
            f.write(f"{n_types} atom types\n\n")
            f.write(f"{x_lo:.1f} {x_hi:.1f} xlo xhi\n")
            f.write(f"{y_lo:.1f} {y_hi:.1f} ylo yhi\n")
            f.write(f"{z_lo:.1f} {z_hi:.1f} zlo zhi\n\n")
            f.write("Masses\n\n")
            for t in range(1, n_types + 1):
                f.write(f"{t} 1.0\n")
            f.write("\nAtoms  # atomic\n\n")
            for idx, (type_id, x, y, z) in enumerate(all_atoms):
                f.write(f"{idx+1} {type_id} {x:.15e} {y:.15e} {z:.15e}\n")

    def _write_weights_file(self, filepath, n_input, n_output, W):
        """Write pair_neural weight file. W is shape (n_output, n_input)."""
        with open(filepath, 'w') as f:
            for i in range(n_output):
                out_id = n_input + i + 1
                for j in range(n_input):
                    in_id = j + 1
                    if W[i, j] != 0.0:
                        f.write(f"{in_id} {out_id} {W[i, j]:.15e}\n")

    def _write_bias_file(self, filepath, n_input, n_output, b):
        """Write neural/bias file. b is shape (n_output,)."""
        with open(filepath, 'w') as f:
            for i in range(n_output):
                atom_id = n_input + i + 1
                f.write(f"{atom_id} {b[i]:.15e}\n")

    def _write_layernorm_params(self, filepath, atom_ids, gamma, beta):
        """Write neural/layernorm parameter file."""
        with open(filepath, 'w') as f:
            for aid, g, b in zip(atom_ids, gamma, beta):
                f.write(f"{aid} {g:.15e} {b:.15e}\n")

    def _extract_x_positions(self, lmp, start_id, count):
        """Extract x-coordinates of atoms by ID range [start_id, start_id+count)."""
        x_all = lmp.gather_atoms("x", 1, 3)
        result = np.zeros(count)
        for i in range(count):
            result[i] = x_all[(start_id - 1 + i) * 3]
        return result

    def linear(self, x, w, b):
        """Compute y = x @ w + b via LAMMPS energy minimization.

        x: input vector, shape (n_input,)
        w: weight matrix, shape (n_input, n_output)
        b: bias vector, shape (n_output,)
        Returns: y, shape (n_output,)
        """
        x = np.ascontiguousarray(x, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)

        n_input = x.shape[0]
        n_output = b.shape[0]

        # W for pair_neural is (n_output, n_input), but gpt2.py uses x @ w
        # where w is (n_input, n_output). So W_pair = w.T
        W = np.ascontiguousarray(w.T)  # (n_output, n_input)

        data_file = self._unique_path("linear_data")
        weights_file = self._unique_path("linear_weights")
        bias_file = self._unique_path("linear_bias")

        # Input particles: type 1, x-coord = activation, y = index for separation
        input_pos = np.zeros((n_input, 3))
        input_pos[:, 0] = x
        input_pos[:, 1] = np.arange(n_input, dtype=float)

        # Output particles: type 2, x-coord = 0, y = offset for separation
        output_pos = np.zeros((n_output, 3))
        output_pos[:, 1] = np.arange(n_output, dtype=float) + n_input

        self._write_data_file(data_file, [(1, input_pos), (2, output_pos)])
        self._write_weights_file(weights_file, n_input, n_output, W)
        self._write_bias_file(bias_file, n_input, n_output, b)

        lmp = lammps(cmdargs=self.lammps_args)
        lmp.commands_string(f"""
            units           lj
            atom_style      atomic
            boundary        f f f
            atom_modify     map array sort 0 0.0

            read_data       {data_file}

            pair_style      neural {weights_file} 3000.0 nocheck
            pair_coeff      * *
            neighbor        0 nsq
            neigh_modify    once yes

            group           outputs type 2
            group           inputs type 1

            fix             self outputs neural/self
            fix_modify      self energy yes
            fix             bias outputs neural/bias {bias_file}
            fix_modify      bias energy yes
            fix             freeze inputs setforce 0.0 0.0 0.0

            min_style       cg
            minimize        1.0e-20 1.0e-20 100000 100000
        """)

        result = self._extract_x_positions(lmp, n_input + 1, n_output)
        lmp.close()
        return result

    def gelu(self, x):
        """Compute GELU(x) via LAMMPS energy minimization.

        x: input vector, shape (n,)
        Returns: gelu(x), shape (n,)
        """
        n = x.shape[0]

        data_file = self._unique_path("gelu_data")
        positions = np.zeros((n, 3))
        positions[:, 0] = x
        positions[:, 1] = np.arange(n, dtype=float)

        self._write_data_file(data_file, [(1, positions)])

        lmp = lammps(cmdargs=self.lammps_args)
        lmp.commands_string(f"""
            units           lj
            atom_style      atomic
            boundary        f f f
            atom_modify     map array sort 0 0.0

            read_data       {data_file}

            pair_style      zero 0.1
            pair_coeff      * *
            neighbor        0 nsq
            neigh_modify    once yes

            fix             gelu all neural/gelu
            fix_modify      gelu energy yes

            min_style       fire
            minimize        1.0e-15 1.0e-15 100000 100000
        """)

        result = self._extract_x_positions(lmp, 1, n)
        lmp.close()
        return result

    def softmax(self, x):
        """Compute softmax(x) via LAMMPS energy minimization.

        x: input vector, shape (n,)
        Returns: softmax(x), shape (n,)
        """
        n = x.shape[0]

        data_file = self._unique_path("softmax_data")
        positions = np.zeros((n, 3))
        positions[:, 0] = x
        positions[:, 1] = np.arange(n, dtype=float)

        self._write_data_file(data_file, [(1, positions)])

        lmp = lammps(cmdargs=self.lammps_args)
        lmp.commands_string(f"""
            units           lj
            atom_style      atomic
            boundary        f f f
            atom_modify     map array sort 0 0.0

            read_data       {data_file}

            pair_style      zero 0.1
            pair_coeff      * *
            neighbor        0 nsq
            neigh_modify    once yes

            fix             sm all neural/softmax
            fix_modify      sm energy yes

            min_style       fire
            minimize        1.0e-15 1.0e-15 100000 100000
        """)

        result = self._extract_x_positions(lmp, 1, n)
        lmp.close()
        return result

    def layer_norm(self, x, g, b, eps=1e-5):
        """Compute layer_norm(x, g, b) via LAMMPS energy minimization.

        x: input vector, shape (n,)
        g: gamma (scale), shape (n,)
        b: beta (shift), shape (n,)
        Returns: layer_norm(x, g, b), shape (n,)
        """
        n = x.shape[0]

        data_file = self._unique_path("ln_data")
        params_file = self._unique_path("ln_params")

        positions = np.zeros((n, 3))
        positions[:, 0] = x
        positions[:, 1] = np.arange(n, dtype=float)

        self._write_data_file(data_file, [(1, positions)])
        atom_ids = np.arange(1, n + 1)
        self._write_layernorm_params(params_file, atom_ids, g, b)

        lmp = lammps(cmdargs=self.lammps_args)
        lmp.commands_string(f"""
            units           lj
            atom_style      atomic
            boundary        f f f
            atom_modify     map array sort 0 0.0

            read_data       {data_file}

            pair_style      zero 0.1
            pair_coeff      * *
            neighbor        0 nsq
            neigh_modify    once yes

            fix             ln all neural/layernorm {params_file}
            fix_modify      ln energy yes

            min_style       fire
            minimize        1.0e-15 1.0e-15 100000 100000
        """)

        result = self._extract_x_positions(lmp, 1, n)
        lmp.close()
        return result

    def attention(self, q, k, v, mask):
        """Compute attention(q, k, v, mask) = softmax(q @ k.T / sqrt(d) + mask) @ v

        q, k, v: shape (T, d_head)
        mask: shape (T, T)
        Returns: shape (T, d_head)
        """
        T, d_head = q.shape

        # Compute attention scores: q @ k.T / sqrt(d_head)
        scores = q @ k.T / np.sqrt(d_head) + mask  # (T, T)

        # Apply softmax per row via LAMMPS
        attn_weights = np.zeros((T, T))
        for i in range(T):
            row = scores[i]
            if len(row) == 1:
                # Softmax of single element is always 1.0
                attn_weights[i] = np.array([1.0])
            else:
                attn_weights[i] = self.softmax(row)

        # Compute weighted sum: attn_weights @ v
        return attn_weights @ v

    def mha(self, x, c_attn, c_proj, n_head):
        """Multi-head attention via LAMMPS.

        x: input, shape (T, d_model)
        c_attn: dict with 'w' (d_model, 3*d_model) and 'b' (3*d_model,)
        c_proj: dict with 'w' (d_model, d_model) and 'b' (d_model,)
        n_head: number of attention heads
        Returns: shape (T, d_model)
        """
        T, d_model = x.shape

        # QKV projection: for each position, linear(x[t], c_attn.w, c_attn.b)
        qkv = np.zeros((T, 3 * d_model))
        for t in range(T):
            qkv[t] = self.linear(x[t], c_attn['w'], c_attn['b'])

        # Split into Q, K, V and reshape into heads
        q, k, v = np.split(qkv, 3, axis=-1)  # each (T, d_model)
        d_head = d_model // n_head

        # Split into heads: (T, d_model) -> list of n_head x (T, d_head)
        q_heads = np.split(q, n_head, axis=-1)
        k_heads = np.split(k, n_head, axis=-1)
        v_heads = np.split(v, n_head, axis=-1)

        # Causal mask: use -100 instead of -1e10 to keep atom positions
        # within a manageable box for LAMMPS. exp(-100) ≈ 3.7e-44, effectively 0.
        causal_mask = (1 - np.tri(T)) * -100.0

        # Attention per head
        out_heads = []
        for qh, kh, vh in zip(q_heads, k_heads, v_heads):
            out_heads.append(self.attention(qh, kh, vh, causal_mask))

        # Concatenate heads and project
        concat = np.hstack(out_heads)  # (T, d_model)

        # Output projection: for each position
        result = np.zeros((T, d_model))
        for t in range(T):
            result[t] = self.linear(concat[t], c_proj['w'], c_proj['b'])

        return result

    def ffn(self, x, c_fc, c_proj):
        """Feed-forward network via LAMMPS.

        x: input, shape (T, d_model)
        c_fc: dict with 'w' (d_model, 4*d_model) and 'b' (4*d_model,)
        c_proj: dict with 'w' (4*d_model, d_model) and 'b' (d_model,)
        Returns: shape (T, d_model)
        """
        T, d_model = x.shape
        d_ff = c_fc['b'].shape[0]

        result = np.zeros((T, d_model))
        for t in range(T):
            # First linear: d_model -> 4*d_model
            h = self.linear(x[t], c_fc['w'], c_fc['b'])
            # GELU activation
            h = self.gelu(h)
            # Second linear: 4*d_model -> d_model
            result[t] = self.linear(h, c_proj['w'], c_proj['b'])

        return result

    def transformer_block(self, x, block_params, n_head):
        """One transformer block via LAMMPS.

        x: input, shape (T, d_model)
        block_params: dict with 'mlp', 'attn', 'ln_1', 'ln_2'
        n_head: number of attention heads
        Returns: shape (T, d_model)
        """
        T, d_model = x.shape

        # Layer norm 1
        ln1_out = np.zeros_like(x)
        for t in range(T):
            ln1_out[t] = self.layer_norm(
                x[t], block_params['ln_1']['g'], block_params['ln_1']['b'])

        # Multi-head attention
        attn_out = self.mha(ln1_out, block_params['attn']['c_attn'],
                           block_params['attn']['c_proj'], n_head)

        # Residual
        x = x + attn_out

        # Layer norm 2
        ln2_out = np.zeros_like(x)
        for t in range(T):
            ln2_out[t] = self.layer_norm(
                x[t], block_params['ln_2']['g'], block_params['ln_2']['b'])

        # FFN
        ffn_out = self.ffn(ln2_out, block_params['mlp']['c_fc'],
                          block_params['mlp']['c_proj'])

        # Residual
        x = x + ffn_out

        return x
