"""
Test: Verify that LAMMPS energy minimization with pair_style neural
computes y = Wx + b correctly via physical equilibrium.

Setup:
- 4 input particles (frozen, x-coords = input values)
- 3 output particles (free, x-coords = output activations)
- pair_style neural encodes weight matrix W (3x4 = 12 pairs)
- fix neural/self applies self-restoring spring F = -x
- fix neural/bias applies per-atom bias F = b_i
- Energy minimization drives output particles to y = Wx + b
"""

import sys
import os
import tempfile
import numpy as np

# Add LAMMPS python bindings to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lammps', 'python'))

from lammps import lammps


def write_data_file(filepath, n_input, n_output, input_values):
    """Write LAMMPS data file with input and output particles.

    Input particles: type 1, IDs 1..n_input, x-coord = input_values
    Output particles: type 2, IDs (n_input+1)..(n_input+n_output), x-coord = 0

    Particles are separated in y-direction to avoid unwanted interactions.
    """
    n_total = n_input + n_output
    with open(filepath, 'w') as f:
        f.write("LAMMPS data file for neural network linear layer test\n\n")
        f.write(f"{n_total} atoms\n")
        f.write("2 atom types\n\n")
        # Large box to avoid periodic boundary issues
        f.write("-1000.0 1000.0 xlo xhi\n")
        f.write("-1000.0 1000.0 ylo yhi\n")
        f.write("-1000.0 1000.0 zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("1 1.0\n")
        f.write("2 1.0\n\n")
        f.write("Atoms  # atomic\n\n")
        # Input particles (type 1): x-coord = activation value
        for i in range(n_input):
            atom_id = i + 1
            f.write(f"{atom_id} 1 {input_values[i]:.15e} {float(i):.1f} 0.0\n")
        # Output particles (type 2): x-coord = 0 (will be determined by minimization)
        for i in range(n_output):
            atom_id = n_input + i + 1
            y_pos = float(n_input + i)
            f.write(f"{atom_id} 2 0.0 {y_pos:.1f} 0.0\n")


def write_weights_file(filepath, n_input, n_output, W):
    """Write weight file for pair_style neural.

    W is shape (n_output, n_input).
    Each line: input_atom_id  output_atom_id  weight
    """
    with open(filepath, 'w') as f:
        for i in range(n_output):
            output_id = n_input + i + 1  # output atom IDs start after inputs
            for j in range(n_input):
                input_id = j + 1
                f.write(f"{input_id} {output_id} {W[i, j]:.15e}\n")


def write_bias_file(filepath, n_input, n_output, b):
    """Write bias file for fix neural/bias.

    b is shape (n_output,).
    Each line: atom_id  bias_value
    """
    with open(filepath, 'w') as f:
        for i in range(n_output):
            atom_id = n_input + i + 1
            f.write(f"{atom_id} {b[i]:.15e}\n")


def test_linear_layer():
    """Test y = Wx + b via LAMMPS energy minimization."""

    # Problem dimensions
    n_input = 4
    n_output = 3

    # Random but reproducible test data
    rng = np.random.RandomState(42)
    x_input = rng.randn(n_input)
    W = rng.randn(n_output, n_input)
    b = rng.randn(n_output)

    # Expected output (numpy reference)
    y_expected = W @ x_input + b

    print("=== Linear Layer Test: y = Wx + b ===")
    print(f"Input x:    {x_input}")
    print(f"Weight W:\n{W}")
    print(f"Bias b:     {b}")
    print(f"Expected y: {y_expected}")

    # Create temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, "atoms.data")
        weights_file = os.path.join(tmpdir, "weights.dat")
        bias_file = os.path.join(tmpdir, "bias.dat")

        write_data_file(data_file, n_input, n_output, x_input)
        write_weights_file(weights_file, n_input, n_output, W)
        write_bias_file(bias_file, n_input, n_output, b)

        # Run LAMMPS
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])

        lmp.commands_string(f"""
            units           lj
            atom_style      atomic
            boundary        f f f
            atom_modify     map array

            read_data       {data_file}

            # Bilinear coupling potential (weights)
            pair_style      neural {weights_file} 3000.0 nocheck
            pair_coeff      * *

            # Self-restoring spring on output atoms only
            group           outputs type 2
            group           inputs type 1
            fix             self outputs neural/self
            fix_modify      self energy yes

            # Bias force on output atoms
            fix             bias outputs neural/bias {bias_file}
            fix_modify      bias energy yes

            # Freeze input atoms
            fix             freeze inputs setforce 0.0 0.0 0.0

            # Constrain all atoms to x-axis only (zero y,z forces)
            fix             yz all setforce NULL 0.0 0.0

            # Energy minimization
            min_style       cg
            minimize        1.0e-15 1.0e-15 10000 10000
        """)

        # Extract output particle positions
        n_total = n_input + n_output
        x_all = lmp.gather_atoms("x", 1, 3)  # all positions, shape (n_total*3,)

        y_lammps = np.zeros(n_output)
        for i in range(n_output):
            atom_idx = n_input + i  # 0-based index
            y_lammps[i] = x_all[atom_idx * 3]  # x-coordinate

        lmp.close()

    print(f"\nLAMMPS y:   {y_lammps}")
    print(f"Expected y: {y_expected}")
    print(f"Difference: {y_lammps - y_expected}")
    print(f"Max error:  {np.max(np.abs(y_lammps - y_expected)):.2e}")

    # Check agreement
    tol = 1e-6
    if np.allclose(y_lammps, y_expected, atol=tol):
        print(f"\nPASS: y = Wx + b computed correctly via LAMMPS physics (tol={tol})")
        return True
    else:
        print(f"\nFAIL: Results differ by more than {tol}")
        return False


def test_larger_linear_layer():
    """Test with a larger layer (64 -> 32) to verify scaling."""

    n_input = 64
    n_output = 32

    rng = np.random.RandomState(123)
    x_input = rng.randn(n_input)
    W = rng.randn(n_output, n_input) * 0.1  # smaller weights for stability
    b = rng.randn(n_output) * 0.1

    y_expected = W @ x_input + b

    print("\n=== Larger Linear Layer Test (64 -> 32) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, "atoms.data")
        weights_file = os.path.join(tmpdir, "weights.dat")
        bias_file = os.path.join(tmpdir, "bias.dat")

        write_data_file(data_file, n_input, n_output, x_input)
        write_weights_file(weights_file, n_input, n_output, W)
        write_bias_file(bias_file, n_input, n_output, b)

        lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])

        lmp.commands_string(f"""
            units           lj
            atom_style      atomic
            boundary        f f f
            atom_modify     map array

            read_data       {data_file}

            pair_style      neural {weights_file} 3000.0 nocheck
            pair_coeff      * *

            group           outputs type 2
            group           inputs type 1
            fix             self outputs neural/self
            fix_modify      self energy yes
            fix             bias outputs neural/bias {bias_file}
            fix_modify      bias energy yes
            fix             freeze inputs setforce 0.0 0.0 0.0
            fix             yz all setforce NULL 0.0 0.0

            min_style       cg
            minimize        1.0e-15 1.0e-15 10000 10000
        """)

        n_total = n_input + n_output
        x_all = lmp.gather_atoms("x", 1, 3)

        y_lammps = np.zeros(n_output)
        for i in range(n_output):
            y_lammps[i] = x_all[(n_input + i) * 3]

        lmp.close()

    max_err = np.max(np.abs(y_lammps - y_expected))
    print(f"Max error: {max_err:.2e}")

    tol = 1e-6
    if np.allclose(y_lammps, y_expected, atol=tol):
        print(f"PASS: 64->32 linear layer correct (tol={tol})")
        return True
    else:
        print(f"FAIL: Max error {max_err:.2e} exceeds {tol}")
        # Print first few mismatches
        diffs = np.abs(y_lammps - y_expected)
        worst = np.argsort(diffs)[-5:]
        for idx in worst:
            print(f"  output[{idx}]: lammps={y_lammps[idx]:.10f}, expected={y_expected[idx]:.10f}, diff={diffs[idx]:.2e}")
        return False


if __name__ == "__main__":
    # Set library path for LAMMPS shared library
    build_dir = os.path.join(os.path.dirname(__file__), '..', 'lammps', 'build')
    if sys.platform == 'darwin':
        os.environ.setdefault('DYLD_LIBRARY_PATH', build_dir)
    else:
        os.environ.setdefault('LD_LIBRARY_PATH', build_dir)

    results = []
    results.append(test_linear_layer())
    results.append(test_larger_linear_layer())

    print("\n" + "=" * 50)
    if all(results):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
