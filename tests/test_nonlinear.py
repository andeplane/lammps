"""
Test: Verify GELU, softmax, and layer_norm via LAMMPS energy minimization
against numpy reference implementations.
"""

import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lammps', 'python'))
from lammps import lammps


def gelu_np(x):
    """Reference GELU (matches gpt2.py)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax_np(x):
    """Reference softmax (matches gpt2.py)."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def layer_norm_np(x, g, b, eps=1e-5):
    """Reference layer norm (matches gpt2.py)."""
    mean = np.mean(x)
    variance = np.var(x)
    return g * (x - mean) / np.sqrt(variance + eps) + b


def write_data_file(filepath, n_atoms, x_values):
    """Write LAMMPS data file with atoms at given x-positions."""
    # Box must be large enough to contain all atoms with some margin
    x_lo = min(np.min(x_values), 0) - 50.0
    x_hi = max(np.max(x_values), 0) + 50.0
    y_hi = float(n_atoms) + 50.0
    with open(filepath, 'w') as f:
        f.write("LAMMPS data file\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write("1 atom types\n\n")
        f.write(f"{x_lo:.1f} {x_hi:.1f} xlo xhi\n")
        f.write(f"-50.0 {y_hi:.1f} ylo yhi\n")
        f.write("-50.0 50.0 zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("1 1.0\n\n")
        f.write("Atoms  # atomic\n\n")
        for i in range(n_atoms):
            f.write(f"{i+1} 1 {x_values[i]:.15e} {float(i):.1f} 0.0\n")


def write_layernorm_params(filepath, n_atoms, gamma, beta):
    """Write layernorm gamma/beta parameter file."""
    with open(filepath, 'w') as f:
        for i in range(n_atoms):
            f.write(f"{i+1} {gamma[i]:.15e} {beta[i]:.15e}\n")


def run_lammps_minimize(data_file, extra_commands):
    """Run LAMMPS with given data file and extra commands, return final x-positions."""
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
    lmp.commands_string(f"""
        units           lj
        atom_style      atomic
        boundary        f f f
        atom_modify     map array
        read_data       {data_file}
        pair_style      zero 1.0
        pair_coeff      * *
    """)
    lmp.commands_string(extra_commands)
    lmp.commands_string("""
        min_style       fire
        minimize        1.0e-15 1.0e-15 100000 100000
    """)

    natoms = lmp.get_natoms()
    x_all = lmp.gather_atoms("x", 1, 3)
    result = np.array([x_all[i * 3] for i in range(natoms)])
    lmp.close()
    return result


def test_gelu():
    """Test GELU activation."""
    print("=== GELU Test ===")
    n = 8
    rng = np.random.RandomState(42)
    x_input = rng.randn(n) * 2  # range roughly [-4, 4]

    y_expected = gelu_np(x_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, "atoms.data")
        write_data_file(data_file, n, x_input)

        y_lammps = run_lammps_minimize(data_file, f"""
            fix gelu all neural/gelu
            fix_modify gelu energy yes
        """)

    print(f"Input:    {x_input}")
    print(f"Expected: {y_expected}")
    print(f"LAMMPS:   {y_lammps}")
    max_err = np.max(np.abs(y_lammps - y_expected))
    print(f"Max err:  {max_err:.2e}")

    tol = 1e-10
    if np.allclose(y_lammps, y_expected, atol=tol):
        print(f"PASS (tol={tol})\n")
        return True
    else:
        print(f"FAIL\n")
        return False


def test_gelu_edge_cases():
    """Test GELU with extreme values."""
    print("=== GELU Edge Cases ===")
    x_input = np.array([0.0, -5.0, 5.0, -0.001, 0.001, -10.0, 10.0, 1.0])
    y_expected = gelu_np(x_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, "atoms.data")
        write_data_file(data_file, len(x_input), x_input)

        y_lammps = run_lammps_minimize(data_file, """
            fix gelu all neural/gelu
            fix_modify gelu energy yes
        """)

    max_err = np.max(np.abs(y_lammps - y_expected))
    print(f"Max err:  {max_err:.2e}")

    tol = 1e-10
    if np.allclose(y_lammps, y_expected, atol=tol):
        print(f"PASS (tol={tol})\n")
        return True
    else:
        print(f"FAIL: {y_lammps} vs {y_expected}\n")
        return False


def test_softmax():
    """Test softmax."""
    print("=== Softmax Test ===")
    n = 6
    rng = np.random.RandomState(99)
    x_input = rng.randn(n) * 3

    y_expected = softmax_np(x_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, "atoms.data")
        write_data_file(data_file, n, x_input)

        y_lammps = run_lammps_minimize(data_file, """
            fix sm all neural/softmax
            fix_modify sm energy yes
        """)

    print(f"Input:    {x_input}")
    print(f"Expected: {y_expected}")
    print(f"LAMMPS:   {y_lammps}")
    max_err = np.max(np.abs(y_lammps - y_expected))
    print(f"Max err:  {max_err:.2e}")

    tol = 1e-10
    if np.allclose(y_lammps, y_expected, atol=tol):
        print(f"PASS (tol={tol})\n")
        return True
    else:
        print(f"FAIL\n")
        return False


def test_softmax_large():
    """Test softmax with larger input and extreme values."""
    print("=== Softmax Large Test ===")
    n = 50
    rng = np.random.RandomState(77)
    x_input = rng.randn(n) * 10  # large values to test numerical stability

    y_expected = softmax_np(x_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, "atoms.data")
        write_data_file(data_file, n, x_input)

        y_lammps = run_lammps_minimize(data_file, """
            fix sm all neural/softmax
            fix_modify sm energy yes
        """)

    max_err = np.max(np.abs(y_lammps - y_expected))
    print(f"Max err:  {max_err:.2e}")

    tol = 1e-10
    if np.allclose(y_lammps, y_expected, atol=tol):
        print(f"PASS (tol={tol})\n")
        return True
    else:
        print(f"FAIL\n")
        return False


def test_layernorm():
    """Test layer normalization."""
    print("=== Layer Norm Test ===")
    n = 8
    rng = np.random.RandomState(55)
    x_input = rng.randn(n) * 2
    gamma = rng.randn(n) * 0.5 + 1.0  # roughly around 1
    beta = rng.randn(n) * 0.1

    y_expected = layer_norm_np(x_input, gamma, beta)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, "atoms.data")
        params_file = os.path.join(tmpdir, "ln_params.dat")
        write_data_file(data_file, n, x_input)
        write_layernorm_params(params_file, n, gamma, beta)

        y_lammps = run_lammps_minimize(data_file, f"""
            fix ln all neural/layernorm {params_file}
            fix_modify ln energy yes
        """)

    print(f"Input:    {x_input}")
    print(f"Gamma:    {gamma}")
    print(f"Beta:     {beta}")
    print(f"Expected: {y_expected}")
    print(f"LAMMPS:   {y_lammps}")
    max_err = np.max(np.abs(y_lammps - y_expected))
    print(f"Max err:  {max_err:.2e}")

    tol = 1e-10
    if np.allclose(y_lammps, y_expected, atol=tol):
        print(f"PASS (tol={tol})\n")
        return True
    else:
        print(f"FAIL\n")
        return False


def test_layernorm_large():
    """Test layer norm with GPT-2-sized dimension (768)."""
    print("=== Layer Norm Large (768-dim) Test ===")
    n = 768
    rng = np.random.RandomState(33)
    x_input = rng.randn(n)
    gamma = rng.randn(n) * 0.02 + 1.0
    beta = rng.randn(n) * 0.02

    y_expected = layer_norm_np(x_input, gamma, beta)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, "atoms.data")
        params_file = os.path.join(tmpdir, "ln_params.dat")
        write_data_file(data_file, n, x_input)
        write_layernorm_params(params_file, n, gamma, beta)

        y_lammps = run_lammps_minimize(data_file, f"""
            fix ln all neural/layernorm {params_file}
            fix_modify ln energy yes
        """)

    max_err = np.max(np.abs(y_lammps - y_expected))
    print(f"Max err:  {max_err:.2e}")

    tol = 1e-10
    if np.allclose(y_lammps, y_expected, atol=tol):
        print(f"PASS (tol={tol})\n")
        return True
    else:
        print(f"FAIL\n")
        return False


if __name__ == "__main__":
    build_dir = os.path.join(os.path.dirname(__file__), '..', 'lammps', 'build')
    if sys.platform == 'darwin':
        os.environ.setdefault('DYLD_LIBRARY_PATH', build_dir)
    else:
        os.environ.setdefault('LD_LIBRARY_PATH', build_dir)

    results = []
    results.append(test_gelu())
    results.append(test_gelu_edge_cases())
    results.append(test_softmax())
    results.append(test_softmax_large())
    results.append(test_layernorm())
    results.append(test_layernorm_large())

    print("=" * 50)
    passed = sum(results)
    total = len(results)
    if all(results):
        print(f"ALL {total} TESTS PASSED")
    else:
        print(f"{passed}/{total} TESTS PASSED")
        sys.exit(1)
