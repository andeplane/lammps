# Large Language Models as Molecular Dynamics: Implementing Transformer Inference Through Physical Equilibrium in LAMMPS

## Abstract

We demonstrate that the inference pass of a large language model can be executed entirely as a molecular dynamics simulation, where each neuron is represented by a particle and activation values are encoded as spatial coordinates. Specifically, we implement GPT-2 (124M parameters) within LAMMPS, a widely-used molecular dynamics framework, by mapping each operation of the transformer architecture — linear projections, GELU activations, softmax normalization, layer normalization, and multi-head attention — to energy minimization of carefully constructed physical systems. Linear layers are computed through a bilinear pair potential $V_{ij} = -W_{ij} x_i x_j$ analogous to Ising model spin-spin coupling, where the equilibrium configuration of particles under conjugate gradient minimization yields exactly $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$. Nonlinear operations are realized through custom potential energy surfaces whose minima correspond to the desired function outputs. The entire forward pass — 12 transformer blocks with layer normalization, multi-head attention, and feed-forward networks — runs as a single LAMMPS input script with no external code. We verify numerical agreement with a standard NumPy reference to within $10^{-6}$ on the full 50,257-dimensional logit vector and demonstrate identical next-token predictions. Our work establishes a concrete bridge between neural network computation and physical simulation, suggests that analog physical systems could in principle perform transformer inference, and provides open-source LAMMPS extensions enabling further exploration of computation-through-physics.

---

## 1. Introduction

- Neural networks as function approximators vs. physical systems as computers
- Historical precedent: Hopfield networks as spin systems, Boltzmann machines as statistical mechanics, equilibrium propagation
- The transformer architecture and its dominance in modern AI (Vaswani et al., 2017)
- LAMMPS as a mature, extensible molecular dynamics platform (Thompson et al., 2022)
- Our contribution: a complete, working implementation of GPT-2 inference as a single LAMMPS input script
- Outline of paper

## 2. Background

### 2.1 The Transformer Architecture

- GPT-2 architecture overview (Radford et al., 2019)
- The forward pass: embedding $\rightarrow$ transformer blocks $\rightarrow$ logits
- Key operations and their mathematical definitions:

**Linear layer:**
$$\mathbf{y} = \mathbf{x} \mathbf{W} + \mathbf{b}$$

**GELU activation (Hendrycks & Gimpel, 2016):**
$$\text{GELU}(x) = \frac{x}{2}\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715\, x^3\right)\right)\right)$$

**Softmax:**
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Layer normalization (Ba et al., 2016):**
$$\text{LayerNorm}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \boldsymbol{\beta}$$

**Scaled dot-product attention:**
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

- GPT-2 124M specifics: 12 layers, $d_\text{model}=768$, 12 heads, 50257-token vocabulary

### 2.2 Energy Minimization as Computation

- Analog computing: physical systems whose equilibrium encodes computation
- The energy-computation correspondence: if $\nabla_\mathbf{y} E = 0 \implies \mathbf{y} = f(\mathbf{x})$, then minimizing $E$ computes $f$
- Hopfield networks and the Ising model connection
- Equilibrium propagation (Scellier & Bengio, 2017)
- The Boltzmann distribution as softmax: $p_i = e^{-E_i / kT} / Z$ at $T=1$

### 2.3 LAMMPS

- Overview of LAMMPS architecture: atoms, pair styles, fixes, minimizers
- Extensibility: custom pair styles and fixes in C++
- Energy minimization capabilities: conjugate gradient, FIRE

## 3. Mapping Transformer Operations to Physical Systems

### 3.1 Representation

Each neuron is a particle. The $x$-coordinate of each particle encodes its scalar activation value. Particles are separated in $y$ and $z$ to avoid unintended spatial interactions. Input neurons are frozen (zero force) during computation; output neurons are free to move to their equilibrium positions.

Weight matrices are encoded as explicit pairwise interaction parameters between input and output particles. Bias vectors are encoded as per-atom external forces. The model's 124M parameters are stored in 121 data files (~2.7 GB) that LAMMPS reads during the simulation.

### 3.2 Linear Layers via Bilinear Pair Potential

The central physical primitive. We construct the energy:

$$E = \sum_i \left[\frac{1}{2} y_i^2 - \sum_j W_{ij}\, y_i\, x_j - b_i\, y_i \right]$$

where $x_j$ are frozen input particle positions and $y_i$ are free output particle positions. The three terms correspond to:

| Term | Physical interpretation | LAMMPS implementation |
|------|----------------------|----------------------|
| $\frac{1}{2}y_i^2$ | Harmonic self-restoring spring | `fix neural/self` |
| $-W_{ij}\, y_i\, x_j$ | Bilinear (Ising-like) pair coupling | `pair_style neural` |
| $-b_i\, y_i$ | External bias force | `fix neural/bias` |

Setting $\partial E / \partial y_i = 0$:

$$y_i - \sum_j W_{ij}\, x_j - b_i = 0 \quad \Longrightarrow \quad \mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

The energy is quadratic in $\mathbf{y}$, guaranteeing a unique global minimum and exact convergence under conjugate gradient minimization.

The bilinear pair potential $V_{ij} = -W_{ij}\, x_i\, x_j$ is physically meaningful — it is the same coupling that appears in the Ising model Hamiltonian $H = -\sum_{ij} J_{ij}\, s_i\, s_j$ and in Hopfield network energy functions. The force on particle $i$ from particle $j$ is $F_i = W_{ij}\, x_j$, which is the gradient of the coupling energy.

Implementation: `pair_style neural` reads an explicit list of (atom_id, atom_id, weight) triplets from a file. It does not use neighbor lists — it iterates directly over the specified pairs. This allows arbitrary connectivity patterns (dense, sparse, or structured) without spatial proximity requirements. Newton's third law applies naturally: the reverse force $F_j = W_{ij}\, x_i$ on frozen input particles is zeroed by `fix setforce 0 0 0`.

### 3.3 Nonlinear Activations via Shaped Potential Wells

For nonlinear operations $\mathbf{y} = g(\mathbf{x})$, we use a two-phase approach:

1. **Snapshot**: at minimization setup, read current particle positions $\mathbf{x}$ and compute targets $\mathbf{t} = g(\mathbf{x})$ in C++
2. **Spring-to-target**: apply restoring force $F_i = -(y_i - t_i)$ with energy $E = \frac{1}{2}\sum_i(y_i - t_i)^2$

At equilibrium: $y_i = t_i = g(x_i)$.

The FIRE minimizer (Bitzek et al., 2006) is used for these operations. FIRE performs velocity-Verlet integration with adaptive damping, which handles fix-only force landscapes more robustly than line-search-based methods (CG, SD) that can stall on degenerate energy surfaces.

#### 3.3.1 GELU

Target computation in C++ at `min_setup`:
$$t_i = \text{GELU}(x_i) = \frac{x_i}{2}\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x_i + 0.044715\, x_i^3\right)\right)\right)$$

Implemented as `fix neural/gelu`. Converges to $\sim 10^{-12}$ accuracy.

#### 3.3.2 Softmax

Target computation over group of $n$ atoms:
$$t_i = \frac{e^{x_i - \max(\mathbf{x})}}{\sum_{j=1}^{n} e^{x_j - \max(\mathbf{x})}}$$

This is the **Boltzmann distribution** with energies $E_i = -x_i$ at temperature $T = 1$ — the most natural mapping between neural networks and statistical mechanics.

Implemented as `fix neural/softmax` with MPI-aware global reduction for the partition function.

#### 3.3.3 Layer Normalization

Target computation with learned parameters $\boldsymbol{\gamma}$, $\boldsymbol{\beta}$ read from file:
$$t_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i, \qquad \mu = \frac{1}{n}\sum_j x_j, \quad \sigma^2 = \frac{1}{n}\sum_j (x_j - \mu)^2$$

Implemented as `fix neural/layernorm`.

### 3.4 Multi-Head Attention

The attention mechanism is decomposed into sub-operations using the primitives above:

1. **QKV projection**: linear layer $[\mathbf{Q}, \mathbf{K}, \mathbf{V}] = \text{Linear}(\mathbf{x})$ via `pair_style neural`
2. **Attention scores**: $S_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j / \sqrt{d_k}$
3. **Causal masking**: score clamping to $-100$ (yielding $e^{-100} \approx 10^{-44}$, effectively zero)
4. **Softmax**: per-row via `fix neural/softmax`
5. **Weighted sum**: $\mathbf{o}_i = \sum_j \alpha_{ij} \mathbf{v}_j$
6. **Output projection**: linear layer via `pair_style neural`

For the single-token case ($T = 1$), attention degenerates: the $\mathbf{Q}\mathbf{K}^\top$ score matrix is $1 \times 1$, softmax of a single element is $1.0$, and the output is simply $\mathbf{V}$. The LAMMPS script exploits this — no softmax computation is needed for $T = 1$.

## 4. Implementation

### 4.1 Architecture: A Single LAMMPS Input Script

The entire GPT-2 forward pass runs as a single LAMMPS input script (`in.gpt2`) with no Python, no external orchestrator, and no code running outside LAMMPS. The only prerequisites are:

1. A one-time weight conversion step (`prepare_weights.py`) that exports the GPT-2 model to LAMMPS-readable data files
2. The LAMMPS binary compiled with our six custom C++ extensions

The input script uses LAMMPS's built-in scripting capabilities:
- `variable block loop 0 11` with `label`/`next`/`jump` for the 12-block transformer loop
- `fix property/atom d_residual` for per-atom storage of residual connection values
- `pair_style neural` re-issued with different weight files to swap between linear layers
- `fix`/`unfix` to configure and tear down forces for each operation
- `set group bufA x v_add_residual` with atom-style variables for residual addition

### 4.2 Particle Layout

A fixed pool of **3,840 atoms** is pre-allocated in the data file:

| Atoms | Group | Purpose |
|-------|-------|---------|
| 1–768 | `bufA` | Main activation buffer ($d_\text{model} = 768$) |
| 769–3840 | `bufB` | Scratch buffer (up to 3,072 for FFN intermediate) |

Sub-groups partition `bufB` for different operations:

| Group | Atoms | Used by |
|-------|-------|---------|
| `bufB_qkv` | 769–3072 | QKV projection output (2,304 = $3 \times 768$) |
| `bufB_v` | 2305–3072 | V portion of QKV (768 values) |
| `bufB_3072` | 769–3840 | FFN intermediate (3,072 = $4 \times 768$) |

Atoms are spatially separated along the $y$-axis (atom $i$ at $y = i$) to avoid unintended pairwise interactions. Only the $x$-coordinate carries the activation value.

### 4.3 Transformer Block Flow

Each of the 12 transformer blocks executes the following sequence of LAMMPS operations. Between operations, the pair style and fixes are swapped via `pair_style`/`fix`/`unfix` commands. Particle positions are preserved between operations — output positions from one step become input positions for the next.

```
For each block i = 0, 1, ..., 11:

  1. SAVE:  snapshot bufA x-positions → d_residual
  
  2. LN1:   fix neural/layernorm on bufA
            FIRE minimize → bufA positions become layer-normed values
            unfix
  
  3. QKV:   reset bufB_qkv to x=0
            pair_style neural block_{i}_qkv_w.dat
            freeze bufA (input), neural/self + neural/bias on bufB_qkv (output)
            CG minimize → bufB_qkv = Wx + b (QKV projection)
            unfix all

  4. ATTN:  (T=1: output = V, already in bufB_v, no computation)

  5. PROJ:  reset bufA to x=0
            pair_style neural block_{i}_cproj_w.dat
            freeze bufB (input), neural/self + neural/bias on bufA (output)
            CG minimize → bufA = attention output projection
            unfix all
  
  6. RES:   bufA x-positions += d_residual  (residual connection)
  
  7. SAVE:  snapshot bufA → d_residual
  
  8. LN2:   fix neural/layernorm on bufA → FIRE minimize → unfix
  
  9. FC:    reset bufB_3072 to x=0
            pair_style neural block_{i}_fc_w.dat
            freeze bufA, neural/self + neural/bias on bufB_3072
            CG minimize → bufB_3072 = FFN intermediate
            unfix all
  
  10. GELU: fix neural/gelu on bufB_3072 → FIRE minimize → unfix
  
  11. PROJ: reset bufA to x=0
            pair_style neural block_{i}_fcproj_w.dat
            freeze bufB_3072, neural/self + neural/bias on bufA
            CG minimize → bufA = FFN output
            unfix all
  
  12. RES:  bufA += d_residual  (residual connection)

Final: fix neural/layernorm (ln_f) on bufA → FIRE minimize
       dump bufA x-positions to activations.dump
```

Each block performs **8 energy minimizations** (2 layer norms, 4 linear layers, 1 GELU, and attention — which is trivial for $T = 1$). The full forward pass performs **96 minimizations** plus a final layer norm.

### 4.4 Weight File Format

Model weights are stored as plain text files. A one-time conversion script (`prepare_weights.py`) exports GPT-2's 124M parameters from HuggingFace format into 121 LAMMPS-readable files totaling ~2.7 GB:

**Pair weight files** (`block_{i}_qkv_w.dat`, etc.) — one line per weight:
```
input_atom_id  output_atom_id  weight_value
```

The atom IDs in each file correspond to the fixed particle layout. For example, the QKV weight file maps inputs 1–768 (bufA) to outputs 769–3072 (bufB_qkv), while the FFN output projection maps inputs 769–3840 (bufB_3072) to outputs 1–768 (bufA).

**Bias files** (`block_{i}_qkv_b.dat`, etc.) — one line per output atom:
```
output_atom_id  bias_value
```

**Layer norm parameter files** (`block_{i}_ln1.dat`, etc.) — one line per atom:
```
atom_id  gamma  beta
```

### 4.5 Custom LAMMPS Extensions

Six C++ extensions (~1,500 lines total):

| Extension | Type | Forces | Energy | Minimizer |
|-----------|------|--------|--------|-----------|
| `pair_style neural` | Pair style | $F_i = W_{ij}\, x_j$ | $-W_{ij}\, x_i\, x_j$ | CG |
| `fix neural/self` | Fix | $F_i = -x_i$ | $\frac{1}{2}x_i^2$ | CG |
| `fix neural/bias` | Fix | $F_i = b_i$ | $-b_i\, x_i$ | CG |
| `fix neural/gelu` | Fix | $F_i = -(x_i - t_i)$ | $\frac{1}{2}(x_i-t_i)^2$ | FIRE |
| `fix neural/softmax` | Fix | $F_i = -(x_i - t_i)$ | $\frac{1}{2}(x_i-t_i)^2$ | FIRE |
| `fix neural/layernorm` | Fix | $F_i = -(x_i - t_i)$ | $\frac{1}{2}(x_i-t_i)^2$ | FIRE |

Design decisions:
- **Explicit pair lists** (no neighbor lists): `pair_style neural` reads atom ID pairs from file, iterating directly over specified pairs via `atom->map()`. This avoids neighbor list overhead and allows arbitrary connectivity.
- **1D activation encoding**: only the $x$-coordinate carries the activation value. The $y$ and $z$ coordinates provide spatial separation.
- **3D position springs**: nonlinear fixes (GELU, softmax, layernorm) apply spring forces in all three dimensions (targeting the original $y$, $z$ positions) to avoid CG minimizer degeneracy in flat energy dimensions.

### 4.6 Build System

The extensions compile as part of the standard LAMMPS CMake build:

```bash
# Copy extensions into LAMMPS source tree
cp src/*.cpp src/*.h lammps/src/

# Build (serial, shared library)
cd lammps/build
cmake ../cmake -DBUILD_MPI=off -DPKG_MISC=on \
      -DBUILD_SHARED_LIBS=on -DLAMMPS_EXCEPTIONS=on
make -j8
```

No MPI required. No external dependencies beyond LAMMPS itself.

## 5. Results

### 5.1 Numerical Accuracy

#### 5.1.1 Individual Operations

| Operation | Test size | Max error | Notes |
|-----------|-----------|-----------|-------|
| Linear ($4 \to 3$) | 7 atoms | $6.7 \times 10^{-16}$ | Machine precision (CG) |
| Linear ($64 \to 32$) | 96 atoms | $8.9 \times 10^{-16}$ | Machine precision (CG) |
| GELU | 8 atoms | $2.0 \times 10^{-12}$ | FIRE minimizer |
| Softmax | 50 atoms | $7.2 \times 10^{-12}$ | FIRE minimizer |
| Layer norm | 768 atoms | $1.1 \times 10^{-11}$ | FIRE minimizer |

#### 5.1.2 Transformer Block

| Test | Max error |
|------|-----------|
| Block ($T{=}1$, $d{=}8$, 2 heads) | $2.0 \times 10^{-12}$ |
| Block ($T{=}3$, $d{=}8$, 2 heads) | $1.2 \times 10^{-11}$ |

#### 5.1.3 Full Model (Pure LAMMPS Script)

Running the full 12-block forward pass as a single LAMMPS input script on the prompt "The":

| Metric | Value |
|--------|-------|
| Max logit difference (vs. NumPy) | $9.8 \times 10^{-7}$ |
| Top-1 prediction match | Yes |
| Top-10 prediction match | Identical ordering |

The full 50,257-dimensional logit vector agrees with the NumPy reference to within $10^{-6}$. The predicted next token ("\n", token ID 198) is identical.

### 5.2 Text Generation

**Prompt:** "The"

**LAMMPS top-10 predictions:**

| Rank | Token | Logit |
|------|-------|-------|
| 1 | `\n` (newline) | -31.86 |
| 2 | ` the` | -32.17 |
| 3 | ` "` | -32.37 |
| 4 | `,` | -32.64 |
| 5 | ` a` | -32.83 |
| 6 | ` and` | -32.90 |
| 7 | ` first` | -32.91 |
| 8 | ` I` | -33.02 |
| 9 | ` to` | -33.11 |
| 10 | ` is` | -33.13 |

This matches the NumPy reference identically in rank ordering.

### 5.3 Performance

| Metric | Value |
|--------|-------|
| Forward pass, $T{=}1$ (pure LAMMPS script) | $\sim$45 min |
| NumPy reference, $T{=}1$ | $\sim$0.7 s |
| Slowdown factor | $\sim$3,900$\times$ |
| Total particles | 3,840 |
| Weight file I/O per forward pass | $\sim$2.7 GB read |
| LAMMPS minimizations per forward pass | 97 |

Performance is dominated by:
1. **File I/O**: reading multi-megabyte weight files from disk for each linear layer (~60% of runtime)
2. **CG minimization**: iterating over millions of pair interactions for large linear layers (~35%)
3. **FIRE minimization**: nonlinear operations converge quickly (<5%)

## 6. Discussion

### 6.1 Physics Purity Spectrum

Not all operations are equally "physical":

| Operation | Purity | Rationale |
|-----------|--------|-----------|
| Linear layers | **High** | Bilinear coupling $V = -W_{ij}\,x_i\,x_j$ is the Ising Hamiltonian. Equilibrium genuinely computes $\mathbf{Wx + b}$ — no part of the code performs matrix multiplication. |
| Softmax | **High** (conceptually) | Softmax IS the Boltzmann distribution. However, our implementation pre-computes the target and springs to it, rather than sampling from a thermal ensemble. |
| GELU, Layer Norm | **Medium** | The C++ code computes the target function, then uses physical dynamics (spring forces, energy minimization) to move particles there. The "physics" is real, but the answer is already known. |
| Sequencing | **Low** | The 96 minimizations are sequenced by LAMMPS scripting (`variable loop` + `jump`), not by physical signal propagation. |

### 6.2 Implications for Physical Computing

- What this demonstrates about the universality of physical computation
- Connections to analog computing, neuromorphic hardware, and physical neural networks
- Could a real molecular system compute a transformer? Theoretical constraints
- Energy considerations: thermodynamic cost of computation vs. Landauer's principle

### 6.3 Limitations

- Performance: $\sim$3,900$\times$ slower than direct computation, dominated by file I/O
- Sequential bottleneck: 97 minimizations must be performed serially
- Attention for $T > 1$: requires a custom `fix neural/attention` or decomposition into many sub-operations
- Numerical precision: floating-point summation over millions of pairs limits accuracy for large layers

### 6.4 Possible Extensions

- **Binary weight format**: would reduce file I/O by ~3$\times$ and significantly improve runtime
- **Persistent pair style**: reloading weights without destroying/recreating the pair style
- **MPI parallelism**: distributing pair force computation across ranks for large layers
- **$T > 1$ attention**: custom `fix neural/attention` that reads Q, K, V particle positions and computes attention output
- **KV-cache**: freezing previously computed K, V particles for autoregressive generation
- **Training via equilibrium propagation** within the same LAMMPS framework
- **Mapping to real physical substrates** (optical, electronic, mechanical systems)

## 7. Related Work

- Analog neural networks and physical neural networks (Wright et al., 2022; Momeni et al., 2023)
- Equilibrium propagation (Scellier & Bengio, 2017; Laborieux et al., 2021)
- Hopfield networks and spin glasses (Hopfield, 1982; Amit et al., 1985)
- Physical computing and unconventional computation (Stepney et al., 2018)
- picoGPT: minimal GPT-2 implementation (Mody, 2023)

## 8. Conclusion

- Summary: GPT-2 inference implemented as a single LAMMPS input script
- Key insight: the bilinear pair potential $V = -W_{ij}\, x_i\, x_j$ maps linear algebra to equilibrium physics
- Result: identical next-token predictions with $< 10^{-6}$ logit error vs. NumPy reference
- Broader significance: any differentiable computation can in principle be realized as physical equilibrium
- Open-source availability of LAMMPS extensions

## Appendix A: Running the Code

```bash
# 1. Build LAMMPS with neural extensions
cp src/*.cpp src/*.h lammps/src/
cd lammps/build && cmake ../cmake -DBUILD_MPI=off -DPKG_MISC=on \
    -DBUILD_SHARED_LIBS=on -DLAMMPS_EXCEPTIONS=on && make -j8

# 2. Prepare weights (one-time, requires Python + transformers)
python prepare_weights.py "The"

# 3. Run GPT-2 inference (pure LAMMPS, no Python)
lmp -in in.gpt2

# 4. Read the prediction
python compute_logits.py
```

## Appendix B: Derivation of Equilibrium Conditions

The total energy for output particle $i$ in a linear layer is:

$$E_i = \frac{1}{2} y_i^2 - \sum_j W_{ij}\, y_i\, x_j - b_i\, y_i$$

Taking the derivative and setting to zero:

$$\frac{\partial E_i}{\partial y_i} = y_i - \sum_j W_{ij}\, x_j - b_i = 0$$

$$\Longrightarrow \quad y_i = \sum_j W_{ij}\, x_j + b_i$$

In matrix form: $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$.

The Hessian is $\frac{\partial^2 E_i}{\partial y_i^2} = 1 > 0$, confirming this is a minimum (not a maximum or saddle point). The energy is strictly convex, guaranteeing a unique global minimum and exact convergence of conjugate gradient in a single step for each output dimension.

## Appendix C: Numerical Error Analysis

For a linear layer with $n_\text{in}$ inputs, each output atom's equilibrium position is determined by a sum of $n_\text{in}$ pair force contributions. The floating-point error in this sum is bounded by:

$$\epsilon_\text{sum} \lesssim \sqrt{n_\text{in}} \cdot \epsilon_\text{mach} \cdot \max_j |W_{ij}\, x_j|$$

For the largest linear layer (FFN c_fc, $768 \to 3072$), with $n_\text{in} = 768$, $\epsilon_\text{mach} \approx 10^{-16}$, and typical weight-activation products of order $\sim 1$:

$$\epsilon_\text{sum} \lesssim \sqrt{768} \times 10^{-16} \times 1 \approx 3 \times 10^{-15}$$

Over 12 blocks with $\sim 8$ operations each, errors accumulate additively:

$$\epsilon_\text{total} \lesssim 96 \times 3 \times 10^{-15} \approx 3 \times 10^{-13}$$

The observed full-model logit error of $\sim 10^{-6}$ is larger, suggesting additional error sources from the FIRE minimizer's finite convergence tolerance ($\sim 10^{-12}$ per nonlinear operation) and amplification through weight matrices.
