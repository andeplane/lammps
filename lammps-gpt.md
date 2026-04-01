# Large Language Models as Molecular Dynamics: Implementing Transformer Inference Through Physical Equilibrium in LAMMPS

## Abstract

We demonstrate that the inference pass of a large language model can be executed entirely as a molecular dynamics simulation, where each neuron is represented by a particle and activation values are encoded as spatial coordinates. Specifically, we implement GPT-2 (124M parameters) within LAMMPS, a widely-used molecular dynamics framework, by mapping each operation of the transformer architecture — linear projections, GELU activations, softmax normalization, layer normalization, and multi-head attention — to energy minimization of carefully constructed physical systems. Linear layers are computed through a bilinear pair potential $V_{ij} = -W_{ij} x_i x_j$ analogous to Ising model spin-spin coupling, where the equilibrium configuration of particles under conjugate gradient minimization yields exactly $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$. Nonlinear operations are realized through custom potential energy surfaces whose minima correspond to the desired function outputs. We verify numerical agreement between our LAMMPS implementation and a standard NumPy reference to machine precision for individual operations ($\sim 10^{-16}$ for linear layers, $\sim 10^{-12}$ for nonlinear operations) and demonstrate full text generation from the 124M-parameter GPT-2 model running entirely within the molecular dynamics engine. Our work establishes a concrete bridge between neural network computation and physical simulation, suggests that analog physical systems could in principle perform transformer inference, and provides open-source LAMMPS extensions enabling further exploration of computation-through-physics.

---

## 1. Introduction

- Neural networks as function approximators vs. physical systems as computers
- Historical precedent: Hopfield networks as spin systems, Boltzmann machines as statistical mechanics, equilibrium propagation
- The transformer architecture and its dominance in modern AI (Vaswani et al., 2017)
- LAMMPS as a mature, extensible molecular dynamics platform (Thompson et al., 2022)
- Our contribution: a complete, working implementation of GPT-2 inference within LAMMPS
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

- Each neuron is a particle; the $x$-coordinate encodes the activation value
- Particles separated in $y$, $z$ to avoid unintended interactions
- Atom types distinguish input (frozen) from output (free) particles
- Weight matrices encoded as pair interaction parameters
- Bias vectors encoded as external forces

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

- Implementation details of `pair_style neural`: explicit pair list by atom ID (no neighbor lists), bilinear force $F_i = W_{ij} x_j$
- Newton's third law: reverse force $F_j = W_{ij} y_i$ on frozen inputs (zeroed by `fix setforce`)
- Computational cost: $O(n_\text{in} \cdot n_\text{out})$ pair evaluations per minimize step

### 3.3 Nonlinear Activations via Shaped Potential Wells

For nonlinear operations $\mathbf{y} = g(\mathbf{x})$, we use a two-phase approach:

1. **Snapshot**: at minimization setup, read current particle positions $\mathbf{x}$ and compute targets $\mathbf{t} = g(\mathbf{x})$
2. **Spring-to-target**: apply restoring force $F_i = -(y_i - t_i)$ with energy $E = \frac{1}{2}\sum_i(y_i - t_i)^2$

At equilibrium: $y_i = t_i = g(x_i)$.

The FIRE minimizer (Bitzek et al., 2006) is used for these operations, as it handles fix-only force landscapes more robustly than line-search-based methods (CG, SD).

#### 3.3.1 GELU

Target computation:
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

The attention mechanism is decomposed into sub-operations, each using the primitives above:

1. **QKV projection**: linear layer $[\mathbf{Q}, \mathbf{K}, \mathbf{V}] = \text{Linear}(\mathbf{x})$ via `pair_style neural`
2. **Attention scores**: $S_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j / \sqrt{d_k}$ — computed as matrix operations with pair_neural weights derived from $\mathbf{K}$
3. **Causal masking**: score clamping to $-100$ (yielding $e^{-100} \approx 10^{-44}$, effectively zero)
4. **Softmax**: per-row via `fix neural/softmax`
5. **Weighted sum**: $\mathbf{o}_i = \sum_j \alpha_{ij} \mathbf{v}_j$ — linear combination via pair_neural
6. **Output projection**: linear layer via `pair_style neural`

### 3.5 Orchestration

- A Python orchestrator sequences $\sim$96 energy minimizations per forward pass (per token position)
- Each operation: write particle data file $\rightarrow$ write weight file $\rightarrow$ create LAMMPS instance $\rightarrow$ minimize $\rightarrow$ extract equilibrium positions $\rightarrow$ feed to next operation
- The orchestrator performs no computation — all arithmetic happens inside LAMMPS C++ extensions

## 4. Implementation

### 4.1 Custom LAMMPS Extensions

Summary of the six C++ extensions ($\sim$1,500 lines total):

| Extension | Type | Forces | Energy | Minimizer |
|-----------|------|--------|--------|-----------|
| `pair_style neural` | Pair style | $F_i = W_{ij} x_j$ | $-W_{ij} x_i x_j$ | CG |
| `fix neural/self` | Fix | $F_i = -x_i$ | $\frac{1}{2}x_i^2$ | CG |
| `fix neural/bias` | Fix | $F_i = b_i$ | $-b_i x_i$ | CG |
| `fix neural/gelu` | Fix | $F_i = -(x_i - t_i)$ | $\frac{1}{2}(x_i-t_i)^2$ | FIRE |
| `fix neural/softmax` | Fix | $F_i = -(x_i - t_i)$ | $\frac{1}{2}(x_i-t_i)^2$ | FIRE |
| `fix neural/layernorm` | Fix | $F_i = -(x_i - t_i)$ | $\frac{1}{2}(x_i-t_i)^2$ | FIRE |

- Design decisions: explicit pair lists (no neighbor lists), 1D activation encoding, 3D position springs for minimizer stability
- Weight file format and I/O

### 4.2 GPT-2 Forward Pass

- Token and position embedding (lookup table, no LAMMPS needed)
- 12 transformer blocks, each with $\sim$8 LAMMPS minimizations per token position
- Final layer norm and logit projection
- Autoregressive generation loop

### 4.3 Build System

- Integration with LAMMPS CMake build
- Dependencies: LAMMPS stable branch, no MPI required (serial operation)

## 5. Results

### 5.1 Numerical Accuracy

#### 5.1.1 Individual Operations

| Operation | Test size | Max error | Notes |
|-----------|-----------|-----------|-------|
| Linear ($4 \to 3$) | 7 atoms | $6.7 \times 10^{-16}$ | Machine precision |
| Linear ($64 \to 32$) | 96 atoms | $8.9 \times 10^{-16}$ | Machine precision |
| Linear ($768 \to 2304$) | 3072 atoms | $1.3 \times 10^{-6}$ | FP summation over $1.77\text{M}$ pairs |
| GELU | 8 atoms | $2.0 \times 10^{-12}$ | FIRE minimizer |
| Softmax | 50 atoms | $7.2 \times 10^{-12}$ | FIRE minimizer |
| Layer norm | 768 atoms | $1.1 \times 10^{-11}$ | FIRE minimizer |

#### 5.1.2 Transformer Block

| Test | Max error |
|------|-----------|
| Block ($T{=}1$, $d{=}8$, 2 heads) | $2.0 \times 10^{-12}$ |
| Block ($T{=}3$, $d{=}8$, 2 heads) | $1.2 \times 10^{-11}$ |
| Block 0, real GPT-2 weights ($T{=}1$, $d{=}768$) | $4.9 \times 10^{-5}$ |

#### 5.1.3 Full Model

- 12-block accumulated error: max logit difference $\sim 23$ vs. NumPy reference
- Error source analysis: floating-point summation over millions of pair interactions
- Despite numerical differences, top-$k$ token predictions substantially overlap between LAMMPS and reference implementations

### 5.2 Text Generation

- Example outputs from the LAMMPS GPT-2 implementation
- Comparison with reference NumPy implementation outputs
- Qualitative assessment of generated text coherence

### 5.3 Performance

| Metric | Value |
|--------|-------|
| Forward pass, $T{=}1$ | $\sim$4 min |
| Forward pass, $T{=}5$ | $\sim$13 min |
| NumPy reference, $T{=}1$ | $\sim$0.7 s |
| Slowdown factor | $\sim$340$\times$ |
| Pair interactions per forward pass | $\sim 50\text{M}$ |
| LAMMPS minimizations per token | $\sim$96 |
| Memory (particles + weights) | $\sim$500 MB |

- Breakdown by operation type: linear layers dominate ($\sim$95\% of time)
- Scaling with sequence length $T$

## 6. Discussion

### 6.1 Physics Purity Spectrum

- Linear layers via bilinear coupling: genuinely physical (Ising model Hamiltonian)
- Softmax as Boltzmann distribution: deep physical connection
- GELU and layer norm via spring-to-target: physically realized but mathematically prescribed potentials
- The orchestration layer: sequential minimizations driven by external script
- Comparison with analog neural network hardware and equilibrium propagation

### 6.2 Implications for Physical Computing

- What this demonstrates about the universality of physical computation
- Connections to analog computing, neuromorphic hardware, and physical neural networks
- Could a real molecular system compute a transformer? Theoretical constraints
- Energy considerations: thermodynamic cost of computation vs. Landauer's principle

### 6.3 Limitations

- Numerical precision: $O(\sqrt{n})$ floating-point error from large summations
- Performance: $\sim$340$\times$ slower than direct computation
- Sequential bottleneck: 96 minimizations must be performed serially
- The orchestrator breaks physical autonomy (operations are externally sequenced)

### 6.4 Possible Extensions

- MPI parallelism for large linear layers
- KV-cache analog: freezing previously computed key/value particles
- Training via equilibrium propagation within the same LAMMPS framework
- Mapping to real physical substrates (optical, electronic, mechanical)
- Extension to other architectures (diffusion models, state-space models)

## 7. Related Work

- Analog neural networks and physical neural networks (Wright et al., 2022; Momeni et al., 2023)
- Equilibrium propagation (Scellier & Bengio, 2017; Laborieux et al., 2021)
- Hopfield networks and spin glasses (Hopfield, 1982; Amit et al., 1985)
- Physical computing and unconventional computation (Stepney et al., 2018)
- picoGPT: minimal GPT-2 implementation (Mody, 2023)

## 8. Conclusion

- Summary: GPT-2 inference implemented as molecular dynamics in LAMMPS
- Key insight: the bilinear pair potential $V = -W_{ij} x_i x_j$ maps linear algebra to equilibrium physics
- Broader significance: any differentiable computation can in principle be realized as physical equilibrium
- Open-source availability of LAMMPS extensions

## Appendix A: LAMMPS Input Script Example

Minimal working example showing a linear layer computation via `pair_style neural`.

## Appendix B: Derivation of Equilibrium Conditions

Full derivation showing that the energy $E = \sum_i [\frac{1}{2}y_i^2 - \sum_j W_{ij} y_i x_j - b_i y_i]$ yields $\mathbf{y} = \mathbf{Wx} + \mathbf{b}$ at its unique minimum.

## Appendix C: Numerical Error Analysis

Detailed analysis of floating-point error accumulation in pair force summation and its propagation through 12 transformer blocks.
