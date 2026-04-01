/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   fix neural/layernorm - layer normalization via energy minimization.

   At min_setup: gathers x-positions of all group atoms, computes
   layer_norm(x, gamma, beta), stores targets, then applies spring
   forces during minimization.

   layer_norm(x) = gamma * (x - mean) / sqrt(var + eps) + beta

   Gamma and beta are read from a parameter file.

   Parameter file format (one atom per line):
     atom_id  gamma  beta

   Usage:
     fix ID group neural/layernorm params.dat

   Contributing author: lammps-llm project
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(neural/layernorm,FixNeuralLayerNorm);
// clang-format on
#else

#ifndef LMP_FIX_NEURAL_LAYERNORM_H
#define LMP_FIX_NEURAL_LAYERNORM_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNeuralLayerNorm : public Fix {
 public:
  FixNeuralLayerNorm(class LAMMPS *, int, char **);
  ~FixNeuralLayerNorm() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void min_post_force(int) override;
  double compute_scalar() override;
  void grow_arrays(int) override;
  double memory_usage() override;

 private:
  struct ln_param {
    tagint id;
    double gamma;
    double beta;
  };

  ln_param *params;   // per-atom gamma/beta from file
  int nparams;
  double **target;    // per-atom 3D targets [nmax][3]
  int maxatom;
  double eps;         // epsilon for numerical stability
  double energy;
  double energy_all;
  int force_flag;

  void compute_targets();
};

}    // namespace LAMMPS_NS

#endif
#endif
