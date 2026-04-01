/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   fix neural/gelu - GELU activation function via energy minimization.

   At min_setup: snapshots current positions, computes gelu(x) for the
   x-component target, preserves y,z. Springs all 3 components to
   targets during minimization to avoid CG degeneracy.

   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

   Usage:
     fix ID group neural/gelu

   Contributing author: lammps-llm project
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(neural/gelu,FixNeuralGelu);
// clang-format on
#else

#ifndef LMP_FIX_NEURAL_GELU_H
#define LMP_FIX_NEURAL_GELU_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNeuralGelu : public Fix {
 public:
  FixNeuralGelu(class LAMMPS *, int, char **);
  ~FixNeuralGelu() override;
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
  double **target;    // per-atom 3D targets [nmax][3]
  int maxatom;
  double energy;
  double energy_all;
  int force_flag;

  void compute_targets();
  static double gelu(double x);
};

}    // namespace LAMMPS_NS

#endif
#endif
