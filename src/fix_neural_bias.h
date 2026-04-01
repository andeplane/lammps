/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   fix neural/bias - per-atom bias force for neural network linear layers.
   Reads a file mapping atom IDs to bias values.
   Applies F_i[0] = b_i to each specified atom.

   Contributing author: lammps-llm project
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(neural/bias,FixNeuralBias);
// clang-format on
#else

#ifndef LMP_FIX_NEURAL_BIAS_H
#define LMP_FIX_NEURAL_BIAS_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNeuralBias : public Fix {
 public:
  FixNeuralBias(class LAMMPS *, int, char **);
  ~FixNeuralBias() override;
  int setmask() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void min_post_force(int) override;
  double compute_scalar() override;

 private:
  struct bias_param {
    tagint id;       // global atom ID
    double bias;     // bias value
  };

  bias_param *params;
  int nbias;
  double energy;
  double energy_all;
  int force_flag;
};

}    // namespace LAMMPS_NS

#endif
#endif
