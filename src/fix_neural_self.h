/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   fix neural/self - self-restoring harmonic spring to origin
   V_i = (1/2) * x_i[0]^2, F_i[0] = -x_i[0]

   Used with pair_style neural to implement linear layers:
   at equilibrium, x_i = sum_j(W_ij * x_j) + b_i

   Contributing author: lammps-llm project
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(neural/self,FixNeuralSelf);
// clang-format on
#else

#ifndef LMP_FIX_NEURAL_SELF_H
#define LMP_FIX_NEURAL_SELF_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNeuralSelf : public Fix {
 public:
  FixNeuralSelf(class LAMMPS *, int, char **);
  int setmask() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void min_post_force(int) override;
  double compute_scalar() override;

 private:
  double energy;
  double energy_all;
  int force_flag;
};

}    // namespace LAMMPS_NS

#endif
#endif
