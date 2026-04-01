/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   fix neural/softmax - softmax activation via energy minimization.

   At min_setup: gathers x-positions of all atoms in group across all
   ranks, computes softmax(x), then applies spring forces driving
   each particle to its softmax output.

   softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

   This IS the Boltzmann distribution with energies E_i = -x_i at T=1.

   Usage:
     fix ID group neural/softmax

   Contributing author: lammps-llm project
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(neural/softmax,FixNeuralSoftmax);
// clang-format on
#else

#ifndef LMP_FIX_NEURAL_SOFTMAX_H
#define LMP_FIX_NEURAL_SOFTMAX_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNeuralSoftmax : public Fix {
 public:
  FixNeuralSoftmax(class LAMMPS *, int, char **);
  ~FixNeuralSoftmax() override;
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
};

}    // namespace LAMMPS_NS

#endif
#endif
