/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   pair_style neural - bilinear coupling potential for neural network
   linear layers: V_ij = -W_ij * x_i * x_j (Ising-model-like coupling)

   Contributing author: lammps-llm project
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(neural,PairNeural);
// clang-format on
#else

#ifndef LMP_PAIR_NEURAL_H
#define LMP_PAIR_NEURAL_H

#include "pair.h"

namespace LAMMPS_NS {

class PairNeural : public Pair {
 public:
  PairNeural(class LAMMPS *);
  ~PairNeural() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  double memory_usage() override;

 protected:
  void allocate();

  struct neural_param {
    tagint id1, id2;   // global atom IDs
    double weight;     // coupling weight W_ij
  };

  neural_param *params;  // list of pair interactions
  int npairs;            // number of pairs
  double cut_global;     // global cutoff (for LAMMPS bookkeeping)
  int check_flag;        // whether to check for missing pairs
};

}    // namespace LAMMPS_NS

#endif
#endif
