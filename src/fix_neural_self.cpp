/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   fix neural/self - self-restoring harmonic spring to origin.
   Applies F_i[0] = -x_i[0] to all atoms in group.
   Energy: E_i = (1/2) * x_i[0]^2

   Combined with pair_style neural (bilinear coupling) and an external
   bias force, this gives equilibrium at y = Wx + b for linear layers.

   Usage:
     fix ID group neural/self

   Contributing author: lammps-llm project
------------------------------------------------------------------------- */

#include "fix_neural_self.h"

#include "atom.h"
#include "error.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNeuralSelf::FixNeuralSelf(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR, "Illegal fix neural/self command");

  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  energy_global_flag = 1;

  energy = 0.0;
  energy_all = 0.0;
  force_flag = 0;
}

/* ---------------------------------------------------------------------- */

int FixNeuralSelf::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNeuralSelf::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixNeuralSelf::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixNeuralSelf::post_force(int /*vflag*/)
{
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  energy = 0.0;
  force_flag = 0;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      // self-restoring force: F = -x (x-component only)
      f[i][0] += -x[i][0];

      // energy: E = 0.5 * x^2
      energy += 0.5 * x[i][0] * x[i][0];
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixNeuralSelf::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   return total potential energy from self-restoring springs
------------------------------------------------------------------------- */

double FixNeuralSelf::compute_scalar()
{
  if (force_flag == 0) {
    MPI_Allreduce(&energy, &energy_all, 1, MPI_DOUBLE, MPI_SUM, world);
    force_flag = 1;
  }
  return energy_all;
}
