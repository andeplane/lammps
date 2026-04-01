/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   fix neural/gelu - GELU activation function via energy minimization.

   At min_setup: snapshots all positions, computes gelu(x_i[0]) as
   x-target, keeps y,z as targets. During minimization, applies
   spring forces F = -(pos - target) in all 3 dimensions.

   This avoids CG minimizer degeneracy from flat y,z energy landscape.

   Contributing author: lammps-llm project
------------------------------------------------------------------------- */

#include "fix_neural_gelu.h"

#include "atom.h"
#include "error.h"
#include "memory.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNeuralGelu::FixNeuralGelu(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), target(nullptr)
{
  if (narg != 3) error->all(FLERR, "Illegal fix neural/gelu command");

  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  energy_global_flag = 1;

  maxatom = 0;
  energy = 0.0;
  energy_all = 0.0;
  force_flag = 0;
}

/* ---------------------------------------------------------------------- */

FixNeuralGelu::~FixNeuralGelu()
{
  memory->destroy(target);
}

/* ---------------------------------------------------------------------- */

int FixNeuralGelu::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNeuralGelu::init() {}

/* ---------------------------------------------------------------------- */

void FixNeuralGelu::setup(int vflag)
{
  compute_targets();
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixNeuralGelu::min_setup(int vflag)
{
  compute_targets();
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

double FixNeuralGelu::gelu(double x)
{
  const double SQRT_2_OVER_PI = 0.7978845608028654;
  return 0.5 * x * (1.0 + tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)));
}

/* ---------------------------------------------------------------------- */

void FixNeuralGelu::compute_targets()
{
  if (atom->nmax > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(target);
    memory->create(target, maxatom, 3, "neural_gelu:target");
  }

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      target[i][0] = gelu(x[i][0]);   // x-component: GELU transform
      target[i][1] = x[i][1];         // y-component: stay in place
      target[i][2] = x[i][2];         // z-component: stay in place
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixNeuralGelu::post_force(int /*vflag*/)
{
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  energy = 0.0;
  force_flag = 0;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      for (int d = 0; d < 3; d++) {
        double dd = x[i][d] - target[i][d];
        f[i][d] += -dd;
        energy += 0.5 * dd * dd;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixNeuralGelu::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

double FixNeuralGelu::compute_scalar()
{
  if (force_flag == 0) {
    MPI_Allreduce(&energy, &energy_all, 1, MPI_DOUBLE, MPI_SUM, world);
    force_flag = 1;
  }
  return energy_all;
}

/* ---------------------------------------------------------------------- */

void FixNeuralGelu::grow_arrays(int nmax)
{
  memory->grow(target, nmax, 3, "neural_gelu:target");
  maxatom = nmax;
}

/* ---------------------------------------------------------------------- */

double FixNeuralGelu::memory_usage()
{
  return (double) maxatom * 3 * sizeof(double);
}
