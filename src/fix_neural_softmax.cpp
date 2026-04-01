/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   fix neural/softmax - softmax activation via energy minimization.

   At min_setup: gathers x-positions of all atoms in group, computes
   softmax, then applies spring forces F = -(x - target) to drive
   particles to softmax outputs.

   For single-rank (no MPI) operation: straightforward gather over
   local atoms in the group.

   Contributing author: lammps-llm project
------------------------------------------------------------------------- */

#include "fix_neural_softmax.h"

#include "atom.h"
#include "error.h"
#include "memory.h"

#include <cmath>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNeuralSoftmax::FixNeuralSoftmax(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), target(nullptr)
{
  if (narg != 3) error->all(FLERR, "Illegal fix neural/softmax command");

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

FixNeuralSoftmax::~FixNeuralSoftmax()
{
  memory->destroy(target);
}

/* ---------------------------------------------------------------------- */

int FixNeuralSoftmax::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNeuralSoftmax::init()
{
  // nothing needed
}

/* ---------------------------------------------------------------------- */

void FixNeuralSoftmax::setup(int vflag)
{
  compute_targets();
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixNeuralSoftmax::min_setup(int vflag)
{
  compute_targets();
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   Gather x-positions of group atoms, compute softmax, store targets.
   Uses MPI_Allreduce to handle distributed atoms.
------------------------------------------------------------------------- */

void FixNeuralSoftmax::compute_targets()
{
  if (atom->nmax > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(target);
    memory->create(target, maxatom, 3, "neural_softmax:target");
  }

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  // Collect logits and indices for atoms in this group
  std::vector<int> group_indices;
  std::vector<double> logits;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      group_indices.push_back(i);
      logits.push_back(x[i][0]);
    }
  }

  int n = group_indices.size();
  if (n == 0) return;

  int n_global = 0;
  MPI_Allreduce(&n, &n_global, 1, MPI_INT, MPI_SUM, world);

  // Find global max for numerical stability
  double local_max = -1e300;
  for (int k = 0; k < n; k++)
    if (logits[k] > local_max) local_max = logits[k];

  double global_max;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, world);

  // Compute local sum of exp(x - max)
  std::vector<double> exp_vals(n);
  double local_sum = 0.0;
  for (int k = 0; k < n; k++) {
    exp_vals[k] = exp(logits[k] - global_max);
    local_sum += exp_vals[k];
  }

  double global_sum;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, world);

  // Compute softmax targets (x-component), preserve y,z
  for (int k = 0; k < n; k++) {
    int i = group_indices[k];
    target[i][0] = exp_vals[k] / global_sum;
    target[i][1] = x[i][1];
    target[i][2] = x[i][2];
  }
}

/* ---------------------------------------------------------------------- */

void FixNeuralSoftmax::post_force(int /*vflag*/)
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

void FixNeuralSoftmax::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

double FixNeuralSoftmax::compute_scalar()
{
  if (force_flag == 0) {
    MPI_Allreduce(&energy, &energy_all, 1, MPI_DOUBLE, MPI_SUM, world);
    force_flag = 1;
  }
  return energy_all;
}

/* ---------------------------------------------------------------------- */

void FixNeuralSoftmax::grow_arrays(int nmax)
{
  memory->grow(target, nmax, 3, "neural_softmax:target");
  maxatom = nmax;
}

/* ---------------------------------------------------------------------- */

double FixNeuralSoftmax::memory_usage()
{
  return (double) maxatom * 3 * sizeof(double);
}
