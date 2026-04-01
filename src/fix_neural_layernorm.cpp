/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   fix neural/layernorm - layer normalization via energy minimization.

   layer_norm(x) = gamma * (x - mean) / sqrt(var + eps) + beta

   At min_setup: reads group atom x-positions, computes mean/variance
   across the group, normalizes, applies affine transform with
   gamma/beta, stores targets, then springs atoms toward targets.

   Contributing author: lammps-llm project
------------------------------------------------------------------------- */

#include "fix_neural_layernorm.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "text_file_reader.h"

#include <cmath>
#include <cstring>
#include <exception>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNeuralLayerNorm::FixNeuralLayerNorm(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), params(nullptr), target(nullptr)
{
  if (narg != 4) error->all(FLERR, "Illegal fix neural/layernorm command");

  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  energy_global_flag = 1;

  maxatom = 0;
  eps = 1e-5;
  energy = 0.0;
  energy_all = 0.0;
  force_flag = 0;
  nparams = 0;

  // Read gamma/beta parameter file on rank 0, broadcast
  if (comm->me == 0) {
    FILE *fp = utils::open_potential(arg[3], lmp, nullptr);
    if (!fp)
      error->one(FLERR, "Error opening layernorm params file {}: {}",
                 arg[3], utils::getsyserror());
    TextFileReader reader(fp, "layernorm params");

    std::vector<ln_param> myparams;
    char *line;

    try {
      while ((line = reader.next_line())) {
        ValueTokenizer values(line);
        ln_param oneparam;
        oneparam.id = values.next_tagint();
        oneparam.gamma = values.next_double();
        oneparam.beta = values.next_double();
        myparams.push_back(oneparam);
        ++nparams;
      }
    } catch (std::exception &e) {
      error->one(FLERR, "Error reading layernorm params file: {}\n{}", e.what(), line);
    }

    memory->create(params, nparams, "neural_layernorm:params");
    memcpy(params, myparams.data(), nparams * sizeof(ln_param));
    fclose(fp);

    utils::logmesg(lmp, "Read {} layernorm params from {}\n", nparams, arg[3]);
  }

  MPI_Bcast(&nparams, 1, MPI_INT, 0, world);
  if (comm->me != 0)
    memory->create(params, nparams, "neural_layernorm:params");
  MPI_Bcast(params, nparams * sizeof(ln_param), MPI_BYTE, 0, world);
}

/* ---------------------------------------------------------------------- */

FixNeuralLayerNorm::~FixNeuralLayerNorm()
{
  memory->destroy(params);
  memory->destroy(target);
}

/* ---------------------------------------------------------------------- */

int FixNeuralLayerNorm::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNeuralLayerNorm::init()
{
  // nothing needed
}

/* ---------------------------------------------------------------------- */

void FixNeuralLayerNorm::setup(int vflag)
{
  compute_targets();
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixNeuralLayerNorm::min_setup(int vflag)
{
  compute_targets();
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   Gather group x-positions, compute mean/var, normalize, apply
   gamma/beta affine transform, store as targets.
------------------------------------------------------------------------- */

void FixNeuralLayerNorm::compute_targets()
{
  if (atom->nmax > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(target);
    memory->create(target, maxatom, 3, "neural_layernorm:target");
  }

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  // Collect x-values for group atoms
  std::vector<int> group_indices;
  std::vector<double> values;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      group_indices.push_back(i);
      values.push_back(x[i][0]);
    }
  }

  int n = group_indices.size();

  // Compute mean across all ranks
  double local_sum = 0.0;
  for (int k = 0; k < n; k++) local_sum += values[k];

  double global_sum;
  int n_global;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&n, &n_global, 1, MPI_INT, MPI_SUM, world);

  if (n_global == 0) return;

  double mean = global_sum / n_global;

  // Compute variance
  double local_var_sum = 0.0;
  for (int k = 0; k < n; k++) {
    double diff = values[k] - mean;
    local_var_sum += diff * diff;
  }

  double global_var_sum;
  MPI_Allreduce(&local_var_sum, &global_var_sum, 1, MPI_DOUBLE, MPI_SUM, world);

  double variance = global_var_sum / n_global;
  double inv_std = 1.0 / sqrt(variance + eps);

  // Build a map from atom tag -> gamma/beta for fast lookup
  // For each local group atom, find its gamma/beta and compute target
  for (int k = 0; k < n; k++) {
    int i = group_indices[k];
    tagint atag = atom->tag[i];

    // Find gamma/beta for this atom
    double gamma = 1.0;  // default
    double beta = 0.0;   // default
    for (int p = 0; p < nparams; p++) {
      if (params[p].id == atag) {
        gamma = params[p].gamma;
        beta = params[p].beta;
        break;
      }
    }

    double normalized = (values[k] - mean) * inv_std;
    target[i][0] = gamma * normalized + beta;
    target[i][1] = x[i][1];
    target[i][2] = x[i][2];
  }
}

/* ---------------------------------------------------------------------- */

void FixNeuralLayerNorm::post_force(int /*vflag*/)
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

void FixNeuralLayerNorm::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

double FixNeuralLayerNorm::compute_scalar()
{
  if (force_flag == 0) {
    MPI_Allreduce(&energy, &energy_all, 1, MPI_DOUBLE, MPI_SUM, world);
    force_flag = 1;
  }
  return energy_all;
}

/* ---------------------------------------------------------------------- */

void FixNeuralLayerNorm::grow_arrays(int nmax)
{
  memory->grow(target, nmax, 3, "neural_layernorm:target");
  maxatom = nmax;
}

/* ---------------------------------------------------------------------- */

double FixNeuralLayerNorm::memory_usage()
{
  return (double) maxatom * 3 * sizeof(double) + (double) nparams * sizeof(ln_param);
}
