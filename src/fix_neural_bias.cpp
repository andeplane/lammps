/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   fix neural/bias - per-atom bias force for neural network linear layers.
   Applies F_i[0] = b_i to each atom specified in a bias file.
   Energy contribution: E_i = -b_i * x_i[0]

   Bias file format (one atom per line):
     atom_id  bias_value

   Usage:
     fix ID group neural/bias bias_file.dat

   Contributing author: lammps-llm project
------------------------------------------------------------------------- */

#include "fix_neural_bias.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "text_file_reader.h"

#include <cstring>
#include <exception>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNeuralBias::FixNeuralBias(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), params(nullptr)
{
  if (narg != 4) error->all(FLERR, "Illegal fix neural/bias command");

  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  energy_global_flag = 1;

  energy = 0.0;
  energy_all = 0.0;
  force_flag = 0;
  nbias = 0;

  // read bias file on rank 0, broadcast
  if (comm->me == 0) {
    FILE *fp = utils::open_potential(arg[3], lmp, nullptr);
    if (!fp)
      error->one(FLERR, "Error opening neural bias file {}: {}",
                 arg[3], utils::getsyserror());
    TextFileReader reader(fp, "neural bias");

    std::vector<bias_param> myparams;
    char *line;

    try {
      while ((line = reader.next_line())) {
        ValueTokenizer values(line);
        bias_param oneparam;
        oneparam.id = values.next_tagint();
        oneparam.bias = values.next_double();
        myparams.push_back(oneparam);
        ++nbias;
      }
    } catch (std::exception &e) {
      error->one(FLERR, "Error reading neural bias file: {}\n{}", e.what(), line);
    }

    memory->create(params, nbias, "neural_bias:params");
    memcpy(params, myparams.data(), nbias * sizeof(bias_param));
    fclose(fp);

    utils::logmesg(lmp, "Read {} neural bias entries from {}\n", nbias, arg[3]);
  }

  MPI_Bcast(&nbias, 1, MPI_INT, 0, world);
  if (comm->me != 0)
    memory->create(params, nbias, "neural_bias:params");
  MPI_Bcast(params, nbias * sizeof(bias_param), MPI_BYTE, 0, world);
}

/* ---------------------------------------------------------------------- */

FixNeuralBias::~FixNeuralBias()
{
  memory->destroy(params);
}

/* ---------------------------------------------------------------------- */

int FixNeuralBias::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNeuralBias::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixNeuralBias::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixNeuralBias::post_force(int /*vflag*/)
{
  double **x = atom->x;
  double **f = atom->f;
  int nlocal = atom->nlocal;

  energy = 0.0;
  force_flag = 0;

  for (int n = 0; n < nbias; ++n) {
    int i = atom->map(params[n].id);
    if (i < 0 || i >= nlocal) continue;

    // bias force: F_i[0] = b_i
    f[i][0] += params[n].bias;

    // energy: E_i = -b_i * x_i[0]  (so that F = -dE/dx = b_i)
    energy += -params[n].bias * x[i][0];
  }
}

/* ---------------------------------------------------------------------- */

void FixNeuralBias::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

double FixNeuralBias::compute_scalar()
{
  if (force_flag == 0) {
    MPI_Allreduce(&energy, &energy_all, 1, MPI_DOUBLE, MPI_SUM, world);
    force_flag = 1;
  }
  return energy_all;
}
