/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   pair_style neural - bilinear coupling potential for neural network
   linear layers.

   Energy:  E_ij = -W_ij * x_i[0] * x_j[0]
   Force on i (x-component): F_i[0] += W_ij * x_j[0]
   Force on j (x-component): F_j[0] += W_ij * x_i[0]

   Only the x-coordinate of each particle carries the activation value.
   y,z coordinates are used for spatial separation only.

   Weight file format (one pair per line):
     atom_id1  atom_id2  weight_value

   Usage:
     pair_style neural weights.dat cutoff [nocheck]
     pair_coeff * *

   Contributing author: lammps-llm project
------------------------------------------------------------------------- */

#include "pair_neural.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "text_file_reader.h"

#include <cmath>
#include <cstring>
#include <exception>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairNeural::PairNeural(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  respa_enable = 0;
  cut_global = 0.0;
  params = nullptr;
  npairs = 0;
  check_flag = 1;
}

/* ---------------------------------------------------------------------- */

PairNeural::~PairNeural()
{
  memory->destroy(setflag);
  memory->destroy(cutsq);
  memory->destroy(params);
}

/* ----------------------------------------------------------------------
   Compute bilinear coupling forces.
   No neighbor list — loop through explicit pair list.
   Energy: E_ij = -W_ij * x_i * x_j  (x-coordinates only)
   Force on i: F_i[0] = +W_ij * x_j[0]  (gradient of -E w.r.t. x_i)
   Force on j: F_j[0] = +W_ij * x_i[0]  (gradient of -E w.r.t. x_j)
------------------------------------------------------------------------- */

void PairNeural::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const int nlocal = atom->nlocal;
  const int newton_pair = force->newton_pair;
  double **x = atom->x;
  double **f = atom->f;

  // get maximum valid tag for bounds checking
  bigint maxtag_one = 0, maxtag = 0;
  const tagint *tag = atom->tag;
  for (int i = 0; i < nlocal; ++i)
    maxtag_one = MAX(maxtag_one, tag[i]);
  MPI_Allreduce(&maxtag_one, &maxtag, 1, MPI_LMP_TAGINT, MPI_MAX, world);

  double epair;
  int i, j;
  int pc = 0;

  for (int n = 0; n < npairs; ++n) {
    const neural_param &par = params[n];

    // bounds check on atom IDs
    if ((par.id1 < 1) || (par.id1 > maxtag)) {
      if (check_flag)
        error->all(FLERR, "Invalid pair neural atom ID {}", par.id1);
      else
        continue;
    }
    if ((par.id2 < 1) || (par.id2 > maxtag)) {
      if (check_flag)
        error->all(FLERR, "Invalid pair neural atom ID {}", par.id2);
      else
        continue;
    }

    i = atom->map(par.id1);
    j = atom->map(par.id2);

    // skip if either atom not on this processor
    if ((i < 0) || (j < 0)) continue;

    // both atoms are ghosts -> skip
    if ((i >= nlocal) && (j >= nlocal)) continue;

    // newton pair ghost skipping (same logic as pair_list)
    if (newton_pair) {
      if ((i >= nlocal) && ((par.id1 + par.id2) & 1) == 0) continue;
      if ((j >= nlocal) && ((par.id1 + par.id2) & 1) == 1) continue;
    }

    if (check_flag) {
      if (newton_pair || i < nlocal) ++pc;
      if (newton_pair || j < nlocal) ++pc;
    }

    const double xi = x[i][0];  // activation value of atom i
    const double xj = x[j][0];  // activation value of atom j
    const double wij = par.weight;

    // bilinear coupling forces (x-component only)
    // F_i[0] = -dE/dx_i = -d(-W*xi*xj)/dxi = W*xj
    // F_j[0] = -dE/dx_j = -d(-W*xi*xj)/dxj = W*xi

    if (newton_pair || i < nlocal) {
      f[i][0] += wij * xj;
    }

    if (newton_pair || j < nlocal) {
      f[j][0] += wij * xi;
    }

    // energy: E_ij = -W_ij * x_i * x_j
    epair = 0.0;
    if (eflag_either) {
      epair = -wij * xi * xj;
    }

    if (evflag) {
      // For ev_tally we need dx, dy, dz and fpair such that
      // f[i] += dx*fpair, f[j] -= dx*fpair
      // But our forces are NOT proportional to displacement.
      // We pass zeros for dx,dy,dz and fpair=0, and handle
      // energy directly via the epair argument.
      ev_tally(i, j, nlocal, newton_pair, epair, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
  }

  if (check_flag) {
    int tmp;
    MPI_Allreduce(&pc, &tmp, 1, MPI_INT, MPI_SUM, world);
    if (tmp != 2 * npairs)
      error->all(FLERR, "Not all pairs processed in pair_style neural: {} vs {}",
                 tmp, 2 * npairs);
  }
}

/* ---------------------------------------------------------------------- */

void PairNeural::allocate()
{
  allocated = 1;
  int np1 = atom->ntypes + 1;

  memory->create(setflag, np1, np1, "pair:setflag");
  for (int i = 1; i < np1; i++)
    for (int j = i; j < np1; j++) setflag[i][j] = 0;

  memory->create(cutsq, np1, np1, "pair:cutsq");
}

/* ----------------------------------------------------------------------
   Read weight file and global cutoff.
   pair_style neural weights.dat cutoff [nocheck]
------------------------------------------------------------------------- */

void PairNeural::settings(int narg, char **arg)
{
  if (narg < 2) utils::missing_cmd_args(FLERR, "pair_style neural", error);

  cut_global = utils::numeric(FLERR, arg[1], false, lmp);

  int iarg = 2;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "nocheck") == 0) {
      check_flag = 0;
      ++iarg;
    } else if (strcmp(arg[iarg], "check") == 0) {
      check_flag = 1;
      ++iarg;
    } else
      error->all(FLERR, "Unknown pair_style neural keyword: {}", arg[iarg]);
  }

  // read weight file on MPI rank 0, broadcast to all
  if (comm->me == 0) {
    FILE *fp = utils::open_potential(arg[0], lmp, nullptr);
    if (!fp)
      error->one(FLERR, "Error opening neural weights file {}: {}",
                 arg[0], utils::getsyserror());
    TextFileReader reader(fp, "neural weights");
    npairs = 0;

    std::vector<neural_param> myparams;
    char *line;

    try {
      while ((line = reader.next_line())) {
        ValueTokenizer values(line);
        neural_param oneparam;
        oneparam.id1 = values.next_tagint();
        oneparam.id2 = values.next_tagint();
        oneparam.weight = values.next_double();
        myparams.push_back(oneparam);
        ++npairs;
      }
    } catch (std::exception &e) {
      error->one(FLERR, "Error reading neural weights file: {}\n{}", e.what(), line);
    }

    memory->create(params, npairs, "pair_neural:params");
    memcpy(params, myparams.data(), npairs * sizeof(neural_param));
    fclose(fp);

    utils::logmesg(lmp, "Read {} neural weight pairs from {}\n", npairs, arg[0]);
  }

  MPI_Bcast(&npairs, 1, MPI_INT, 0, world);
  if (comm->me != 0)
    memory->create(params, npairs, "pair_neural:params");
  MPI_Bcast(params, npairs * sizeof(neural_param), MPI_BYTE, 0, world);
}

/* ---------------------------------------------------------------------- */

void PairNeural::coeff(int narg, char **arg)
{
  if (narg < 2) utils::missing_cmd_args(FLERR, "pair_coeff neural", error);
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      setflag[i][j] = 1;
      count++;
    }
  }
  if (count == 0)
    error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ---------------------------------------------------------------------- */

void PairNeural::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style neural requires atom IDs");
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, "Pair style neural requires an atom map");
}

/* ---------------------------------------------------------------------- */

double PairNeural::init_one(int, int)
{
  return cut_global;
}

/* ---------------------------------------------------------------------- */

double PairNeural::memory_usage()
{
  double bytes = (double) npairs * sizeof(neural_param);
  const int n = atom->ntypes + 1;
  bytes += (double) n * (n * sizeof(int) + sizeof(int *));
  bytes += (double) n * (n * sizeof(double) + sizeof(double *));
  return bytes;
}
