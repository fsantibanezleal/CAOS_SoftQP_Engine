#include "polish.h"
#include "lin_alg.h"
#include "penalty.h"
#include "osqp_api_constants.h"
#include "printing.h"
#include "util.h"
#include "auxil.h"
#include "error.h"
#include "timing.h"

typedef struct {
  OSQPInt   flag;
  OSQPFloat kkt_diag;
  OSQPFloat kkt_exact;
  OSQPFloat rhs_bound;
  OSQPFloat y_fixed;
} OSQPPolishRow;

static OSQPPolishRow classify_polish_row(OSQPFloat z,
                                         OSQPFloat y,
                                         OSQPFloat l,
                                         OSQPFloat u,
                                         OSQPInt   type,
                                         OSQPFloat alpha1,
                                         OSQPFloat alpha2,
                                         OSQPFloat delta,
                                         OSQPFloat sigma) {
  OSQPPolishRow row = {0, 0.0, 0.0, 0.0, 0.0};
  OSQPFloat     s   = z - c_min(c_max(z, l), u);

  /* A group penalty couples the row to the rest of its group, so neither the
     active-set guess nor the (2,2) block can be decided row by row: the
     Euclidean norm has a curvature alpha*(I - ss'/|s|^2)/|s| that is not
     diagonal, and the subdifferential of the infinity norm depends on which
     rows of the group attain the maximum. Grouped rows are therefore left out
     of Ared with their dual frozen at the ADMM value, which keeps the polished
     system consistent while the remaining rows are refined. */
  if ((type == OSQP_PENALTY_NORM2) || (type == OSQP_PENALTY_NORMINF)) {
    row.y_fixed = y;
    return row;
  }

  if ((type != OSQP_PENALTY_NONE) && (s != 0.0)) {
    OSQPFloat bound = s > 0.0 ? u : l;
    OSQPFloat sign  = s > 0.0 ? 1.0 : -1.0;

    if ((type == OSQP_PENALTY_L1L2) && (alpha2 > 0.0)) {
      row.flag      = s > 0.0 ? +2 : -2;
      row.kkt_diag  = alpha2;
      row.kkt_exact = 1.0 / alpha2;
      row.rhs_bound = bound - sign * alpha1 / alpha2;
    }
    else if ((type == OSQP_PENALTY_HUBER) && (c_absval(s) <= delta)) {
      row.flag      = s > 0.0 ? +2 : -2;
      row.kkt_diag  = alpha1;
      row.kkt_exact = 1.0 / alpha1;
      row.rhs_bound = bound;
    }
    else if (type == OSQP_PENALTY_L1L2) {
      row.y_fixed = sign * alpha1;
    }
    else {
      row.y_fixed = sign * alpha1 * delta;
    }

    return row;
  }

  if ((z - l < -y) || (l == u)) {
    row.flag      = -1;
    row.kkt_diag  = 1.0 / sigma;
    row.rhs_bound = l;
  }
  else if (u - z < y) {
    row.flag      = +1;
    row.kkt_diag  = 1.0 / sigma;
    row.rhs_bound = u;
  }

  return row;
}

/**
 * Form reduced matrix A that contains only rows that are active at the
 * solution.
 * Ared = vstack[Alow, Aupp]
 * Active constraints are guessed from the primal and dual solution returned by
 * the ADMM.
 * @param  work Workspace
 * @return      Exitflag
 */
static OSQPInt form_Ared(OSQPSolver* solver){

  OSQPInt j, n_active;
  OSQPWorkspace* work = solver->work;
  OSQPInt m = work->data->m;

  OSQPPenaltyData* pen = work->data->penalty;

  OSQPInt* active_flags = OSQP_NULL;
  OSQPFloat* z = OSQP_NULL;
  OSQPFloat* y = OSQP_NULL;
  OSQPFloat* u = OSQP_NULL;
  OSQPFloat* l = OSQP_NULL;
  OSQPFloat* diag = OSQP_NULL;
  OSQPFloat* exct = OSQP_NULL;
  OSQPFloat* bnd  = OSQP_NULL;
  OSQPFloat* yfix = OSQP_NULL;
  OSQPInt*   type = OSQP_NULL;
  OSQPFloat* a1 = OSQP_NULL;
  OSQPFloat* a2 = OSQP_NULL;
  OSQPFloat* dl = OSQP_NULL;

  // Allocate raw arrays
  active_flags = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));
  z = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  y = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  l = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  u = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  diag = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  exct = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  bnd  = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  yfix = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  if (pen) {
    if (!pen->uniform) type = (OSQPInt *)c_malloc(m * sizeof(OSQPInt));
    a1 = (OSQPFloat *)c_malloc(m * sizeof(OSQPFloat));
    a2 = (OSQPFloat *)c_malloc(m * sizeof(OSQPFloat));
    dl = (OSQPFloat *)c_malloc(m * sizeof(OSQPFloat));
  }

  /* Handle memory allocation errors */
  if (!active_flags || !z || !y || !l || !u ||
      !diag || !exct || !bnd || !yfix ||
      (pen && ((!pen->uniform && !type) || !a1 || !a2 || !dl))) {
    c_free(active_flags); c_free(z); c_free(y); c_free(l); c_free(u);
    c_free(diag); c_free(exct); c_free(bnd); c_free(yfix);
    c_free(type); c_free(a1); c_free(a2); c_free(dl);

    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }

  // Copy data to raw arrays
  OSQPVectorf_to_raw(z, work->z);
  OSQPVectorf_to_raw(y, work->y);
  OSQPVectorf_to_raw(l, work->data->l);
  OSQPVectorf_to_raw(u, work->data->u);

  if (pen) {
    if (!pen->uniform) OSQPVectori_to_raw(type, pen->type);
    OSQPVectorf_to_raw(a1, pen->alpha1);
    OSQPVectorf_to_raw(a2, pen->alpha2);
    OSQPVectorf_to_raw(dl, pen->delta);
  }

  // Initialize counters for active constraints
  n_active = 0;

  /* Guess which rows enter the reduced KKT system, and how.
   *
   *    active_flags is -1/0/1 to indicate  lower/ inactive / upper.
   *    equality constraints are treated as lower active
   *
   *    A soft row sitting on a smooth piece of its penalty is -2/+2: it also
   *    enters Ared, but with a finite diagonal 1/kappa in the (2,2) block
   *    rather than the regularizer, since z moves with y there.  A soft row on
   *    a linear piece leaves Ared with its dual pinned in y_fixed, as does
   *    every row of a penalty group, which is not polished at all.
   *
   *    Ared is formed by selecting all of the rows flagged nonzero.
   */

  for (j = 0; j < work->data->m; j++) {

    OSQPInt row_type = pen ? (pen->uniform ? pen->default_penalty_type : type[j])
                           : OSQP_PENALTY_NONE;
    OSQPPolishRow row = classify_polish_row(z[j], y[j], l[j], u[j], row_type,
                                            pen ? a1[j] : 0.0,
                                            pen ? a2[j] : 0.0,
                                            pen ? dl[j] : 0.0,
                                            solver->settings->sigma);

    active_flags[j] = row.flag;
    bnd[j]          = row.rhs_bound;
    yfix[j]         = row.y_fixed;

    if (row.flag != 0) {
      diag[n_active] = row.kkt_diag;
      exct[n_active] = row.kkt_exact;
      n_active++;
    }
  }

  // Copy raw vectors into OSQPVector structures
  OSQPVectori_from_raw(work->pol->active_flags, active_flags);
  OSQPVectorf_from_raw(work->pol->kkt_diag, diag);
  OSQPVectorf_from_raw(work->pol->kkt_exact, exct);
  OSQPVectorf_from_raw(work->pol->rhs_bound, bnd);
  OSQPVectorf_from_raw(work->pol->y_fixed, yfix);

  //total active constraints
  work->pol->n_active = n_active;

  //extract the relevant rows
  work->pol->Ared = OSQPMatrix_submatrix_byrows(work->data->A, work->pol->active_flags);

  // Memory clean-up
  c_free(active_flags); c_free(z); c_free(y); c_free(l); c_free(u);
  c_free(diag); c_free(exct); c_free(bnd); c_free(yfix);
  c_free(type); c_free(a1); c_free(a2); c_free(dl);

  if (!work->pol->Ared)
    return osqp_error(OSQP_MEM_ALLOC_ERROR);;

  return OSQP_NO_ERROR;
}

/**
 * Form reduced right-hand side rhs_red = vstack[-q, l_low, u_upp]
 * @param  work Workspace
 * @param  rhs  right-hand-side
 * @return      Exitflag
 */
static OSQPInt form_rhs_red(OSQPWorkspace* work, OSQPVectorf* rhs) {

  OSQPInt j, counter;
  OSQPInt n = work->data->n;
  OSQPInt m = work->data->m;
  OSQPInt n_plus_mred = OSQPVectorf_length(rhs);

  OSQPInt *active_flags = OSQP_NULL;
  OSQPFloat* rhsv = OSQP_NULL;
  OSQPFloat* q = OSQP_NULL;
  OSQPFloat* bnd = OSQP_NULL;

  // Allocate raw arrays
  active_flags = (OSQPInt *)   c_malloc(m           * sizeof(OSQPInt));
  rhsv         = (OSQPFloat *) c_malloc(n_plus_mred * sizeof(OSQPFloat));
  q            = (OSQPFloat *) c_malloc(n           * sizeof(OSQPFloat));
  bnd          = (OSQPFloat *) c_malloc(m           * sizeof(OSQPFloat));

  if (!active_flags || !rhsv || !q || !bnd) {
    c_free(active_flags);
    c_free(rhsv);
    c_free(q);
    c_free(bnd);

    return osqp_error(OSQP_MEM_ALLOC_ERROR);;
  }

  // Copy data to raw arrays
  OSQPVectori_to_raw(active_flags, work->pol->active_flags);
  OSQPVectorf_to_raw(rhsv, rhs);
  OSQPVectorf_to_raw(q, work->data->q);
  OSQPVectorf_to_raw(bnd, work->pol->rhs_bound);

  for(j = 0; j < work->data->n; j++){
    rhsv[j] = -q[j];
  }

  /* NB: form_Ared already worked out what each selected row contributes: the
     bound for an active row, and the bound offset by the constant part of the
     subdifferential for a soft row on a smooth piece of its penalty. */
  counter = 0;

  for (j = 0; j < work->data->m; j++) {
    if (active_flags[j] != 0) {
       rhsv[work->data->n + counter] = bnd[j];
       counter++;
    }
  }

  // Copy raw vector into OSQPVectorf structure
  OSQPVectorf_from_raw(rhs, rhsv);

  // Memory clean-up
  c_free(active_flags);
  c_free(rhsv);
  c_free(q);
  c_free(bnd);

  return OSQP_NO_ERROR;
}

/**
 * Perform iterative refinement on the polished solution:
 *    (repeat)
 *    1. (K + dK) * dz = b - K*z
 *    2. z <- z + dz
 * @param  solver Solver instance
 * @param  p    Private variable for solving linear system
 * @param  z    Initial z value
 * @param  b    RHS of the linear system
 * @return      Exitflag
 */
static OSQPInt iterative_refinement(OSQPSolver*   solver,
                                    LinSysSolver* p,
                                    OSQPVectorf*  z,
                                    OSQPVectorf*  b) {
  OSQPInt i, mred;
  OSQPVectorf *rhs, *rhs1, *rhs2;
  OSQPVectorf *z1, *z2;
  OSQPVectorf *kkt_exact = OSQP_NULL, *tmp = OSQP_NULL;

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  if (settings->polish_refine_iter > 0) {
    mred = OSQPMatrix_get_m(work->pol->Ared);

    // Allocate dz and rhs vectors
    rhs = OSQPVectorf_malloc(work->data->n + mred);

    //form views of the top/bottom parts of rhs and z
    rhs1 = OSQPVectorf_view(rhs,0,work->data->n);
    rhs2 = OSQPVectorf_view(rhs,work->data->n,mred);
    z1   = OSQPVectorf_view(z,0,work->data->n);
    z2   = OSQPVectorf_view(z,work->data->n,mred);

    /* Where a soft row's slack moves with its dual, the (2,2) block is part of
       the exact system rather than regularization, so it must be carried here
       too; otherwise refinement drives the row back to an equality at its
       bound and undoes the polish. */
    if (work->data->penalty) {
      kkt_exact = OSQPVectorf_view(work->pol->kkt_exact, 0, mred);
      tmp       = OSQPVectorf_malloc(mred);
    }

    if (!rhs || !rhs1 || !rhs2 || !z1 || !z2 ||
        (work->data->penalty && (!kkt_exact || !tmp))) {
      return osqp_error(OSQP_MEM_ALLOC_ERROR);
    }

    for (i = 0; i < settings->polish_refine_iter; i++) {

      // Form the RHS for the iterative refinement:  b - K*z
      OSQPVectorf_copy(rhs,b);

      // Upper Part: R^{n}
      // -= Px  (in the top partition)
      OSQPMatrix_Axpy(work->data->P, z1, rhs1, -1.0, 1.0);

      // -= Ared'*y_red  (in the top partition)
      OSQPMatrix_Atxpy(work->pol->Ared, z2, rhs1, -1.0, 1.0);

      // Lower Part: R^{m}
      // -= A*x  (in the bottom partition)
      OSQPMatrix_Axpy(work->pol->Ared, z1, rhs2, -1.0, 1.0);

      // += y_red / kappa  (on the rows that carry an exact diagonal)
      if (kkt_exact) {
        OSQPVectorf_ew_prod(tmp, kkt_exact, z2);
        OSQPVectorf_plus(rhs2, rhs2, tmp);
      }

      // Solve linear system. Store solution in rhs
      p->solve(p, rhs, 1);

      // Update solution
      OSQPVectorf_plus(z,z,rhs);
    }

    OSQPVectorf_free(rhs);
    OSQPVectorf_view_free(rhs1);
    OSQPVectorf_view_free(rhs2);
    OSQPVectorf_view_free(z1);
    OSQPVectorf_view_free(z2);
    OSQPVectorf_view_free(kkt_exact);
    OSQPVectorf_free(tmp);
  }
  return 0;
}

/**
 * Compute dual variable y from yred
 * @param work Workspace
 * @param yred_vf Dual variables associated to active constraints
 * @return Exitflag
 */
static OSQPInt get_ypol_from_yred(OSQPWorkspace* work, OSQPVectorf* yred_vf) {

  OSQPInt j, counter;
  OSQPInt m = work->data->m;
  OSQPInt mred = OSQPVectorf_length(yred_vf);

  OSQPInt *active_flags = OSQP_NULL;
  OSQPFloat* y = OSQP_NULL;
  OSQPFloat* yred = OSQP_NULL;

  // Allocate raw arrays
  active_flags = (OSQPInt *)   c_malloc(m    * sizeof(OSQPInt));
  y            = (OSQPFloat *) c_malloc(m    * sizeof(OSQPFloat));
  yred         = (OSQPFloat *) c_malloc(mred * sizeof(OSQPFloat));

  if (!active_flags || !y || !yred) {
    // Memory clean-up
    c_free(active_flags);
    c_free(y);
    c_free(yred);

    return osqp_error(OSQP_MEM_ALLOC_ERROR);;
  }

  // Copy data to raw arrays
  OSQPVectori_to_raw(active_flags, work->pol->active_flags);
  OSQPVectorf_to_raw(y, work->pol->y_fixed);
  OSQPVectorf_to_raw(yred, yred_vf);

  // If there are no active constraints
  if (work->pol->n_active == 0) {
    OSQPVectorf_copy(work->pol->y, work->pol->y_fixed);

    // Memory clean-up
    c_free(active_flags);
    c_free(y);
    c_free(yred);

    return OSQP_NO_ERROR;
  }

  counter = 0;

  /* NB: y already holds y_fixed, which is zero except on a soft row sitting on
     a linear piece of its penalty, where the dual is pinned and z is free. */
  for (j = 0; j < work->data->m; j++) {

    if (active_flags[j] != 0) {
      y[j] = yred[counter];
      counter++;
    }
  }

  // Copy raw vector into OSQPVectorf structure
  OSQPVectorf_from_raw(work->pol->y, y);

  // Memory clean-up
  c_free(active_flags);
  c_free(y);
  c_free(yred);

  return OSQP_NO_ERROR;
}

/* The complementarity fix-up of polish(), applied to hard rows only */
static OSQPInt bound_hard_rows(OSQPWorkspace* work) {

  OSQPInt j;
  OSQPInt m = work->data->m;

  OSQPPenaltyData* pen = work->data->penalty;

  OSQPFloat* z = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  OSQPFloat* y = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  OSQPFloat* l = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  OSQPFloat* u = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  OSQPInt* type = pen->uniform ? OSQP_NULL
                               : (OSQPInt *)c_malloc(m * sizeof(OSQPInt));

  if (!z || !y || !l || !u || (!pen->uniform && !type)) {
    c_free(z); c_free(y); c_free(l); c_free(u); c_free(type);
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }

  OSQPVectorf_to_raw(z, work->pol->z);
  OSQPVectorf_to_raw(y, work->pol->y);
  OSQPVectorf_to_raw(l, work->data->l);
  OSQPVectorf_to_raw(u, work->data->u);
  if (!pen->uniform) OSQPVectori_to_raw(type, pen->type);

  for (j = 0; j < m; j++) {
    OSQPFloat t;

    if ((pen->uniform ? pen->default_penalty_type : type[j]) != OSQP_PENALTY_NONE)
      continue;

    t    = y[j] + z[j];
    z[j] = c_min(c_max(t, l[j]), u[j]);
    y[j] = t - z[j];
  }

  OSQPVectorf_from_raw(work->pol->z, z);
  OSQPVectorf_from_raw(work->pol->y, y);

  c_free(z); c_free(y); c_free(l); c_free(u); c_free(type);

  return OSQP_NO_ERROR;
}

OSQPInt polish(OSQPSolver* solver) {

  OSQPInt polish_successful = 0;
  OSQPInt exitflag = 0;

  LinSysSolver* plsh = OSQP_NULL;
  OSQPVectorf*  kkt_diag = OSQP_NULL;
  OSQPVectorf*  rhs_red = OSQP_NULL;
  OSQPVectorf*  pol_sol = OSQP_NULL; // Polished solution (x and reduced y)
  OSQPVectorf*  pol_sol_xview = OSQP_NULL; // view into x part of polished solution
  OSQPVectorf*  pol_sol_yview = OSQP_NULL; // view into (reduced) y part of polished solutions

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

#ifdef OSQP_ENABLE_PROFILING
  osqp_tic(work->timer); // Start timer
#endif /* ifdef OSQP_ENABLE_PROFILING */

  // Form Ared by assuming the active constraints and store in work->pol->Ared
  exitflag = form_Ared(solver);

  if (exitflag) {
    /* Failure finding active constraints */
    info->status_polish = OSQP_POLISH_FAILED;
    return exitflag;
  } else if (work->pol->n_active == 0 &&
             (!work->data->penalty ||
              OSQPVectorf_norm_inf(work->pol->y_fixed) == 0.0)) {
    /* No active constraints and no pinned duals, so skip polishing.
       NB: pinned duals still determine x through P*x + q + A'*y = 0. */
    c_print("Polishing not needed - no active set detected at optimal point\n");
    info->status_polish = OSQP_POLISH_NO_ACTIVE_SET_FOUND;

    /* Memory clean-up */
    OSQPMatrix_free(work->pol->Ared);
    OSQPVectorf_view_free(kkt_diag);

    return OSQP_NO_ERROR;
  }

  /* Form and factorize reduced KKT. The (2,2) block is -1/kkt_diag per row:
     the regularizer on an active row, and 1/kappa_i where a soft row's slack
     moves with its dual. */
  kkt_diag = OSQPVectorf_view(work->pol->kkt_diag, 0, work->pol->n_active);

  if (!kkt_diag) {
    info->status_polish = OSQP_POLISH_FAILED;
    OSQPMatrix_free(work->pol->Ared);
    OSQPVectorf_view_free(kkt_diag);
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }

  exitflag = osqp_algebra_init_linsys_solver(&plsh, work->data->P, work->pol->Ared,
                                             kkt_diag, settings, OSQP_NULL, OSQP_NULL, 1);

  if (exitflag) {
    /* Failure to initialize the linear system */
    info->status_polish = OSQP_POLISH_LINSYS_ERROR;

    /* Memory clean-up */
    OSQPMatrix_free(work->pol->Ared);
    OSQPVectorf_view_free(kkt_diag);

    return exitflag;
  }

  // Form reduced right-hand side rhs_red
  rhs_red = OSQPVectorf_malloc(work->data->n + work->pol->n_active);

  if (!rhs_red) {
    /* Failure to allocate memory */
    info->status_polish = OSQP_POLISH_FAILED;

    /* Memory clean-up */
    OSQPMatrix_free(work->pol->Ared);
    OSQPVectorf_view_free(kkt_diag);

    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }

  exitflag = form_rhs_red(work, rhs_red);

  /* Rows left out of Ared with a pinned dual still act on x */
  if (!exitflag && work->data->penalty) {
    OSQPVectorf* rhs_x = OSQPVectorf_view(rhs_red, 0, work->data->n);

    if (rhs_x) {
      OSQPMatrix_Atxpy(work->data->A, work->pol->y_fixed, rhs_x, -1.0, 1.0);
      OSQPVectorf_view_free(rhs_x);
    }
    else exitflag = osqp_error(OSQP_MEM_ALLOC_ERROR);
  }

  if (exitflag) {
    /* Failure to form reduced right hand side */
    info->status_polish = OSQP_POLISH_FAILED;

    /* Memory clean-up */
    OSQPMatrix_free(work->pol->Ared);
    OSQPVectorf_view_free(kkt_diag);

    return exitflag;
  }

  pol_sol = OSQPVectorf_copy_new(rhs_red);

  if (!pol_sol) {
    /* Failure to allocate vector */
    info->status_polish = OSQP_POLISH_FAILED;

    /* Memory clean-up */
    OSQPMatrix_free(work->pol->Ared);
    OSQPVectorf_view_free(kkt_diag);
    OSQPVectorf_free(rhs_red);

    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }

  pol_sol_xview = OSQPVectorf_view(pol_sol,0,work->data->n);
  pol_sol_yview = OSQPVectorf_view(pol_sol,work->data->n, work->pol->n_active);

  if (!pol_sol_xview || !pol_sol_yview) {

    // Polishing failed
    info->status_polish = OSQP_POLISH_FAILED;

    // Memory clean-up
    OSQPMatrix_free(work->pol->Ared);
    OSQPVectorf_view_free(kkt_diag);
    OSQPVectorf_free(rhs_red);
    OSQPVectorf_free(pol_sol);
    OSQPVectorf_view_free(pol_sol_xview);
    OSQPVectorf_view_free(pol_sol_yview);

    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }

  // Warm start the polished solution
  plsh->warm_start(plsh, work->x);

  // Solve the reduced KKT system
  plsh->solve(plsh, pol_sol, 1);

  // Perform iterative refinement to compensate for the regularization error
  exitflag = iterative_refinement(solver, plsh, pol_sol, rhs_red);

  if (exitflag) {
    // Polishing failed
    info->status_polish = OSQP_POLISH_FAILED;

    // Memory clean-up
    OSQPMatrix_free(work->pol->Ared);
    OSQPVectorf_view_free(kkt_diag);
    OSQPVectorf_free(rhs_red);
    OSQPVectorf_free(pol_sol);
    OSQPVectorf_view_free(pol_sol_xview);
    OSQPVectorf_view_free(pol_sol_yview);

    return exitflag;
  }

  // Store the polished solution (x,z,y)
  OSQPVectorf_copy(work->pol->x, pol_sol_xview);   // pol->x
  OSQPMatrix_Axpy(work->data->A, work->pol->x, work->pol->z, 1.0, 0.0);
  get_ypol_from_yred(work, pol_sol_yview);     // pol->y

  /* Ensure z is in C and y is in the normal cone N_C(z)
     by doing: y <- y + z;  z <- proj_C(y);  y <- y - z.
     NB: only on hard rows. A soft row's z is meant to sit outside [l,u] -- that
     is its slack, not a primal residual -- and its (z,y) pair already satisfies
     the optimality conditions by construction of the reduced KKT. */
  if (work->data->penalty) exitflag = bound_hard_rows(work);
  else {
    OSQPVectorf_plus(work->pol->y, work->pol->y, work->pol->z);
    OSQPVectorf_ew_bound_vec(work->pol->z, work->pol->y, work->data->l, work->data->u);
    OSQPVectorf_minus(work->pol->y, work->pol->y, work->pol->z);
  }

  if (exitflag) {
    info->status_polish = OSQP_POLISH_FAILED;

    OSQPMatrix_free(work->pol->Ared);
    OSQPVectorf_view_free(kkt_diag);
    OSQPVectorf_free(rhs_red);
    OSQPVectorf_free(pol_sol);
    OSQPVectorf_view_free(pol_sol_xview);
    OSQPVectorf_view_free(pol_sol_yview);
    plsh->free(plsh);

    return exitflag;
  }

  // Compute primal and dual residuals at the polished solution
  update_info(solver, 0, 1);

  // Check if polish was successful
  polish_successful = (work->pol->prim_res < info->prim_res &&
                       work->pol->dual_res < info->dual_res) || // Residuals
                                                                    // are
                                                                    // reduced
                      (work->pol->prim_res < info->prim_res &&
                       info->dual_res < 1e-10) ||              // Dual
                                                                    // residual
                                                                    // already
                                                                    // tiny
                      (work->pol->dual_res < info->dual_res &&
                       info->prim_res < 1e-10);                // Primal
                                                                    // residual
                                                                    // already
                                                                    // tiny

  if (polish_successful) {
    // Update solver information
    info->obj_val       = work->pol->obj_val;
    info->penalty_val   = work->pol->penalty_val;
    info->dual_obj_val  = work->pol->dual_obj_val;
    info->duality_gap   = work->pol->duality_gap;
    info->prim_res      = work->pol->prim_res;
    info->dual_res      = work->pol->dual_res;
    info->status_polish = OSQP_POLISH_SUCCESS;

    // Update (x, z, y) in ADMM iterations
    // NB: z needed for warm starting
    OSQPVectorf_copy(work->x, work->pol->x);
    OSQPVectorf_copy(work->z, work->pol->z);
    OSQPVectorf_copy(work->y, work->pol->y);

    // Print summary
#ifdef OSQP_ENABLE_PRINTING

    if (settings->verbose) print_polish(solver);
#endif /* ifdef OSQP_ENABLE_PRINTING */
  } else { // Polishing failed
    info->status_polish = OSQP_POLISH_FAILED;

    // TODO: Try to find a better solution on the line connecting ADMM
    //       and polished solution
  }

  // Memory clean-up
  plsh->free(plsh);

  // Checks that they are not NULL are already performed earlier
  OSQPMatrix_free(work->pol->Ared);
  OSQPVectorf_view_free(kkt_diag);
  OSQPVectorf_free(rhs_red);
  OSQPVectorf_free(pol_sol);
  OSQPVectorf_view_free(pol_sol_xview);
  OSQPVectorf_view_free(pol_sol_yview);

  return OSQP_NO_ERROR;
}
