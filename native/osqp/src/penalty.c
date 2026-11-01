/* Modified for SoftQP, 2026-10-31: native asymmetric penalty support and bindings.
 * Original upstream copyright and Apache-2.0 license are retained. */
#include <math.h>
#include "osqp.h"
#include "penalty.h"
#include "algebra_vector.h"
#include "auxil.h"
#include "error.h"
#include "printing.h"
#include "scaling.h"
#include "algebra_impl.h" /* SoftQP CPU builtin extension */

/***********************************************************
* Soft-constraint penalties                              * *
***********************************************************/

/* The per-row types, or OSQP_NULL when every row uses default_penalty_type.
 * The element-wise kernels take the fast path on OSQP_NULL. */
static const OSQPVectori* penalty_types(const OSQPPenaltyData* pen) {
  return pen->uniform ? OSQP_NULL : pen->type;
}

/* Weights are required to be finite, so the declared type is the whole story:
 * a row is soft if and only if its type is not OSQP_PENALTY_NONE. */
const OSQPVectori* penalty_row_types(const OSQPSolver* solver) {
  OSQPPenaltyData* pen = solver->work->data->penalty;

  return pen ? penalty_types(pen) : OSQP_NULL;
}

OSQPInt penalty_default_type(const OSQPSolver* solver) {
  OSQPPenaltyData* pen = solver->work->data->penalty;

  return pen ? pen->default_penalty_type : OSQP_PENALTY_NONE;
}

void penalty_project(OSQPSolver*        solver,
                     OSQPVectorf*       z,
                     const OSQPVectorf* v) {

  OSQPWorkspace*   work = solver->work;
  OSQPPenaltyData* pen  = work->data->penalty;

  if (!pen) {
    OSQPVectorf_ew_bound_vec(z, v, work->data->l, work->data->u);
    return;
  }

  if (pen->upper1) {
    OSQPInt i;
    for (i=0; i<v->length; ++i) {
      OSQPFloat vi=v->values[i], p=c_min(c_max(vi,work->data->l->values[i]),work->data->u->values[i]);
      OSQPFloat residual=vi-p;
      OSQPInt type=pen->uniform ? pen->default_penalty_type : pen->type->values[i];
      OSQPFloat rho=solver->settings->rho_is_vec ? work->rho_vec->values[i] : solver->settings->rho;
      OSQPFloat a=residual>0 ? pen->upper1->values[i] : pen->alpha1->values[i];
      OSQPFloat b=residual>0 ? pen->upper2->values[i] : pen->alpha2->values[i];
      OSQPFloat t=c_max(c_absval(residual)-a/rho,0.0)*(rho/(rho+b));
      z->values[i]=p+(type==OSQP_PENALTY_NONE ? 0.0 : (residual>0 ? t : -t));
    }
    return;
  }

  /* The group pass goes first and the elementwise pass then skips its rows, so
     that z may alias v: each pass needs v intact on the other's rows. The
     scalar rho is the right one for a group, set_rho_vec giving every grouped
     row rho_ineq. */
  if (pen->ngroups)
    OSQPVectorf_group_prox_penalty(z, v, work->data->l, work->data->u,
                                   solver->settings->rho,
                                   pen->alpha1,
                                   penalty_types(pen), pen->default_penalty_type,
                                   pen->group_ptr, pen->group_rows, pen->ngroups,
                                   work->penalty_sort_tmp);

  OSQPVectorf_ew_prox_penalty(z, v, work->data->l, work->data->u,
                              solver->settings->rho_is_vec ? work->rho_vec : OSQP_NULL,
                              solver->settings->rho,
                              pen->alpha1, pen->alpha2, pen->delta,
                              penalty_types(pen), pen->default_penalty_type);
}

/* Phi(R(z)), the penalty contribution to the primal objective. Scaled like the
 * rest of the objective, since the scaled penalty is c*phi(s/E). */
OSQPFloat penalty_obj_value(const OSQPSolver*  solver,
                            const OSQPVectorf* z) {

  OSQPWorkspace*   work = solver->work;
  OSQPPenaltyData* pen  = work->data->penalty;

  OSQPFloat val;

  if (!pen) return 0.0;

  if (pen->upper1) {
    OSQPInt i; val=0.0;
    for (i=0;i<z->length;++i) {
      OSQPFloat residual=z->values[i]-c_min(c_max(z->values[i],work->data->l->values[i]),work->data->u->values[i]);
      OSQPInt type=pen->uniform ? pen->default_penalty_type : pen->type->values[i];
      OSQPFloat a=residual>0 ? pen->upper1->values[i] : pen->alpha1->values[i];
      OSQPFloat b=residual>0 ? pen->upper2->values[i] : pen->alpha2->values[i];
      if(type!=OSQP_PENALTY_NONE) val+=a*c_absval(residual)+0.5*b*residual*residual;
    }
    return val;
  }
  val = OSQPVectorf_penalty_value(z, work->data->l, work->data->u,
                                  pen->alpha1, pen->alpha2, pen->delta,
                                  penalty_types(pen), pen->default_penalty_type,
                                  work->penalty_val_tmp);

  if (pen->ngroups)
    val += OSQPVectorf_group_penalty_value(z, work->data->l, work->data->u,
                                           pen->alpha1,
                                           penalty_types(pen), pen->default_penalty_type,
                                           pen->group_ptr, pen->group_rows, pen->ngroups,
                                           work->penalty_val_tmp);

  return val;
}

/* Phi*(y), the penalty contribution to the dual objective. OSQP_INFTY if y is
 * outside dom Phi*, which ADMM iterates never are but polish and warm starts
 * can be. */
OSQPFloat penalty_conj_value(const OSQPSolver*  solver,
                             const OSQPVectorf* y) {

  OSQPWorkspace*   work = solver->work;
  OSQPPenaltyData* pen  = work->data->penalty;

  OSQPFloat val;

  if (!pen) return 0.0;

  if (pen->upper1) {
    OSQPInt i; val=0.0;
    for (i=0;i<y->length;++i) {
      OSQPFloat yi=y->values[i], ay=c_absval(yi);
      OSQPInt type=pen->uniform ? pen->default_penalty_type : pen->type->values[i];
      OSQPFloat a=yi>0 ? pen->upper1->values[i] : pen->alpha1->values[i];
      OSQPFloat b=yi>0 ? pen->upper2->values[i] : pen->alpha2->values[i];
      OSQPFloat t=c_max(ay-a,0.0);
      if(type==OSQP_PENALTY_NONE) continue;
      if(b<=0) { if(ay>a+OSQP_PENALTY_CONJ_TOL*(1+a)) return OSQP_INFTY; }
      else val+=(0.5*t)*(t/b);
    }
    return val;
  }
  val = OSQPVectorf_penalty_conj_value(y, pen->alpha1, pen->alpha2, pen->delta,
                                       penalty_types(pen), pen->default_penalty_type,
                                       work->penalty_val_tmp);

  if (val >= OSQP_INFTY) return OSQP_INFTY;

  if (pen->ngroups)
    val += OSQPVectorf_group_penalty_conj_value(y, pen->alpha1,
                                                penalty_types(pen),
                                                pen->default_penalty_type,
                                                pen->group_ptr, pen->group_rows,
                                                pen->ngroups,
                                                work->penalty_val_tmp);

  return val;
}

#if OSQP_EMBEDDED_MODE != 1

void penalty_equalize_scaling(OSQPSolver* solver) {

  OSQPWorkspace*   work = solver->work;
  OSQPPenaltyData* pen;

  if (!work || !work->data) return;

  pen = work->data->penalty;

  if (!pen || !pen->ngroups || !solver->settings->scaling) return;

  /* E_temp is the Ruiz iteration's own scratch and is dead by the time
     scale_data calls this */
  OSQPVectorf_group_equalize_scaling(work->E_temp, work->scaling->E,
                                     pen->group_ptr, pen->group_rows,
                                     pen->ngroups);

  /* A must follow E, so that EA keeps meaning the scaled constraint matrix */
  OSQPMatrix_lmult_diag(work->data->A, work->E_temp);
  OSQPVectorf_ew_prod(work->scaling->E, work->scaling->E, work->E_temp);
}

/* Scale or unscale the penalty parameters in place, like l and u */
static void penalty_apply_scaling(OSQPSolver* solver,
                                  OSQPInt     invert) {

  OSQPWorkspace*   work = solver->work;
  OSQPPenaltyData* pen;

  if (!work || !work->data) return;

  pen = work->data->penalty;

  if (!pen || !solver->settings->scaling) return;

  OSQPVectorf_ew_scale_penalty(pen->alpha1, pen->alpha2, pen->delta,
                               penalty_types(pen), pen->default_penalty_type,
                               work->scaling->c, work->scaling->E, invert);
  if (pen->upper1)
    OSQPVectorf_ew_scale_penalty(pen->upper1, pen->upper2, pen->delta,
                                  penalty_types(pen), pen->default_penalty_type,
                                  work->scaling->c, work->scaling->E, invert);
}

void penalty_scale(OSQPSolver* solver) {
  penalty_apply_scaling(solver, 0);
}

void penalty_unscale(OSQPSolver* solver) {
  penalty_apply_scaling(solver, 1);
}

#endif /* if OSQP_EMBEDDED_MODE != 1 */

#ifndef OSQP_EMBEDDED_MODE

void penalty_free(OSQPWorkspace* work) {

  OSQPPenaltyData* pen;

  if (!work) return;

  OSQPVectori_free(work->penalty_type_tmp);
  work->penalty_type_tmp = OSQP_NULL;
  OSQPVectori_free(work->penalty_flags_tmp);
  work->penalty_flags_tmp = OSQP_NULL;
  OSQPVectorf_free(work->penalty_val_tmp);
  work->penalty_val_tmp = OSQP_NULL;
  OSQPVectorf_free(work->penalty_sort_tmp);
  work->penalty_sort_tmp = OSQP_NULL;

  if (!work->data) return;

  pen = work->data->penalty;

  if (pen) {
    OSQPVectori_free(pen->type);
    OSQPVectorf_free(pen->alpha1);
    OSQPVectorf_free(pen->alpha2);
    OSQPVectorf_free(pen->upper1);
    OSQPVectorf_free(pen->upper2);
    OSQPVectorf_free(pen->delta);
    OSQPVectori_free(pen->group_ptr);
    OSQPVectori_free(pen->group_rows);
    c_free(pen);
  }

  work->data->penalty = OSQP_NULL;
}

/* Allocate the penalty data. The weights are left uninitialized: every entry
 * point that creates or changes types also establishes valid weights.
 * The group vectors are allocated only when there is a group to hold. */
static OSQPInt penalty_alloc(OSQPWorkspace* work,
                             OSQPInt        m,
                             OSQPInt        ngroups,
                             OSQPInt        ngrouped) {

  OSQPPenaltyData* pen = c_calloc(1, sizeof(OSQPPenaltyData));

  if (!pen) return 1;

  pen->uniform              = 1;
  pen->default_penalty_type = OSQP_PENALTY_NONE;
  pen->ngroups              = ngroups;
  pen->ngrouped             = ngrouped;

  /* Calloc so the types are a defined OSQP_PENALTY_NONE even while uniform */
  pen->type   = OSQPVectori_calloc(m);
  pen->alpha1 = OSQPVectorf_malloc(m);
  pen->alpha2 = OSQPVectorf_malloc(m);
  pen->delta  = OSQPVectorf_malloc(m);

  work->data->penalty    = pen;
  work->penalty_type_tmp = OSQPVectori_calloc(m);
  work->penalty_flags_tmp = OSQPVectori_calloc(1);
  work->penalty_val_tmp   = OSQPVectorf_malloc(1);

  if (!(pen->type) || !(pen->alpha1) || !(pen->alpha2) || !(pen->delta) ||
      !(work->penalty_type_tmp) || !(work->penalty_flags_tmp) ||
      !(work->penalty_val_tmp)) {
    penalty_free(work);
    return 1;
  }

  if (ngroups) {
    pen->group_ptr  = OSQPVectori_calloc(ngroups + 1);
    pen->group_rows = OSQPVectori_calloc(ngrouped);

    /* Two halves: the staged residuals, then a sortable copy of their
       magnitudes, so the l_inf projection can sort without losing signs */
    work->penalty_sort_tmp = OSQPVectorf_malloc(2 * ngrouped);

    if (!(pen->group_ptr) || !(pen->group_rows) || !(work->penalty_sort_tmp)) {
      penalty_free(work);
      return 1;
    }
  }

  return 0;
}

/* NB: a switch rather than a range check, so adding a type lands here */
static OSQPInt penalty_type_is_valid(OSQPInt type) {

  switch (type) {
    case OSQP_PENALTY_NONE:
    case OSQP_PENALTY_L1L2:
    case OSQP_PENALTY_HUBER:
    case OSQP_PENALTY_NORM2:
    case OSQP_PENALTY_NORMINF:
      return 1;

    default:
      return 0;
  }
}

/* Membership in CSR form, built by counting sort so that a row lands in at
 * most one group by construction and the rows of a group stay ascending.
 * Returns the number of grouped rows, or -1 if group_id is not a valid
 * partition into exactly ngroups non-empty groups. */
static OSQPInt penalty_count_groups(const OSQPInt* group_id,
                                    OSQPInt        m,
                                    OSQPInt        ngroups,
                                    OSQPInt*       counts) {

  OSQPInt i, g;
  OSQPInt ngrouped = 0;

  for (g = 0; g < ngroups; g++) counts[g] = 0;

  for (i = 0; i < m; i++) {
    OSQPInt gid = group_id[i];

    if (gid == OSQP_NO_GROUP) continue;

    if ((gid < 0) || (gid >= ngroups)) {
      c_eprint("group_id %" OSQP_INT_FMT " out of range (row %" OSQP_INT_FMT
               ", ngroups %" OSQP_INT_FMT ")", gid, i, ngroups);
      return -1;
    }

    counts[gid]++;
    ngrouped++;
  }

  /* An empty group has no type or weight to read, and is most often an
     ngroups that does not match the labels actually used */
  for (g = 0; g < ngroups; g++) {
    if (counts[g] == 0) {
      c_eprint("penalty group %" OSQP_INT_FMT " is empty", g);
      return -1;
    }
  }

  return ngrouped;
}

static void penalty_fill_groups(OSQPPenaltyData* pen,
                                const OSQPInt*   group_id,
                                OSQPInt          m,
                                OSQPInt*         ptr,
                                OSQPInt*         rows,
                                const OSQPInt*   counts) {

  OSQPInt i, g;

  ptr[0] = 0;
  for (g = 0; g < pen->ngroups; g++) ptr[g + 1] = ptr[g] + counts[g];

  /* ptr is restored by the walk: each group's cursor ends at its successor */
  for (i = 0; i < m; i++)
    if (group_id[i] != OSQP_NO_GROUP) rows[ptr[group_id[i]]++] = i;

  for (g = pen->ngroups; g > 0; g--) ptr[g] = ptr[g - 1];
  ptr[0] = 0;

  OSQPVectori_from_raw(pen->group_ptr,  ptr);
  OSQPVectori_from_raw(pen->group_rows, rows);
}

/* NB: braces are required, c_eprint expands to several statements */
static void penalty_report_errors(OSQPInt flags) {

  if (flags & OSQP_PENALTY_ERR_NEGATIVE) {
    c_eprint("L1L2 penalty requires alpha1 >= 0 and alpha2 >= 0");
  }

  if (flags & OSQP_PENALTY_ERR_INFINITE) {
    c_eprint("penalty weights must be finite and below OSQP_INFTY; "
             "use OSQP_PENALTY_NONE for a hard constraint");
  }

  if (flags & OSQP_PENALTY_ERR_HUBER_W) {
    c_eprint("Huber penalty requires alpha1 > 0");
  }

  if (flags & OSQP_PENALTY_ERR_HUBER_D) {
    c_eprint("Huber penalty requires delta > 0");
  }

  if (flags & OSQP_PENALTY_ERR_GROUP_TYPE) {
    c_eprint("OSQP_PENALTY_NORM2 and OSQP_PENALTY_NORMINF rows must name a "
             "group through group_id, and only those rows may");
  }

  if (flags & OSQP_PENALTY_ERR_GROUP_MIXED) {
    c_eprint("every row of a penalty group must agree on its type and alpha1");
  }

  if (flags & OSQP_PENALTY_ERR_GROUP_WEIGHT) {
    c_eprint("a group penalty requires a finite alpha1 > 0, with alpha2 and "
             "delta left at 0");
  }
}

/* Validate a default type and an optional per-row type array */
static OSQPInt penalty_validate_types(OSQPInt        default_type,
                                      const OSQPInt* type,
                                      OSQPInt        m) {
  OSQPInt j;

  if (!penalty_type_is_valid(default_type)) {
    c_eprint("default penalty type %" OSQP_INT_FMT " not recognized", default_type);
    return 1;
  }

  if (type) {
    for (j = 0; j < m; j++) {
      if (!penalty_type_is_valid(type[j])) {
        c_eprint("penalty type %" OSQP_INT_FMT " not recognized (row %" OSQP_INT_FMT ")",
                 type[j], j);
        return 1;
      }
    }
  }

  return 0;
}

/* Commit a type layout. Does not allocate. */
static void penalty_commit_types(OSQPPenaltyData* pen,
                                 OSQPInt          default_type,
                                 const OSQPInt*   type) {

  pen->default_penalty_type = default_type;
  pen->uniform              = type ? 0 : 1;

  if (type) OSQPVectori_from_raw(pen->type, type);
}

/* Validate staged weights, in the caller's units, against a type layout */
static OSQPInt penalty_check_params(OSQPWorkspace*     work,
                                    const OSQPVectorf* alpha1,
                                    const OSQPVectorf* alpha2,
                                    const OSQPVectorf* delta,
                                    const OSQPVectori* type,
                                    OSQPInt            default_type) {

  OSQPPenaltyData* pen = work->data->penalty;

  OSQPInt flags = OSQPVectorf_penalty_params_check(alpha1, alpha2, delta,
                                                   type, default_type,
                                                   work->penalty_flags_tmp);

  /* Also with ngroups == 0, where it rejects a group type nothing can pool */
  flags |= OSQPVectorf_penalty_groups_check(alpha1, alpha2, delta,
                                            type, default_type,
                                            pen->group_ptr, pen->group_rows,
                                            pen->ngroups,
                                            work->penalty_flags_tmp);

  if (flags) penalty_report_errors(flags);

  return flags ? 1 : 0;
}

/* Redo the Ruiz scaling now that the groups are known.
 *
 * scale_data runs during osqp_setup, before any penalty exists, so the E it
 * produced is not constant on the groups declared here. Recomputing it is the
 * same unscale/rescale/refactor cycle osqp_update_data_mat performs, and it is
 * only needed at setup: membership never changes afterwards.
 *
 * The weights must be zero on entry, so that the cycle's own scaling passes
 * leave them alone and the caller can load them in user units afterwards. */
static OSQPInt penalty_rescale_for_groups(OSQPSolver* solver) {

  OSQPWorkspace* work = solver->work;
  OSQPInt        exitflag;

  unscale_data(solver);
  scale_data(solver);

  exitflag = work->linsys_solver->update_matrices(work->linsys_solver,
                                                  work->data->P, OSQP_NULL,
                                                  OSQPMatrix_get_nz(work->data->P),
                                                  work->data->A, OSQP_NULL,
                                                  OSQPMatrix_get_nz(work->data->A));

  /* NB: braces are required, c_eprint expands to several statements */
  if (exitflag) {
    c_eprint("new KKT matrix is not quasidefinite");
  }

  return exitflag;
}

/* Refresh state derived from the type layout. */
static OSQPInt penalty_finish_types(OSQPSolver* solver) {

  /* The rho classification depends on which rows are soft, and osqp_update_rho
   * reuses the cached constr_type, so it has to be recomputed here */
  if (solver->settings->rho_is_vec && update_rho_vec(solver))
    return osqp_error(OSQP_LINSYS_SOLVER_INIT_ERROR);

  return 0;
}

OSQPInt osqp_setup_penalty(OSQPSolver*      solver,
                           OSQPInt          default_type,
                           const OSQPInt*   type,
                           const OSQPFloat* alpha1,
                           const OSQPFloat* alpha2,
                           const OSQPFloat* delta,
                           OSQPInt          ngroups,
                           const OSQPInt*   group_id) {

  OSQPWorkspace*   work;
  OSQPPenaltyData* pen;
  OSQPInt*         counts   = OSQP_NULL;
  OSQPInt*         ptr      = OSQP_NULL;
  OSQPInt*         rows     = OSQP_NULL;
  OSQPInt          ngrouped = 0;
  OSQPInt          m;

  /* Check if workspace has been initialized */
  if (!solver || !solver->work || !solver->work->data)
    return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  work = solver->work;
  m    = work->data->m;

  if (work->data->penalty) {
    c_eprint("penalty already set up; use osqp_update_penalty_types");
    return osqp_error(OSQP_DATA_VALIDATION_ERROR);
  }

  if (!alpha1 || !alpha2 || !delta) {
    c_eprint("osqp_setup_penalty requires alpha1, alpha2 and delta; "
             "a soft row is never left without weights");
    return osqp_error(OSQP_DATA_VALIDATION_ERROR);
  }

  if (ngroups < 0) {
    c_eprint("ngroups must not be negative");
    return osqp_error(OSQP_DATA_VALIDATION_ERROR);
  }

  if (ngroups && !group_id) {
    c_eprint("ngroups > 0 requires group_id to say which rows pool");
    return osqp_error(OSQP_DATA_VALIDATION_ERROR);
  }

  if (penalty_validate_types(default_type, type, m))
    return osqp_error(OSQP_DATA_VALIDATION_ERROR);

  /* Membership is validated and laid out before anything is allocated, so
     that the sizes below are known and a bad partition costs no cleanup */
  if (ngroups) {
    counts = c_malloc(ngroups * sizeof(OSQPInt));

    if (!counts) return osqp_error(OSQP_MEM_ALLOC_ERROR);

    ngrouped = penalty_count_groups(group_id, m, ngroups, counts);

    if (ngrouped < 0) {
      c_free(counts);
      return osqp_error(OSQP_DATA_VALIDATION_ERROR);
    }

    ptr  = c_malloc((ngroups + 1) * sizeof(OSQPInt));
    rows = c_malloc(ngrouped * sizeof(OSQPInt));

    if (!ptr || !rows) {
      c_free(counts); c_free(ptr); c_free(rows);
      return osqp_error(OSQP_MEM_ALLOC_ERROR);
    }
  }

  if (penalty_alloc(work, m, ngroups, ngrouped)) {
    c_free(counts); c_free(ptr); c_free(rows);
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }

  pen = work->data->penalty;

  if (ngroups) {
    penalty_fill_groups(pen, group_id, m, ptr, rows, counts);
    c_free(counts); c_free(ptr); c_free(rows);
  }

  penalty_commit_types(pen, default_type, type);

  if (m > 0) {
    OSQPVectorf_from_raw(pen->alpha1, alpha1);
    OSQPVectorf_from_raw(pen->alpha2, alpha2);
    OSQPVectorf_from_raw(pen->delta,  delta);

    /* Validated in the caller's units, and before the rescale below, so that a
       rejected call leaves the problem data exactly as it found it */
    if (penalty_check_params(work, pen->alpha1, pen->alpha2, pen->delta,
                             penalty_types(pen), pen->default_penalty_type)) {
      penalty_free(work);
      return osqp_error(OSQP_DATA_VALIDATION_ERROR);
    }
  }

  if (ngroups && solver->settings->scaling) {
    /* The rescale runs its own scaling passes over the weights, so zero them
       across it and load them again afterwards in the caller's units */
    OSQPVectorf_set_scalar(pen->alpha1, 0.0);
    OSQPVectorf_set_scalar(pen->alpha2, 0.0);
    OSQPVectorf_set_scalar(pen->delta,  0.0);

    if (penalty_rescale_for_groups(solver)) {
      penalty_free(work);
      return osqp_error(OSQP_LINSYS_SOLVER_INIT_ERROR);
    }

    if (m > 0) {
      OSQPVectorf_from_raw(pen->alpha1, alpha1);
      OSQPVectorf_from_raw(pen->alpha2, alpha2);
      OSQPVectorf_from_raw(pen->delta,  delta);
    }
  }

  if (m > 0) penalty_scale(solver);

  return penalty_finish_types(solver);
}

OSQPInt osqp_update_penalty_types(OSQPSolver*    solver,
                                  OSQPInt        default_type,
                                  const OSQPInt* type) {

  OSQPWorkspace*   work;
  OSQPPenaltyData* pen;
  OSQPVectorf*     alpha1;
  OSQPVectorf*     alpha2;
  OSQPVectorf*     delta;

  /* Check if workspace has been initialized */
  if (!solver || !solver->work || !solver->work->data)
    return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  work = solver->work;
  pen  = work->data->penalty;

  if (!pen) {
    c_eprint("no penalty has been set up; call osqp_setup_penalty first");
    return osqp_error(OSQP_DATA_NOT_INITIALIZED);
  }

  if (penalty_validate_types(default_type, type, work->data->m))
    return osqp_error(OSQP_DATA_VALIDATION_ERROR);

  /* Stage parameters in user units because a type change can alter the power
   * of E applied to them. Rejected updates leave the stored values untouched. */
  if (work->data->m > 0) {
    if (type) OSQPVectori_from_raw(work->penalty_type_tmp, type);

    alpha1 = work->z_prev;
    alpha2 = work->delta_y;
    delta  = work->Ax;

    OSQPVectorf_copy(alpha1, pen->alpha1);
    OSQPVectorf_copy(alpha2, pen->alpha2);
    OSQPVectorf_copy(delta,  pen->delta);

    if (solver->settings->scaling)
      OSQPVectorf_ew_scale_penalty(alpha1, alpha2, delta,
                                   penalty_types(pen), pen->default_penalty_type,
                                   work->scaling->c, work->scaling->E, 1);

    if (penalty_check_params(work, alpha1, alpha2, delta,
                             type ? work->penalty_type_tmp : OSQP_NULL,
                             default_type))
      return osqp_error(OSQP_DATA_VALIDATION_ERROR);

    if (solver->settings->scaling)
      OSQPVectorf_ew_scale_penalty(alpha1, alpha2, delta,
                                   type ? work->penalty_type_tmp : OSQP_NULL,
                                   default_type,
                                   work->scaling->c, work->scaling->E, 0);
  }

  penalty_commit_types(pen, default_type, type);

  if (work->data->m > 0) {
    OSQPVectorf_copy(pen->alpha1, alpha1);
    OSQPVectorf_copy(pen->alpha2, alpha2);
    OSQPVectorf_copy(pen->delta,  delta);
  }

  return penalty_finish_types(solver);
}

OSQPInt osqp_update_penalty_params(OSQPSolver*      solver,
                                   const OSQPFloat* alpha1_new,
                                   const OSQPFloat* alpha2_new,
                                   const OSQPFloat* delta_new) {

  OSQPWorkspace*   work;
  OSQPPenaltyData* pen;
  OSQPVectorf*     alpha1;
  OSQPVectorf*     alpha2;
  OSQPVectorf*     delta;

  /* Check if workspace has been initialized */
  if (!solver || !solver->work || !solver->work->data)
    return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  work = solver->work;
  pen  = work->data->penalty;

  if (!pen) {
    c_eprint("no penalty has been set up; call osqp_setup_penalty first");
    return osqp_error(OSQP_DATA_NOT_INITIALIZED);
  }

  if (work->data->m == 0) return 0;

  /* Assemble the candidate in workspace vectors that only carry state during a
   * solve, the same way osqp_update_data_vec borrows z_prev and delta_y */
  alpha1 = work->z_prev;
  alpha2 = work->delta_y;
  delta  = work->Ax;

  OSQPVectorf_copy(alpha1, pen->alpha1);
  OSQPVectorf_copy(alpha2, pen->alpha2);
  OSQPVectorf_copy(delta,  pen->delta);

  // Merge the new values over the current ones in the caller's units
  if (solver->settings->scaling)
    OSQPVectorf_ew_scale_penalty(alpha1, alpha2, delta,
                                 penalty_types(pen), pen->default_penalty_type,
                                 work->scaling->c, work->scaling->E, 1);

  if (alpha1_new) OSQPVectorf_from_raw(alpha1, alpha1_new);
  if (alpha2_new) OSQPVectorf_from_raw(alpha2, alpha2_new);
  if (delta_new)  OSQPVectorf_from_raw(delta,  delta_new);

  /* Validate before committing, so a rejected call leaves the penalty alone */
  if (penalty_check_params(work, alpha1, alpha2, delta,
                           penalty_types(pen), pen->default_penalty_type))
    return osqp_error(OSQP_DATA_VALIDATION_ERROR);

  if (solver->settings->scaling)
    OSQPVectorf_ew_scale_penalty(alpha1, alpha2, delta,
                                 penalty_types(pen), pen->default_penalty_type,
                                 work->scaling->c, work->scaling->E, 0);

  OSQPVectorf_copy(pen->alpha1, alpha1);
  OSQPVectorf_copy(pen->alpha2, alpha2);
  OSQPVectorf_copy(pen->delta,  delta);

  return 0;
}

#endif /* ifndef OSQP_EMBEDDED_MODE */

/* SoftQP: side-specific elastic-net setup, CPU builtin backend only. */
OSQPInt osqp_setup_asymmetric_penalty(OSQPSolver* solver, const OSQPInt* type,
 const OSQPFloat* lower1, const OSQPFloat* lower2,
 const OSQPFloat* upper1, const OSQPFloat* upper2) {
  OSQPInt i,m,status; OSQPPenaltyData* pen; OSQPFloat* zeros;
  if(!solver || !solver->work) return OSQP_WORKSPACE_NOT_INIT_ERROR;
  m=solver->work->data->m;
  if(!lower1 || !lower2 || !upper1 || !upper2 || !type) return OSQP_DATA_VALIDATION_ERROR;
  for(i=0;i<m;++i) {
    if(type[i]!=OSQP_PENALTY_NONE && type[i]!=OSQP_PENALTY_L1L2) return OSQP_DATA_VALIDATION_ERROR;
    if(!isfinite(upper1[i]) || !isfinite(upper2[i]) || upper1[i]<0 || upper2[i]<0) return OSQP_DATA_VALIDATION_ERROR;
  }
  zeros=c_calloc(m ? m : 1,sizeof(OSQPFloat));
  if(!zeros) return OSQP_MEM_ALLOC_ERROR;
  status=osqp_setup_penalty(solver,OSQP_PENALTY_NONE,type,lower1,lower2,zeros,0,OSQP_NULL);
  c_free(zeros); if(status) return status;
  pen=solver->work->data->penalty;
  pen->upper1=OSQPVectorf_malloc(m); pen->upper2=OSQPVectorf_malloc(m);
  if(!pen->upper1 || !pen->upper2) { penalty_free(solver->work); return OSQP_MEM_ALLOC_ERROR; }
  OSQPVectorf_from_raw(pen->upper1,upper1); OSQPVectorf_from_raw(pen->upper2,upper2);
  if(solver->settings->scaling)
    OSQPVectorf_ew_scale_penalty(pen->upper1,pen->upper2,pen->delta,penalty_types(pen),pen->default_penalty_type,
      solver->work->scaling->c,solver->work->scaling->E,0);
  return 0;
}

OSQPFloat penalty_asymmetric_reccone(const OSQPSolver* solver, const OSQPVectorf* w, OSQPFloat tol) {
  OSQPWorkspace* work=solver->work; OSQPPenaltyData* pen=work->data->penalty;
  OSQPInt i; OSQPFloat rate=0.0;
  for(i=0;i<w->length;++i) {
    OSQPFloat wi=w->values[i], a,b;
    OSQPInt type=pen->uniform ? pen->default_penalty_type : pen->type->values[i];
    if((wi>0 && work->data->u->values[i]>=OSQP_INFTY*OSQP_MIN_SCALING) ||
       (wi<0 && work->data->l->values[i]<=-OSQP_INFTY*OSQP_MIN_SCALING)) continue;
    a=wi>0 ? pen->upper1->values[i] : pen->alpha1->values[i];
    b=wi>0 ? pen->upper2->values[i] : pen->alpha2->values[i];
    if(type==OSQP_PENALTY_NONE || b>0) { if(c_absval(wi)>tol) return OSQP_INFTY; }
    else rate+=a*c_absval(wi);
  }
  return rate;
}
