/* Modified for SoftQP, 2026-11-01: directional coupled peak support. */
/* Modified for SoftQP, 2026-10-31: native asymmetric penalty support and bindings.
 * Original upstream copyright and Apache-2.0 license are retained. */
#include "osqp.h"
#include "auxil.h"
#include "lin_alg.h"
#include "penalty.h"
#include "scaling.h"
#include "util.h"
#include "printing.h"
#include "timing.h"

/***********************************************************
* Auxiliary functions needed to compute ADMM iterations * *
***********************************************************/
#if OSQP_EMBEDDED_MODE != 1

OSQPFloat compute_rho_estimate(const OSQPSolver* solver) {

  OSQPFloat prim_res, dual_res;           // Primal and dual residuals
  OSQPFloat prim_res_norm, dual_res_norm; // Normalization for the residuals
  OSQPFloat temp_res_norm;                // Temporary residual norm
  OSQPFloat rho_estimate;                 // Rho estimate value

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // Get primal and dual residuals
  prim_res = work->scaled_prim_res;
  dual_res = work->scaled_dual_res;

  // Normalize primal residual
  prim_res_norm = OSQPVectorf_norm_inf(work->z);        // ||z||
  temp_res_norm = OSQPVectorf_norm_inf(work->Ax);       // ||Ax||
  prim_res_norm = c_max(prim_res_norm, temp_res_norm);  // max (||z||,||Ax||)
  prim_res     /= (prim_res_norm + OSQP_DIVISION_TOL);

  // Normalize dual residual
  dual_res_norm = OSQPVectorf_norm_inf(work->data->q);  // ||q||
  temp_res_norm = OSQPVectorf_norm_inf(work->Aty);      // ||A' y||
  dual_res_norm = c_max(dual_res_norm, temp_res_norm);
  temp_res_norm = OSQPVectorf_norm_inf(work->Px);       //  ||P x||
  dual_res_norm = c_max(dual_res_norm, temp_res_norm);  // max(||q||,||A' y||,||P x||)
  dual_res     /= (dual_res_norm + OSQP_DIVISION_TOL);

  // Return rho estimate
  rho_estimate = settings->rho * c_sqrt(prim_res / dual_res);
  rho_estimate = c_min(c_max(rho_estimate, OSQP_RHO_MIN), OSQP_RHO_MAX);

  return rho_estimate;
}

OSQPInt adapt_rho(OSQPSolver* solver) {

  OSQPInt   exitflag; // Exitflag
  OSQPFloat rho_new;  // New rho value

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;

  exitflag = 0;     // Initialize exitflag to 0

  // Compute new rho
  rho_new = compute_rho_estimate(solver);

  // Set rho estimate in info
  info->rho_estimate = rho_new;

  // Check if the new rho is large or small enough and update it in case
  if ((rho_new > settings->rho * settings->adaptive_rho_tolerance) ||
      (rho_new < settings->rho / settings->adaptive_rho_tolerance)) {
    exitflag                 = osqp_update_rho(solver, rho_new);
    info->rho_updates += 1;
    solver->work->rho_updated = 1;
  }

  return exitflag;
}

OSQPInt set_rho_vec(OSQPSolver* solver) {

  OSQPInt constr_types_changed = 0;

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  settings->rho = c_min(c_max(settings->rho, OSQP_RHO_MIN), OSQP_RHO_MAX);

  /* NB: the soft mask goes into the classification rather than being applied
     afterwards, so that the "did anything change?" return value stays exact */
  constr_types_changed = OSQPVectorf_ew_bounds_type(work->constr_type,
                                                    work->data->l,
                                                    work->data->u,
                                                    penalty_row_types(solver),
                                                    penalty_default_type(solver),
                                                    OSQP_RHO_TOL,
                                                    OSQP_INFTY * OSQP_MIN_SCALING);


  //NB: Always refresh the complete rho vector, since the rho_vals
  //might be garbage if they have not been initialised yet.  This means
  //that there is some wasted effort in the case that the constraint types
  //haven't changed and the rho values are already correct, but such is life.
  OSQPVectorf_set_scalar_conditional(work->rho_vec,
                                     work->constr_type,
                                     OSQP_RHO_MIN,                             //const  == -1
                                     settings->rho,                       //constr == 0
                                     OSQP_RHO_EQ_OVER_RHO_INEQ*settings->rho); //constr == 1

  OSQPVectorf_ew_reciprocal(work->rho_inv_vec, work->rho_vec);

  return constr_types_changed;
}

OSQPInt update_rho_vec(OSQPSolver* solver) {

  OSQPInt constr_type_changed;
  OSQPInt exitflag = 0;
  OSQPWorkspace* work = solver->work;

  //update rho_vec and see if anything changed
  constr_type_changed = set_rho_vec(solver);

  // Update rho_vec in KKT matrix if constraints type has changed
  if (constr_type_changed == 1) {
    exitflag = work->linsys_solver->update_rho_vec(work->linsys_solver, work->rho_vec, solver->settings->rho);
  }

  return exitflag;
}

#endif // OSQP_EMBEDDED_MODE != 1


void swap_vectors(OSQPVectorf** a,
                  OSQPVectorf** b) {
  OSQPVectorf* temp;

  temp = *b;
  *b   = *a;
  *a   = temp;
}

static void compute_rhs(OSQPSolver* solver) {

  OSQPWorkspace* work     = solver->work;
  OSQPSettings*  settings = solver->settings;

  //part related to x variables
  OSQPVectorf_add_scaled(work->xtilde_view,
                         settings->sigma,work->x_prev,
                         -1., work->data->q);

  //part related to dual variable in the equality constrained QP (nu)
  if (settings->rho_is_vec) {
    OSQPVectorf_ew_prod(work->ztilde_view, work->rho_inv_vec, work->y);
    OSQPVectorf_add_scaled(work->ztilde_view,
                           -1.0, work->ztilde_view,
                           1.0, work->z_prev);
  }
  else {
    OSQPVectorf_add_scaled(work->ztilde_view,
                           1.0, work->z_prev,
                           -work->rho_inv, work->y);
  }
}

void update_xz_tilde(OSQPSolver* solver,
                     OSQPInt     admm_iter) {

  OSQPWorkspace* work = solver->work;

  // Compute right-hand side
  compute_rhs(solver);

  // Solve linear system
#ifdef OSQP_ENABLE_PROFILING
  osqp_tic(work->section_timer);
#endif /* ifdef OSQP_ENABLE_PROFILING */
  work->linsys_solver->solve(work->linsys_solver, work->xz_tilde, admm_iter);
#ifdef OSQP_ENABLE_PROFILING
  solver->info->linear_system_time += osqp_toc(work->section_timer);
#endif /* ifdef OSQP_ENABLE_PROFILING */
}

void update_x(OSQPSolver* solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // update x
  OSQPVectorf_add_scaled(work->x,
                         settings->alpha,work->xtilde_view,
                         (1.0 - settings->alpha),work->x_prev);

  // update delta_x
  OSQPVectorf_minus(work->delta_x,work->x,work->x_prev);
}

void update_z(OSQPSolver* solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // update z
  if (settings->rho_is_vec) {
    OSQPVectorf_ew_prod(work->z, work->rho_inv_vec,work->y);
    OSQPVectorf_add_scaled3(work->z,
                            1., work->z,
                            settings->alpha, work->ztilde_view,
                            (1.0 - settings->alpha), work->z_prev);
  }
  else {
    OSQPVectorf_add_scaled3(work->z,
                            settings->alpha, work->ztilde_view,
                            (1.0 - settings->alpha), work->z_prev,
                            work->rho_inv, work->y);
  }

  // project z onto C = [l,u], softened by the row penalties
  penalty_project(solver, work->z, work->z);
}

void update_y(OSQPSolver* solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  OSQPVectorf_add_scaled3(work->delta_y,
                          settings->alpha, work->ztilde_view,
                          (1.0 - settings->alpha), work->z_prev,
                          -1.0, work->z);

  if (settings->rho_is_vec) {
    OSQPVectorf_ew_prod(work->delta_y, work->delta_y, work->rho_vec);
  }
  else {
    OSQPVectorf_mult_scalar(work->delta_y, settings->rho);
  }

  OSQPVectorf_plus(work->y, work->y, work->delta_y);
  penalty_peak_zero_dual(solver);

}

void compute_obj_val_dual_gap(const OSQPSolver*  solver,
                              const OSQPVectorf* x,
                              const OSQPVectorf* y,
                              const OSQPVectorf* z,
                                    OSQPFloat*   prim_obj_val,
                                    OSQPFloat*   penalty_val,
                                    OSQPFloat*   dual_obj_val,
                                    OSQPFloat*   duality_gap) {
  OSQPFloat quad_term = 0.0;
  OSQPFloat lin_term  = 0.0;
  OSQPFloat sup_term  = 0.0;
  OSQPFloat pen_term  = 0.0;
  OSQPFloat conj_term = 0.0;
  OSQPWorkspace* work = solver->work;

  /* NB: The function is always called after dual_res is computed */
  quad_term = OSQPVectorf_dot_prod(work->Px, x);
  lin_term  = OSQPVectorf_dot_prod(work->data->q, x);

  /* Compute the support function of the constraints, SC(y) = u'*max(y, 0) + l'*min(y, 0)
     by projecting y onto the polar of the recession cone of C=[l,u], then doing the dot products */
  /* NB: soft rows keep their bounds here. g = I_[l,u] inf-conv Phi gives
     g*(y) = SC(y) + Phi*(y), so the support function is over the real [l,u]
     and the penalty enters only through the Phi*(y) term below. Only the
     infeasibility certificates treat a soft row as free. */
  OSQPVectorf_copy(work->z_prev, y);
  OSQPVectorf_project_polar_reccone(work->z_prev,
                                    work->data->l,
                                    work->data->u,
                                    OSQP_NULL, OSQP_PENALTY_NONE,
                                    OSQP_INFTY * OSQP_MIN_SCALING);

  // Round anything in the range [-OSQP_ZERO_DEADZONE, OSQP_ZERO_DEADZONE] to 0 to
  // prevent very small (i.e., 1e-20) values from blowing up the numerics.
  OSQPVectorf_round_to_zero(work->z_prev, OSQP_ZERO_DEADZONE);

  sup_term  = OSQPVectorf_dot_prod_signed(work->data->u, work->z_prev, +1);
  sup_term += OSQPVectorf_dot_prod_signed(work->data->l, work->z_prev, -1);

  /* The penalty terms, which are zero unless a penalty is set up. Both scale
     by c like the terms above, so the unscaling below covers them too. */
  pen_term  = penalty_obj_value(solver, z);
  conj_term = penalty_conj_value(solver, y);

  /* Primal objective value is 0.5*x^T P x + q^T x + Phi(R(z)) */
  *prim_obj_val = 0.5 * quad_term + lin_term + pen_term;
  *penalty_val  = pen_term;

  /* Dual objective value is -0.5*x^T P x - SC(y) - Phi*(y) */
  *dual_obj_val = -0.5 * quad_term - sup_term - conj_term;

  /* Duality gap is x^T P x + q^T x + Phi(R(z)) + SC(y) + Phi*(y) */
  work->scaled_dual_gap = quad_term + lin_term + pen_term + sup_term + conj_term;

  if (solver->settings->scaling) {
    *prim_obj_val *= work->scaling->cinv;
    *penalty_val  *= work->scaling->cinv;
    *dual_obj_val *= work->scaling->cinv;

    // We always store the duality gap in the info as unscaled (since it is for the user),
    // but we keep the scaled version to use as a termination check when requested.
    *duality_gap = work->scaling->cinv * work->scaled_dual_gap;
  } else {
    *duality_gap = work->scaled_dual_gap;
  }

  /* Save cost values for later use in termination tolerance computation */
  work->xtPx         = quad_term;
  work->qtx          = lin_term;
  work->SC           = sup_term;
  work->penalty_prim = pen_term;
  work->penalty_dual = conj_term;
}

static OSQPFloat compute_duality_gap_tol(const OSQPSolver* solver,
                                               OSQPFloat   eps_abs,
                                               OSQPFloat   eps_rel) {
  OSQPFloat max_rel_eps = 0.0;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  /* Compute max{ |x'*P*x|, |q'*x|, |SC(y)|, |Phi(R(z))|, |Phi*(y)|} */
  max_rel_eps = c_absval(work->xtPx);                              /* |x'P*x| */
  max_rel_eps = c_max(max_rel_eps, c_absval(work->qtx));           /* |q'*x| */
  max_rel_eps = c_max(max_rel_eps, c_absval(work->SC));            /* |SC(y)| */
  max_rel_eps = c_max(max_rel_eps, c_absval(work->penalty_prim));  /* |Phi(R(z))| */
  max_rel_eps = c_max(max_rel_eps, c_absval(work->penalty_dual));  /* |Phi*(y)| */

  /* Unscale the termination tolerance if required*/
  if (settings->scaling && !settings->scaled_termination) {
    max_rel_eps = work->scaling->cinv * max_rel_eps;
  }

  // eps_duality_gap
  return eps_abs + eps_rel * max_rel_eps;
}

static OSQPFloat compute_prim_res(OSQPSolver*        solver,
                                  const OSQPVectorf* x,
                                  const OSQPVectorf* z) {

  // NB: Use z_prev as working vector
  // pr = Ax - z

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;
  OSQPFloat prim_res;

  OSQPMatrix_Axpy(work->data->A,x,work->Ax, 1.0, 0.0); //Ax = A*x
  OSQPVectorf_minus(work->z_prev, work->Ax, z);

  work->scaled_prim_res = OSQPVectorf_norm_inf(work->z_prev);

  // If scaling active -> rescale residual
  if (settings->scaling && !settings->scaled_termination) {
    prim_res =  OSQPVectorf_scaled_norm_inf(work->scaling->Einv, work->z_prev);
  }
  else{
    prim_res  = work->scaled_prim_res;
  }
  return prim_res;
}

static OSQPFloat compute_prim_tol(const OSQPSolver* solver,
                                  OSQPFloat         eps_abs,
                                  OSQPFloat         eps_rel) {

  OSQPFloat max_rel_eps, temp_rel_eps;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // max_rel_eps = max(||z||, ||A x||)
  if (settings->scaling && !settings->scaled_termination) {
    // ||Einv * z||
    max_rel_eps =
    OSQPVectorf_scaled_norm_inf(work->scaling->Einv, work->z);

    // ||Einv * A * x||
    temp_rel_eps =
    OSQPVectorf_scaled_norm_inf(work->scaling->Einv, work->Ax);

    // Choose maximum
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
  }

  else { // No unscaling required
    // ||z||
    max_rel_eps = OSQPVectorf_norm_inf(work->z);

    // ||A * x||
    temp_rel_eps = OSQPVectorf_norm_inf(work->Ax);

    // Choose maximum
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
  }

  // eps_prim
  return eps_abs + eps_rel * max_rel_eps;
}

static OSQPFloat compute_dual_res(OSQPSolver*        solver,
                                  const OSQPVectorf* x,
                                  const OSQPVectorf* y) {

  // NB: Use x_prev as temporary vector
  // NB: Only upper triangular part of P is stored.
  // dr = q + A'*y + P*x

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;
  OSQPFloat      dual_res;

  // dr = q
  OSQPVectorf_copy(work->x_prev, work->data->q);

  // Px = P * x
  OSQPMatrix_Axpy(work->data->P, x, work->Px, 1.0, 0.0);

  // dr += Px
  OSQPVectorf_plus(work->x_prev, work->x_prev, work->Px);

  // dr += A' * y
  if (work->data->m) {
    OSQPMatrix_Atxpy(work->data->A, y, work->Aty, 1.0, 0.0);
    OSQPVectorf_plus(work->x_prev, work->x_prev, work->Aty);
  }

  work->scaled_dual_res = OSQPVectorf_norm_inf(work->x_prev);

  // If scaling active -> rescale residual
  if (settings->scaling && !settings->scaled_termination) {
    dual_res =  work->scaling->cinv * OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                                                  work->x_prev);
  }
  else {
    dual_res = work->scaled_dual_res;
  }

  return dual_res;
}

static OSQPFloat compute_dual_tol(const OSQPSolver* solver,
                                  OSQPFloat         eps_abs,
                                  OSQPFloat         eps_rel) {

  OSQPFloat max_rel_eps, temp_rel_eps;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // max_rel_eps = max(||q||, ||A' y|, ||P x||)
  if (settings->scaling && !settings->scaled_termination) {
    // || Dinv q||
    max_rel_eps =
    OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                work->data->q);

    // || Dinv A' y ||
    temp_rel_eps =
    OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                work->Aty);

    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);

    // || Dinv P x||
    temp_rel_eps =
    OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                work->Px);

    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);

    // Multiply by cinv
    max_rel_eps *= work->scaling->cinv;
  } else { // No scaling required
    // ||q||
    max_rel_eps = OSQPVectorf_norm_inf(work->data->q);

    // ||A'*y||
    temp_rel_eps = OSQPVectorf_norm_inf(work->Aty);
    max_rel_eps  = c_max(max_rel_eps, temp_rel_eps);

    // ||P*x||
    temp_rel_eps = OSQPVectorf_norm_inf(work->Px);
    max_rel_eps  = c_max(max_rel_eps, temp_rel_eps);
  }

  // eps_dual
  return eps_abs + eps_rel * max_rel_eps;
}

OSQPInt is_primal_infeasible(OSQPSolver* solver,
                             OSQPFloat   eps_prim_inf) {

  /* This function checks the primal infeasibility criteria
     1) || A' * delta_y || < eps * ||delta_y||
     2) u'*max(delta_y, 0) + l'*min(delta_y, 0) < 0
   */

  OSQPFloat norm_delta_y;
  OSQPFloat ineq_lhs = 0.0;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  /* Project delta_y onto the polar of the recession cone of C=[l,u].
     Soft rows are treated as free: the soft problem is infeasible if and only
     if the hard subsystem is, so a certificate may only use hard rows. */
  OSQPVectorf_project_polar_reccone(work->delta_y,
                                    work->data->l,
                                    work->data->u,
                                    penalty_row_types(solver),
                                    penalty_default_type(solver),
                                    OSQP_INFTY * OSQP_MIN_SCALING);

  // Compute infinity norm of delta_y (unscale if necessary)
  if (settings->scaling && !settings->scaled_termination) {
    // Use work->Adelta_x as temporary vector
    OSQPVectorf_ew_prod(work->Adelta_x,
                        work->scaling->E,
                        work->delta_y);
    norm_delta_y = OSQPVectorf_norm_inf(work->Adelta_x);
  }
  else
    norm_delta_y = OSQPVectorf_norm_inf(work->delta_y);

  if (norm_delta_y > OSQP_DIVISION_TOL) {

    /* NB: soft rows need no special case here, the projection above zeroed them */
    ineq_lhs  = OSQPVectorf_dot_prod_signed(work->data->u, work->delta_y, +1);
    ineq_lhs += OSQPVectorf_dot_prod_signed(work->data->l, work->delta_y, -1);

    /* Check if the condition is satisfied */
    if (ineq_lhs < 0.0) {
      OSQPMatrix_Atxpy(work->data->A, work->delta_y, work->Atdelta_y, 1.0, 0.0);

      // Unscale if necessary
      if (settings->scaling && !settings->scaled_termination) {
        OSQPVectorf_ew_prod(work->Atdelta_y,
                            work->Atdelta_y,
                            work->scaling->Dinv);
      }

      return OSQPVectorf_norm_inf(work->Atdelta_y) < eps_prim_inf * norm_delta_y;
    }
  }

  // Conditions not satisfied -> not primal infeasible
  return 0;
}

OSQPInt is_dual_infeasible(OSQPSolver* solver,
                           OSQPFloat   eps_dual_inf) {

  /* This function checks the dual infeasibility criteria
     1) q * delta_x < 0
     2) ||P * delta_x || < eps * || delta_x ||
     3) (A * delta_x)_i > -eps * || delta_x ||,    l_i != -inf
        (A * delta_x)_i <  eps * || delta_x ||,    u_i != +inf
   */

  OSQPFloat norm_delta_x;
  OSQPFloat cost_scaling;
  OSQPFloat qtdx;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // Compute norm of delta_x
  if (settings->scaling && !settings->scaled_termination) { // Unscale if needed

    norm_delta_x =
    OSQPVectorf_scaled_norm_inf(work->scaling->D,
                                work->delta_x);
    cost_scaling = work->scaling->c;
  }
  else {
    norm_delta_x = OSQPVectorf_norm_inf(work->delta_x);
    cost_scaling = 1.0;
  }

  // Prevent 0 division || delta_x || > 0
  if (norm_delta_x > OSQP_DIVISION_TOL) {
    // Normalize delta_x by its norm

    /* vec_mult_scalar(work->delta_x, 1./norm_delta_x, work->data->n); */

    // Check first if q'*delta_x < 0
    qtdx = OSQPVectorf_dot_prod(work->data->q, work->delta_x);

    if (qtdx < 0.0) {
      // Compute product P * delta_x
      OSQPMatrix_Axpy(work->data->P, work->delta_x, work->Pdelta_x, 1.0, 0.0);

      // Scale if necessary
      if (settings->scaling && !settings->scaled_termination) {
        OSQPVectorf_ew_prod(work->Pdelta_x,
                            work->Pdelta_x,
                            work->scaling->Dinv);
      }

      if (work->data->penalty && work->data->penalty->peak_lower &&
          OSQPVectorf_norm_inf(work->Pdelta_x)!=0) return 0;

      // Check if || P * delta_x || = 0
      if (OSQPVectorf_norm_inf(work->Pdelta_x) <
          cost_scaling * eps_dual_inf * norm_delta_x) {
        // Compute A * delta_x
        OSQPMatrix_Axpy(work->data->A, work->delta_x, work->Adelta_x,1.0,0.0);

        /* Evaluate the recession rate before unscaling so A*delta_x, q'dx and
           the penalty parameters use the same units. A blocking row returns
           OSQP_INFTY; a linear-growth row contributes its finite charge. */
        if (work->data->penalty) {
          OSQPPenaltyData* pen = work->data->penalty;

          OSQPFloat rate = pen->peak_lower ?
            penalty_peak_reccone(solver,work->Adelta_x,eps_dual_inf*OSQPVectorf_norm_inf(work->delta_x)) : pen->upper1 ?
            penalty_asymmetric_reccone(solver, work->Adelta_x,
              eps_dual_inf * OSQPVectorf_norm_inf(work->delta_x)) :
            OSQPVectorf_penalty_reccone_rate(work->Adelta_x,
                                             work->data->l, work->data->u,
                                             pen->alpha1, pen->alpha2, pen->delta,
                                             penalty_row_types(solver),
                                             penalty_default_type(solver),
                                             OSQP_INFTY * OSQP_MIN_SCALING,
                                             eps_dual_inf * OSQPVectorf_norm_inf(work->delta_x),
                                             work->penalty_val_tmp);

          /* Grouped rows are skipped above, their charge being a norm over the
             whole group. Both group norms are positively homogeneous, so they
             only ever add a finite rate, never block. */
          if ((rate < OSQP_INFTY) && pen->ngroups && !pen->peak_lower)
            rate += OSQPVectorf_group_penalty_reccone_rate(work->Adelta_x,
                                                           work->data->l, work->data->u,
                                                           pen->alpha1,
                                                           penalty_row_types(solver),
                                                           penalty_default_type(solver),
                                                           pen->group_ptr, pen->group_rows,
                                                           pen->ngroups,
                                                           OSQP_INFTY * OSQP_MIN_SCALING,
                                                           work->penalty_val_tmp);

          if (qtdx + rate >= 0.0) return 0;

          return 1;
        }

        // Scale if necessary
        if (settings->scaling && !settings->scaled_termination) {
          OSQPVectorf_ew_prod(work->Adelta_x, work->Adelta_x, work->scaling->Einv);
        }

        // De Morgan Law Applied to dual infeasibility conditions for A * x
        // NB: Note that MIN_SCALING is used to adjust the infinity value
        // in case the problem is scaled.

        // If you get this far, then all tests passed, so return results from final test
        // Test whether Adelta_x is in the recession cone of C = [l, u]
        /* No penalty data: every row is hard. */
        return OSQPVectorf_in_reccone(work->Adelta_x,
                                      work->data->l,
                                      work->data->u,
                                      OSQP_NULL,
                                      OSQP_NULL,
                                      OSQP_PENALTY_NONE,
                                      OSQP_INFTY * OSQP_MIN_SCALING,
                                      eps_dual_inf * norm_delta_x);
      }
    }
  }

  // Conditions not satisfied -> not dual infeasible
  return 0;
}

OSQPInt has_solution(const OSQPInfo* info) {
  return ((info->status_val != OSQP_PRIMAL_INFEASIBLE) &&
      (info->status_val != OSQP_PRIMAL_INFEASIBLE_INACCURATE) &&
      (info->status_val != OSQP_DUAL_INFEASIBLE) &&
      (info->status_val != OSQP_DUAL_INFEASIBLE_INACCURATE) &&
      (info->status_val != OSQP_NON_CVX) &&
      (info->status_val != OSQP_NUMERICAL_ERROR));
}

void store_solution(OSQPSolver *solver, OSQPSolution* solution) {

#ifndef OSQP_EMBEDDED_MODE
  OSQPFloat norm_vec;
#endif /* ifndef OSQP_EMBEDDED_MODE */

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  /* Bypass function if solution wasn't allocated */
  if (!solution)
    return;


  if (has_solution(info)) {
    // Unscale solution if scaling has been performed
      if (settings->scaling){
          //use x_prev and z_prev as scratch space
          unscale_solution(work->x_prev,work->z_prev, //unscaled solution
                           work->x,work->y,           //scaled solution
                           work);
          OSQPVectorf_to_raw(solution->x, work->x_prev); // primal
          OSQPVectorf_to_raw(solution->y, work->z_prev); // dual
      }
      else{
          OSQPVectorf_to_raw(solution->x, work->x); // primal
          OSQPVectorf_to_raw(solution->y, work->y); // dual
      }
      /* Set infeasibility certificates to NaN */
      OSQPVectorf_set_scalar(work->delta_y, OSQP_NAN);
      OSQPVectorf_set_scalar(work->delta_x, OSQP_NAN);
      OSQPVectorf_to_raw(solution->prim_inf_cert, work->delta_y);
      OSQPVectorf_to_raw(solution->dual_inf_cert, work->delta_x);
  }

  else {

    // No solution present. Solution is NaN
    OSQPVectorf_set_scalar(work->x, OSQP_NAN);
    OSQPVectorf_set_scalar(work->y, OSQP_NAN);
    OSQPVectorf_to_raw(solution->x, work->x); // primal
    OSQPVectorf_to_raw(solution->y, work->y); // dual

    if (info->status_val==OSQP_NUMERICAL_ERROR) {
      OSQPVectorf_set_scalar(work->delta_x,OSQP_NAN);
      OSQPVectorf_set_scalar(work->delta_y,OSQP_NAN);
      OSQPVectorf_to_raw(solution->prim_inf_cert,work->delta_y);
      OSQPVectorf_to_raw(solution->dual_inf_cert,work->delta_x);
    }

    // reset iterates to 0 for next run (they cannot start from NaN)
    osqp_cold_start(solver);


#ifndef OSQP_EMBEDDED_MODE

    // Normalize infeasibility certificates if embedded is off
    // NB: It requires a division
    if ((info->status_val == OSQP_PRIMAL_INFEASIBLE) ||
        ((info->status_val == OSQP_PRIMAL_INFEASIBLE_INACCURATE))) {
      norm_vec = OSQPVectorf_norm_inf(work->delta_y);
      OSQPVectorf_mult_scalar(work->delta_y, 1. / norm_vec);
      OSQPVectorf_to_raw(solution->prim_inf_cert, work->delta_y);

      /* Set dual infeasibility certificate to NaN */
      OSQPVectorf_set_scalar(work->delta_x, OSQP_NAN);
      OSQPVectorf_to_raw(solution->dual_inf_cert, work->delta_x);
    }

    if ((info->status_val == OSQP_DUAL_INFEASIBLE) ||
        ((info->status_val == OSQP_DUAL_INFEASIBLE_INACCURATE))) {
      norm_vec = OSQPVectorf_norm_inf(work->delta_x);
      OSQPVectorf_mult_scalar(work->delta_x, 1. / norm_vec);
      OSQPVectorf_to_raw(solution->dual_inf_cert, work->delta_x);

      /* Set primal infeasibility certificate to NaN */
      OSQPVectorf_set_scalar(work->delta_y, OSQP_NAN);
      OSQPVectorf_to_raw(solution->prim_inf_cert, work->delta_y);
    }

#endif /* ifndef OSQP_EMBEDDED_MODE */
  }
}

void update_info(OSQPSolver* solver,
                 OSQPInt     iter,
                 OSQPInt     polishing) {

  OSQPVectorf* x;
  OSQPVectorf* z;
  OSQPVectorf* y;                   // Allocate pointers to vectors

  // objective value, residuals
  OSQPFloat* prim_obj_val;
  OSQPFloat* penalty_val;
  OSQPFloat* dual_obj_val;
  OSQPFloat* dual_gap;
  OSQPFloat* prim_res;
  OSQPFloat* dual_res;

  OSQPInfo*      info     = solver->info;
  OSQPWorkspace* work     = solver->work;

#ifdef OSQP_ENABLE_PROFILING
  OSQPFloat* run_time;                    // Execution time
#endif /* ifdef OSQP_ENABLE_PROFILING */

#ifndef OSQP_EMBEDDED_MODE

  if (polishing) {
    x            = work->pol->x;
    y            = work->pol->y;
    z            = work->pol->z;
    prim_obj_val = &work->pol->obj_val;
    penalty_val  = &work->pol->penalty_val;
    dual_obj_val = &work->pol->dual_obj_val;
    dual_gap     = &work->pol->duality_gap;
    prim_res     = &work->pol->prim_res;
    dual_res     = &work->pol->dual_res;
# ifdef OSQP_ENABLE_PROFILING
    run_time     = &info->polish_time;
# endif /* ifdef OSQP_ENABLE_PROFILING */
  }
  else {
#endif // OSQP_EMBEDDED_MODE
    x            = work->x;
    y            = work->y;
    z            = work->z;
    prim_obj_val = &info->obj_val;
    penalty_val  = &info->penalty_val;
    dual_obj_val = &info->dual_obj_val;
    dual_gap     = &info->duality_gap;
    prim_res     = &info->prim_res;
    dual_res     = &info->dual_res;
    info->iter   = iter;
#ifdef OSQP_ENABLE_PROFILING
    run_time     = &info->solve_time;
#endif /* ifdef OSQP_ENABLE_PROFILING */
#ifndef OSQP_EMBEDDED_MODE
}

#endif /* ifndef OSQP_EMBEDDED_MODE */

  // Compute primal residual
  if (work->data->m == 0) {
    // No constraints -> Always primal feasible
    *prim_res = 0.;
  } else {
    *prim_res = compute_prim_res(solver, x, z);
  }

  // Compute dual residual; store P*x in work->Px
  *dual_res = compute_dual_res(solver, x, y);

  // Compute the objective and duality gap, store various temp values in work
  compute_obj_val_dual_gap(solver, x, y, z, prim_obj_val, penalty_val,
                           dual_obj_val, dual_gap);

  // Compute the duality gap integral
  if (!polishing) {
    info->primdual_int += c_absval(*dual_gap);
  }

  // Update timing
#ifdef OSQP_ENABLE_PROFILING
  *run_time = osqp_toc(work->timer);
#endif /* ifdef OSQP_ENABLE_PROFILING */

  // Compute the relative KKT error
  info->rel_kkt_error = c_max( c_max(*dual_res, *prim_res), *dual_gap);

#ifdef OSQP_ENABLE_PRINTING
  work->summary_printed = 0; // The just updated info have not been printed
#endif /* ifdef OSQP_ENABLE_PRINTING */
}


void reset_info(OSQPInfo *info) {
#ifdef OSQP_ENABLE_PROFILING

  // Initialize info values.
  info->solve_time = 0.0;  // Solve time to zero
  info->linear_system_time = 0.0;
  info->projection_time = 0.0;
  info->rho_update_time = 0.0;
# ifndef OSQP_EMBEDDED_MODE
  info->polish_time = 0.0; // Polish time to zero
# endif /* ifndef OSQP_EMBEDDED_MODE */

  // NB: We do not reset the setup_time because it is performed only once
#endif /* ifdef OSQP_ENABLE_PROFILING */

  update_status(info, OSQP_UNSOLVED); // Problem is unsolved

#if OSQP_EMBEDDED_MODE != 1
  info->rho_updates = 0;              // Rho updates are now 0
#endif /* if OSQP_EMBEDDED_MODE != 1 */
}

const char *OSQP_STATUS_MESSAGE[] = {
  "",   // Status messages start from 1, so add a buffer
  "solved",
  "solved inaccurate",
  "primal infeasible",
  "primal infeasible inaccurate",
  "dual infeasible",
  "dual infeasible inaccurate",
  "maximum iterations reached",
  "run time limit reached",
  "problem non convex",
  "interrupted",
  "unsolved",
  "numerical error"
};

void update_status(OSQPInfo* info,
                   OSQPInt   status_val) {
  // Update status value
  info->status_val = status_val;

  // Update status string depending on status val
  c_strcpy(info->status, OSQP_STATUS_MESSAGE[status_val]);
}

OSQPInt check_termination(OSQPSolver* solver,
                          OSQPInt     approximate) {

  OSQPFloat eps_prim, eps_dual, eps_duality_gap, eps_prim_inf, eps_dual_inf;
  OSQPInt   exitflag;
  OSQPInt   prim_res_check, dual_res_check, duality_gap_check, prim_inf_check, dual_inf_check;
  OSQPFloat eps_abs, eps_rel;

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  if (work->data->penalty && work->data->penalty->peak_lower &&
      (work->data->penalty->peak_numeric_error || !isfinite(info->obj_val) ||
       !isfinite(info->prim_res) || !isfinite(info->dual_res))) {
    work->data->penalty->peak_numeric_error=1;
    penalty_peak_invalid_info(info); return 1;
  }

  // Initialize variables to 0
  exitflag       = 0;
  prim_res_check = 0; dual_res_check = 0;
  prim_inf_check = 0; dual_inf_check = 0;
  duality_gap_check = 0;

  // Initialize tolerances
  eps_abs      = settings->eps_abs;
  eps_rel      = settings->eps_rel;
  eps_prim_inf = settings->eps_prim_inf;
  eps_dual_inf = settings->eps_dual_inf;

  // If residuals are too large, the problem is probably non convex
  if ((info->prim_res > OSQP_INFTY) ||
      (info->dual_res > OSQP_INFTY)){
    // Looks like residuals are diverging. Probably the problem is non convex!
    // Terminate and report it
    if (work->data->penalty && work->data->penalty->peak_lower) penalty_peak_invalid_info(info);
    else update_status(info, OSQP_NON_CVX);
    info->obj_val     = OSQP_NAN;
    info->penalty_val = OSQP_NAN;
    return 1;
  }

  // If approximate solution required, increase tolerances by 10
  if (approximate) {
    eps_abs      *= 10;
    eps_rel      *= 10;
    eps_prim_inf *= 10;
    eps_dual_inf *= 10;
  }

  // Check residuals
  if (work->data->m == 0) {
    prim_res_check = 1; // No constraints -> Primal feasibility always satisfied
  }
  else {
    // Compute primal tolerance
    eps_prim = compute_prim_tol(solver, eps_abs, eps_rel);

    // Primal feasibility check
    if (info->prim_res < eps_prim) {
      prim_res_check = 1;
    } else {
      // Primal infeasibility check
      prim_inf_check = is_primal_infeasible(solver, eps_prim_inf);
    }
  } // End check if m == 0

  // Compute dual tolerance
  eps_dual = compute_dual_tol(solver, eps_abs, eps_rel);

  // Dual feasibility check
  if (info->dual_res < eps_dual) {
    dual_res_check = 1;
  } else {
    // Check dual infeasibility
    dual_inf_check = is_dual_infeasible(solver, eps_dual_inf);
  }

  if (settings->check_dualgap ) {
    // Compute duality gap tolerance
    eps_duality_gap = compute_duality_gap_tol(solver, eps_abs, eps_rel);

    // Duality gap check
    if (settings->scaling && !settings->scaled_termination) {
      // Use the unscaled duality gap value
      if (c_absval(info->duality_gap) < eps_duality_gap) {
        duality_gap_check = 1;
      }
    } else {
      // Use the scaled duality gap value
      if (c_absval(work->scaled_dual_gap) < eps_duality_gap) {
        duality_gap_check = 1;
      }
    }
  } else {
    // Force to 1 to bypass the check
    duality_gap_check = 1;
  }

  // Compare checks to determine solver status
  if (prim_res_check && dual_res_check && duality_gap_check) {
    // Update final information
    if (approximate) {
      update_status(info, OSQP_SOLVED_INACCURATE);
    } else {
      update_status(info, OSQP_SOLVED);
    }
    exitflag = 1;
  }
  else if (prim_inf_check) {
    // Update final information
    if (approximate) {
      update_status(info, OSQP_PRIMAL_INFEASIBLE_INACCURATE);
    } else {
      update_status(info, OSQP_PRIMAL_INFEASIBLE);
    }

    if (settings->scaling && !settings->scaled_termination) {
      // Update infeasibility certificate
      OSQPVectorf_ew_prod(work->delta_y,
                          work->delta_y,
                          work->scaling->E);
    }
    info->obj_val     = OSQP_INFTY;
    info->penalty_val = OSQP_NAN;
    exitflag            = 1;
  }
  else if (dual_inf_check) {
    // Update final information
    if (approximate) {
      update_status(info, OSQP_DUAL_INFEASIBLE_INACCURATE);
    } else {
      update_status(info, OSQP_DUAL_INFEASIBLE);
    }

    if (settings->scaling && !settings->scaled_termination) {
      // Update infeasibility certificate
      OSQPVectorf_ew_prod(work->delta_x,
                          work->delta_x,
                          work->scaling->D);
    }
    info->obj_val     = -OSQP_INFTY;
    info->penalty_val = OSQP_NAN;
    exitflag            = 1;
  }

  return exitflag;
}


#ifndef OSQP_EMBEDDED_MODE

OSQPInt validate_data(const OSQPCscMatrix* P,
                      const OSQPFloat*     q,
                      const OSQPCscMatrix* A,
                      const OSQPFloat*     l,
                      const OSQPFloat*     u,
                            OSQPInt        m,
                            OSQPInt        n) {
  OSQPInt j, ptr;

  if (!P) {
    c_eprint("Missing quadratic cost matrix P");
    return 1;
  }

  if (!A) {
    c_eprint("Missing constraint matrix A");
    return 1;
  }

  if (!q) {
    c_eprint("Missing linear cost vector q");
    return 1;
  }

  // General dimensions Tests
  if ((n <= 0) || (m < 0)) {
    c_eprint("n must be positive and m nonnegative; n = %i, m = %i",
             (int)n, (int)m);
    return 1;
  }

  // Matrix P
  if (P->m != n) {
    c_eprint("P does not have dimension n x n with n = %i", (int)n);
    return 1;
  }

  if (P->m != P->n) {
    c_eprint("P is not square");
    return 1;
  }

  for (j = 0; j < n; j++) { // COLUMN
    for (ptr = P->p[j]; ptr < P->p[j + 1]; ptr++) {
      if (P->i[ptr] > j) {  // if ROW > COLUMN
        c_eprint("P is not upper triangular");
        return 1;
      }
    }
  }

  // Matrix A
  if ((A->m != m) || (A->n != n)) {
    c_eprint("A does not have dimension %i x %i", (int)m, (int)n);
    return 1;
  }

  // Lower and upper bounds
  for (j = 0; j < m; j++) {
    if (l[j] > u[j]) {
      c_eprint("Lower bound at index %d is greater than upper bound: %.4e > %.4e",
               (int)j, l[j], u[j]);
      return 1;
    }
  }

  return 0;
}

#endif /* ifndef OSQP_EMBEDDED_MODE */


OSQPInt validate_linsys_solver(OSQPInt linsys_solver) {
  /* Verify the algebra backend supports the requested indirect solver */
  if ( (linsys_solver == OSQP_INDIRECT_SOLVER) &&
     (osqp_algebra_linsys_supported() & OSQP_CAPABILITY_INDIRECT_SOLVER) ) {
    return 0;
  }

  /* Verify the algebra backend supports the requested direct solver */
  if ( (linsys_solver == OSQP_DIRECT_SOLVER) &&
     (osqp_algebra_linsys_supported() & OSQP_CAPABILITY_DIRECT_SOLVER) ) {
    return 0;
  }

  // Invalid solver
  return 1;
}


OSQPInt validate_settings(const OSQPSettings* settings,
                          OSQPInt             from_setup) {

  if (!settings) {
    c_eprint("Missing settings!");
    return 1;
  }

  if (from_setup &&
      validate_linsys_solver(settings->linsys_solver)) {
    c_eprint("linsys_solver not recognized");
    return 1;
  }

  if (from_setup &&
      settings->allocate_solution != 0 &&
      settings->allocate_solution != 1) {
    c_eprint("allocate_solution must be either 0 or 1");
    return 1;
  }

  if (settings->verbose != 0 &&
      settings->verbose != 1) {
    c_eprint("verbose must be either 0 or 1");
    return 1;
  }

  if (settings->profiler_level != 0 &&
      settings->profiler_level != 1 &&
      settings->profiler_level != 2) {
    c_eprint("profiler_level must be either 0, 1 or 2");
    return 1;
  }

  if (settings->warm_starting != 0 &&
      settings->warm_starting != 1) {
    c_eprint("warm_start must be either 0 or 1");
    return 1;
  }

  if (from_setup && settings->scaling < 0) {
    c_eprint("scaling must be nonnegative");
    return 1;
  }

  if (settings->polishing != 0 &&
      settings->polishing != 1) {
    c_eprint("polishing must be either 0 or 1");
    return 1;
  }

  if (from_setup && settings->rho <= 0.0) {
    c_eprint("rho must be positive");
    return 1;
  }

  if (from_setup &&
      (settings->rho_is_vec != 0) &&
      (settings->rho_is_vec != 1)) {
    c_eprint("rho_is_vec must be either 0 or 1");
    return 1;
  }

  if (from_setup && settings->sigma <= 0.0) {
    c_eprint("sigma must be positive");
    return 1;
  }

  if (settings->alpha <= 0.0 ||
      settings->alpha >= 2.0) {
    c_eprint("alpha must be strictly between 0 and 2");
    return 1;
  }

  if (settings->cg_max_iter <= 0) {
    c_eprint("cg_max_iter must be positive");
    return 1;
  }

  if (settings->cg_tol_reduction <= 0) {
    c_eprint("cg_tol_reduction must be positive");
    return 1;
  }

  if (settings->cg_tol_fraction <= 0.0 ||
      settings->cg_tol_fraction >= 1.0) {
    c_eprint("cg_tol_fraction must be strictly between 0 and 1");
    return 1;
  }

  if (from_setup &&
      settings->adaptive_rho < 0 ||
      settings->adaptive_rho >= _OSQP_ADAPTIVE_RHO_UPDATE_LAST_VALUE) {
    c_eprint("adaptive_rho not a valid choice");
    return 1;
  }

#if !defined(OSQP_ENABLE_PROFILING)
  if (from_setup && settings->adaptive_rho == OSQP_ADAPTIVE_RHO_UPDATE_TIME) {
    c_eprint("adaptive_rho time-based adaptation requires profiling to be enabled");
    return 1;
  }
#endif /* !defined(OSQP_ENABLE_PROFILING) */

  if (from_setup && settings->adaptive_rho_interval < 0) {
    c_eprint("adaptive_rho_interval must be nonnegative");
    return 1;
  }

  if (from_setup && settings->adaptive_rho_fraction <= 0) {
    c_eprint("adaptive_rho_fraction must be positive");
    return 1;
  }

  if (from_setup && settings->adaptive_rho_tolerance < 1.0) {
    c_eprint("adaptive_rho_tolerance must be >= 1");
    return 1;
  }

  if (settings->max_iter <= 0) {
    c_eprint("max_iter must be positive");
    return 1;
  }

  if (settings->eps_abs < 0.0) {
    c_eprint("eps_abs must be nonnegative");
    return 1;
  }

  if (settings->eps_rel < 0.0) {
    c_eprint("eps_rel must be nonnegative");
    return 1;
  }

  if (settings->eps_rel == 0.0 &&
      settings->eps_abs == 0.0) {
    c_eprint("at least one of eps_abs and eps_rel must be positive");
    return 1;
  }

  if (settings->eps_prim_inf <= 0.0) {
    c_eprint("eps_prim_inf must be positive");
    return 1;
  }

  if (settings->eps_dual_inf <= 0.0) {
    c_eprint("eps_dual_inf must be positive");
    return 1;
  }

  if (settings->scaled_termination != 0 &&
      settings->scaled_termination != 1) {
    c_eprint("scaled_termination must be either 0 or 1");
    return 1;
  }

  if (settings->check_termination < 0) {
    c_eprint("check_termination must be nonnegative");
    return 1;
  }

  if (settings->check_dualgap != 0 &&
      settings->check_dualgap != 1) {
    c_eprint("check_dualgap must be either 0 or 1");
    return 1;
  }

  if (settings->time_limit <= 0.0) {
    c_eprint("time_limit must be positive\n");
    return 1;
  }

  if (settings->delta <= 0.0) {
    c_eprint("delta must be positive");
    return 1;
  }

  if (settings->polish_refine_iter < 0) {
    c_eprint("polish_refine_iter must be nonnegative");
    return 1;
  }

  return 0;
}
