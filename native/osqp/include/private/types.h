#ifndef TYPES_H
#define TYPES_H

#include "osqp.h"       //includes user API types

#include "algebra_matrix.h"
#include "algebra_vector.h"
#include "glob_opts.h"

/******************
* Internal types *
******************/

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Linear system solver structure (sublevel objects initialize it differently)
 */

typedef struct linsys_solver LinSysSolver;

/**
 * OSQP Timer for statistics
 */
typedef struct OSQPTimer_ OSQPTimer;

/**
 * Problem scaling matrices stored as vectors
 */
typedef struct {
  OSQPFloat    c;     ///< objective function scaling
  OSQPVectorf* D;     ///< primal variable scaling
  OSQPVectorf* E;     ///< dual variable scaling
  OSQPFloat    cinv;  ///< objective function rescaling
  OSQPVectorf* Dinv;  ///< primal variable rescaling
  OSQPVectorf* Einv;  ///< dual variable rescaling
} OSQPScaling;




# ifndef OSQP_EMBEDDED_MODE

/**
 * Polish structure
 */

typedef struct {
  OSQPMatrix*  Ared;          ///< active rows of A; Ared = vstack[Alow, Aupp]
  OSQPInt      n_active;      ///< number of active constraints
  OSQPVectori* active_flags;  ///< -1/0/1 to indicate lower / inactive / upper active constraints, -2/2 for a soft row on a smooth piece of its penalty
  OSQPVectorf* kkt_diag;      ///< reciprocal of the (2,2) KKT diagonal, packed over the selected rows
  OSQPVectorf* kkt_exact;     ///< the part of that diagonal that is not regularization, packed likewise
  OSQPVectorf* rhs_bound;     ///< value the reduced rhs takes on each selected row, size m
  OSQPVectorf* y_fixed;       ///< dual of the rows left out of Ared, nonzero only on a linear piece of a penalty, size m
  OSQPVectorf* x;             ///< optimal x-solution obtained by polish
  OSQPVectorf* z;             ///< optimal z-solution obtained by polish
  OSQPVectorf* y;             ///< optimal y-solution obtained by polish
  OSQPFloat    obj_val;       ///< objective value at polished solution
  OSQPFloat    penalty_val;   ///< penalty part of obj_val at polished solution
  OSQPFloat    dual_obj_val;  ///< Dual objective value at polished solution
  OSQPFloat    duality_gap;   ///< Duality gap at polished solution
  OSQPFloat    prim_res;      ///< primal residual at polished solution
  OSQPFloat    dual_res;      ///< dual residual at polished solution
} OSQPPolish;
# endif // ifndef OSQP_EMBEDDED_MODE


/**********************************
* Main structures and Data Types *
**********************************/

/**
 * Soft-constraint penalty data (possibly internally scaled, like l and u).
 * Active parameters use IEEE infinity internally for the public OSQP_INFTY
 * sentinel; finite scaled parameters may exceed OSQP_INFTY.
 */
typedef struct {
  OSQPInt      uniform;              ///< boolean; every row uses default_penalty_type and type is ignored
  OSQPInt      default_penalty_type; ///< penalty type of every row while uniform
  OSQPVectori* type;                 ///< per-row penalty type (size m)
  OSQPVectorf* alpha1;               ///< penalty parameter, size m
  OSQPVectorf* alpha2;               ///< lower-side quadratic parameter, size m
  OSQPVectorf* upper1;               ///< upper-side linear weight, or NULL
  OSQPVectorf* upper2;               ///< upper-side quadratic weight, or NULL
  OSQPVectorf* delta;                ///< penalty parameter, size m

  /**
   * @name Non-separable penalty groups, in CSR form over the rows
   *
   * Group g owns the rows group_rows[group_ptr[g] .. group_ptr[g+1]-1]. The
   * group's type and weight are read from any of its rows, which validation
   * keeps in agreement. Membership is fixed at setup; ngroups is 0 and the
   * vectors are OSQP_NULL when no group penalty is used.
   * @{
   */
  OSQPInt      ngroups;              ///< number of penalty groups
  OSQPInt      ngrouped;             ///< total grouped rows, i.e. group_ptr[ngroups]
  OSQPVectori* group_ptr;            ///< start of each group in group_rows, size ngroups+1
  OSQPVectori* group_rows;           ///< row indices ordered by group, size ngrouped

  /** @} */
} OSQPPenaltyData;

/**
 * QP problem data (possibly internally scaled)
 */
typedef struct {
  OSQPInt          n;       ///< number of variables n
  OSQPInt          m;       ///< number of constraints m
  OSQPMatrix*      P;       ///< the upper triangular part of the quadratic objective matrix P (size n x n).
  OSQPMatrix*      A;       ///< linear constraints matrix A (size m x n)
  OSQPVectorf*     q;       ///< dense array for linear part of objective function (size n)
  OSQPVectorf*     l;       ///< dense array for lower bound (size m)
  OSQPVectorf*     u;       ///< dense array for upper bound (size m)
  OSQPPenaltyData* penalty; ///< soft-constraint penalties, OSQP_NULL if all rows are hard
} OSQPData;

typedef struct {
    OSQPInt n_ineq_l;  ///< number of inequalities where -inf < l < u
    OSQPInt n_ineq_u;  ///< number of inequalities where l < u < inf
    OSQPInt n_eq;      ///< number of equalities where l == u
    OSQPVectorf *y_l;  ///< for internal use, size m
    OSQPVectorf *y_u;  ///< for internal use, size m
    OSQPVectorf *ryl;  ///< for internal use, size m
    OSQPVectorf *ryu;  ///< for internal use, size m
    OSQPVectorf *rhs;  ///< rhs of linear system to solve for derivatives; length 2*(n + n_ineq_l + n_ineq_u + n_eq)
                       ///< conservatively allocated with length 2(n + 2m) in `osqp_setup`
} OSQPDerivativeData;

/**
 * OSQP Workspace
 */

struct OSQPWorkspace_ {
  /// Problem data to work on (possibly scaled)
  OSQPData* data;

  /// Linear System solver structure
  LinSysSolver* linsys_solver;

# ifndef OSQP_EMBEDDED_MODE
  /// Polish structure
  OSQPPolish* pol;
# endif // ifndef OSQP_EMBEDDED_MODE

  /**
   * @name Vector used to store a vectorized rho parameter
   * @{
   */
  OSQPVectorf* rho_vec;     ///< vector of rho values
  OSQPVectorf* rho_inv_vec; ///< vector of inv rho values

  /** @} */

  OSQPVectori* penalty_type_tmp;     ///< staging for candidate per-row types, size m
  OSQPVectori* penalty_flags_tmp;    ///< one-element integer scratch for backend reductions
  OSQPVectorf* penalty_val_tmp;      ///< one-element float scratch for backend reductions
  OSQPVectorf* penalty_sort_tmp;     ///< group prox staging, size 2*ngrouped (residuals, then sortable magnitudes)

# if OSQP_EMBEDDED_MODE != 1
  OSQPVectori* constr_type; ///< Type of constraints: loose (-1), equality (1), inequality (0)
# endif // if OSQP_EMBEDDED_MODE != 1

  /**
   * @name Iterates
   * @{
   */
  OSQPVectorf* x;           ///< Iterate x
  OSQPVectorf* y;           ///< Iterate y
  OSQPVectorf* z;           ///< Iterate z
  OSQPVectorf* xz_tilde;    ///< Iterate xz_tilde
  OSQPVectorf* xtilde_view; ///< xtilde view into xz_tilde
  OSQPVectorf* ztilde_view; ///< ztilde view into xz_tilde

  OSQPVectorf* x_prev;   ///< Previous x, used also as temp vector in primal info computation
  OSQPVectorf* z_prev;   ///< Previous z, used also as temp vector in dual info computation

  /**
   * @name Primal and dual residuals workspace variables
   *
   * Needed for residuals computation, tolerances computation,
   * approximate tolerances computation and adapting rho
   * @{
   */
  OSQPVectorf* Ax;  ///< scaled A * x
  OSQPVectorf* Px;  ///< scaled P * x
  OSQPVectorf* Aty; ///< scaled A' * y

  /** @} */

  /**
   * @name Objective and duality gap workspace variables
   *
   * Needed for objective/duality gap computation, tolerances computation, and
   * approximate tolerances computation.
   * @{
   */
  OSQPFloat    xtPx;            ///< scaled x' * P *x
  OSQPFloat    qtx;             ///< scaled q' * x
  OSQPFloat    SC;              ///< scaled support function value
  OSQPFloat    penalty_prim;    ///< scaled Phi(R(z)), the penalty part of the primal objective
  OSQPFloat    penalty_dual;    ///< scaled Phi*(y), the penalty part of the dual objective
  OSQPFloat    scaled_dual_gap; ///< scaled primal-dual gap

  /** @} */

  /**
   * @name Primal infeasibility variables
   * @{
   */
  OSQPVectorf* delta_y;   ///< difference between consecutive dual iterates
  OSQPVectorf* Atdelta_y; ///< A' * delta_y

  /** @} */

  /**
   * @name Dual infeasibility variables
   * @{
   */
  OSQPVectorf* delta_x;  ///< difference between consecutive primal iterates
  OSQPVectorf* Pdelta_x; ///< P * delta_x
  OSQPVectorf* Adelta_x; ///< A * delta_x

  /** @} */

  /**
   * @name Temporary vectors used in scaling
   * @{
   */
#if OSQP_EMBEDDED_MODE != 1
  OSQPVectorf* D_temp;   ///< temporary primal variable scaling vectors
  OSQPVectorf* D_temp_A; ///< temporary primal variable scaling vectors storing norms of A columns
  OSQPVectorf* E_temp;   ///< temporary constraints scaling vectors storing norms of A' columns
#endif

  /** @} */
  OSQPScaling* scaling;  ///< scaling vectors

  /// Scaled primal and dual residuals used for computing rho estimate.
  /// They are also passed to indirect linear system solvers for computing required accuracy.
  OSQPFloat scaled_prim_res;
  OSQPFloat scaled_dual_res;

  /// Reciprocal of rho
  OSQPFloat rho_inv;

# ifdef OSQP_ENABLE_PROFILING
  OSQPTimer* timer;         ///< timer object
  OSQPTimer* section_timer; ///< timer for accumulated ADMM sections

  /// flag indicating whether the solve function has been run before
  OSQPInt first_run;

  /// flag indicating whether the update_time should be cleared
  OSQPInt clear_update_time;

  /// flag indicating that osqp_update_rho is called from osqp_solve function
  OSQPInt rho_update_from_solve;
# endif // ifdef OSQP_ENABLE_PROFILING

# ifdef OSQP_ENABLE_PRINTING
  OSQPInt summary_printed; ///< Has last summary been printed? (true/false)
# endif // ifdef OSQP_ENABLE_PRINTING

# ifdef OSQP_ENABLE_DERIVATIVES
  OSQPDerivativeData *derivative_data;
# endif // ifdef OSQP_ENABLE_DERIVATIVES

  /// Flag indicating rho was updated during the solve
  OSQPInt rho_updated;

  /// Relative KKT of last update
  OSQPFloat last_rel_kkt;
};

// NB: "typedef struct OSQPWorkspace_ OSQPWorkspace" is declared already
// in the osqp API where the main OSQPSolver is defined.


/**
 * Define linsys_solver prototype structure
 *
 * NB: The details are defined when the linear solver is initialized depending
 *      on the choice
 */
struct linsys_solver {
  enum osqp_linsys_solver_type type;             ///< linear system solver type functions

  const char* (*name)(LinSysSolver* self);

  OSQPInt (*solve)(LinSysSolver* self,
                   OSQPVectorf*  b,
                   OSQPInt       admm_iter);

  void (*update_settings)(LinSysSolver*       self,
                          const OSQPSettings* settings);

  void (*warm_start)(LinSysSolver*      self,
                     const OSQPVectorf* x);

# ifndef OSQP_EMBEDDED_MODE
  OSQPInt (*adjoint_derivative)(LinSysSolver* self);

  void (*free)(LinSysSolver* self);         ///< free linear system solver (only in desktop version)
# endif // ifndef OSQP_EMBEDDED_MODE

# if OSQP_EMBEDDED_MODE != 1
  OSQPInt (*update_matrices)(LinSysSolver*     self,
                             const OSQPMatrix* P,            ///< update matrices P
                             const OSQPInt*    Px_new_idx,
                             OSQPInt           P_new_n,
                             const OSQPMatrix* A,            //   and A in the solver
                             const OSQPInt*    Ax_new_idx,
                             OSQPInt           A_new_n);

  OSQPInt (*update_rho_vec)(LinSysSolver*      self,
                            const OSQPVectorf* rho_vec,
                            OSQPFloat          rho_sc);  ///< Update rho_vec
# endif // if OSQP_EMBEDDED_MODE != 1

  OSQPInt nthreads; ///< number of threads active
};

#ifdef __cplusplus
}
#endif

#endif /* ifndef TYPES_H */
