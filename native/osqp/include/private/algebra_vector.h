#ifndef ALGEBRA_VECTOR_H
#define ALGEBRA_VECTOR_H

#include "osqp_api_types.h"

#include "glob_opts.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 *  OSQPVector[fi] types.  Not defined here since it
 *  is implementation-specific
 */

/* integer valued vectors */
typedef struct OSQPVectori_ OSQPVectori;

/* float valued vectors*/
typedef struct OSQPVectorf_ OSQPVectorf;


/* VECTOR FUNCTIONS ----------------------------------------------------------*/

# ifndef OSQP_EMBEDDED_MODE

OSQPInt OSQPVectorf_is_eq(const OSQPVectorf* A,
                          const OSQPVectorf* B,
                          OSQPFloat          tol);

/* malloc/calloc for floats and ints (USES MALLOC/CALLOC) */
OSQPVectorf* OSQPVectorf_malloc(OSQPInt length);
OSQPVectorf* OSQPVectorf_calloc(OSQPInt length);
OSQPVectori* OSQPVectori_malloc(OSQPInt length);
OSQPVectori* OSQPVectori_calloc(OSQPInt length);

/* Return a float vector using a raw array as input (Uses MALLOC) */
OSQPVectorf* OSQPVectorf_new(const OSQPFloat* a,
                             OSQPInt          length);

/* Return an int vector using a raw array as input (Uses MALLOC) */
OSQPVectori* OSQPVectori_new(const OSQPInt* a,
                             OSQPInt        length);

/* Return a copy of a float vector a as output (Uses MALLOC) */
OSQPVectorf* OSQPVectorf_copy_new(const OSQPVectorf* a);

/* Free a float vector */
void OSQPVectorf_free(OSQPVectorf* a);

/* Free an int vector */
void OSQPVectori_free(OSQPVectori* a);

/*
 * Assign the data from array b to vector A starting at the index given by start.
 */
void OSQPVectorf_subvector_assign(OSQPVectorf* A,
                                  OSQPFloat*   b,
                                  OSQPInt      start,
                                  OSQPInt      length,
                                  OSQPFloat    multiplier);

/*
 * Assign a scalar to vector A starting at the index given by start.
 */
void OSQPVectorf_subvector_assign_scalar(OSQPVectorf* A,
                                  OSQPFloat    sc,
                                  OSQPInt      start,
                                  OSQPInt      length);

OSQPVectorf* OSQPVectorf_subvector_byrows(const OSQPVectorf* A,
                                          const OSQPVectori* rows);

OSQPVectorf* OSQPVectorf_concat(const OSQPVectorf* A,
                                const OSQPVectorf* B);

/* Create subview of a larger vector.  Internal data should not be freed.
 * Behavior is otherwise identical to OSQPVectorf (Uses MALLOC)
 */
OSQPVectorf* OSQPVectorf_view(const OSQPVectorf* a,
                              OSQPInt            head,
                              OSQPInt            length);

/* Points existing subview somewhere else.  (Does not use MALLOC)
 * TODO: Get rid of this function
 */
void OSQPVectorf_view_update(OSQPVectorf* a, const OSQPVectorf* b, OSQPInt head, OSQPInt length);

/* Free a view of a float vector */
void OSQPVectorf_view_free(OSQPVectorf* a);

# endif /* ifndef OSQP_EMBEDDED_MODE */


/* Length of the vector (floats) */
OSQPInt OSQPVectorf_length(const OSQPVectorf* a);

/* Length of the vector (ints) */
OSQPInt OSQPVectori_length(const OSQPVectori* a);

/* Pointer to vector data (floats) */
OSQPFloat* OSQPVectorf_data(const OSQPVectorf* a);

/* Copy a float vector a into another vector b (pre-allocated) */
void OSQPVectorf_copy(OSQPVectorf*       b,
                      const OSQPVectorf* a);

/* Copy an array of floats into a into a vector b (pre-allocated) */
void OSQPVectorf_from_raw(OSQPVectorf*     b,
                          const OSQPFloat* a);

/* copy an array of ints into a into a vector b (pre-allocated) */
void OSQPVectori_from_raw(OSQPVectori*   b,
                          const OSQPInt* a);

/* copy a vector into an array of floats (pre-allocated) */
void OSQPVectorf_to_raw(OSQPFloat*         bv,
                        const OSQPVectorf* a);

/* copy a vector into an array of ints (pre-allocated) */
void OSQPVectori_to_raw(OSQPInt*           bv,
                        const OSQPVectori* a);

/* set float vector to scalar */
void OSQPVectorf_set_scalar(OSQPVectorf* a,
                            OSQPFloat    sc);

/* Set float vector to one of three scalars based on sign of vector of ints */
void OSQPVectorf_set_scalar_conditional(OSQPVectorf*       a,
                                        const OSQPVectori* test,
                                        OSQPFloat          val_if_neg,
                                        OSQPFloat          val_if_zero,
                                        OSQPFloat          val_if_pos);

/**
 * Round any values of a between [-tol, tol] to zero, creating a deadband around 0.
 */
void OSQPVectorf_round_to_zero(OSQPVectorf* a,
                               OSQPFloat    tol);

/* multiply float vector by float */
void OSQPVectorf_mult_scalar(OSQPVectorf* a,
                             OSQPFloat    sc);

/* x = a + b.  Set x == a for x += b. */
void OSQPVectorf_plus(OSQPVectorf*       x,
                      const OSQPVectorf* a,
                      const OSQPVectorf* b);

/* x = a - b.  Set x==a for x -= b. */
void OSQPVectorf_minus(OSQPVectorf*      x,
                      const OSQPVectorf* a,
                      const OSQPVectorf* b);

/* x = sca*a + scb*b.  Set (x == a, sca==1.) for x += scb*b. */
void OSQPVectorf_add_scaled(OSQPVectorf*       x,
                            OSQPFloat          sca,
                            const OSQPVectorf* a,
                            OSQPFloat          scb,
                            const OSQPVectorf* b);

/* x = sca*a + scb*b + scc*c.  Set (x == a, sca==1.) for x += scb*b scc*c. */
void OSQPVectorf_add_scaled3(OSQPVectorf*       x,
                             OSQPFloat          sca,
                             const OSQPVectorf* a,
                             OSQPFloat          scb,
                             const OSQPVectorf* b,
                             OSQPFloat          scc,
                             const OSQPVectorf* c);

/* ||v||_inf */
OSQPFloat OSQPVectorf_norm_inf(const OSQPVectorf* v);

/* ||Sv||_inf */
OSQPFloat OSQPVectorf_scaled_norm_inf(const OSQPVectorf* S,
                                      const OSQPVectorf* v);

/* ||a - b||_inf */
OSQPFloat OSQPVectorf_norm_inf_diff(const OSQPVectorf* a,
                                    const OSQPVectorf* b);

/* ||v||2 */
OSQPFloat OSQPVectorf_norm_2(const OSQPVectorf* v);

/* ||v||1 */
OSQPFloat OSQPVectorf_norm_1(const OSQPVectorf* a);

/* Inner product a'b */
OSQPFloat OSQPVectorf_dot_prod(const OSQPVectorf* a,
                               const OSQPVectorf* b);

/* Inner product a'b, but using only the positive or negative
 * terms in b.  Use sign = 1 for positive terms, sign = -1 for
 * negative terms.  Setting any other value for sign will return
 * the normal dot product
 */
OSQPFloat OSQPVectorf_dot_prod_signed(const OSQPVectorf* a,
                                      const OSQPVectorf* b,
                                      OSQPInt            sign);

/* Elementwise product a.*b stored in c.  Set c==a for c *= b */
void OSQPVectorf_ew_prod(OSQPVectorf*       c,
                         const OSQPVectorf* a,
                         const OSQPVectorf* b);

/* check l <= u elementwise.  Returns 1 if inequality is true
 * for every element pair in both vectors
 */
OSQPInt OSQPVectorf_all_leq(const OSQPVectorf* l,
                            const OSQPVectorf* u);

/* Elementwise bounding vectors x = min(max(z,l),u)
 * It is acceptable to assign x = z in this call, so
 * that x = min(max(x,l),u) is allowed
 */
void OSQPVectorf_ew_bound_vec(OSQPVectorf*       x,
                              const OSQPVectorf* z,
                              const OSQPVectorf* l,
                              const OSQPVectorf* u);


/* Elementwise projection of y onto the polar recession cone
   of the set [l u].  Values of +/- infval or larger are
   treated as infinite.

   A soft row is free, so its bounds are ignored and y is set to 0 there.
   type may be OSQP_NULL, in which case every row takes default_type; passing
   OSQP_PENALTY_NONE for it means "no soft rows".
 */
void OSQPVectorf_project_polar_reccone(OSQPVectorf*       y,
                                       const OSQPVectorf* l,
                                       const OSQPVectorf* u,
                                       const OSQPVectori* type,
                                       OSQPInt            default_type,
                                       OSQPFloat          infval);

/* Elementwise test of whether y is in the polar recession
   cone of the set [l u].  Values of +/- infval or larger are
   treated as infinite.  Values in y within tol of zero are treated
   as zero. A soft row with a positive quadratic L1L2 weight has the same
   recession cone as a hard row; other soft rows are free.
 */
OSQPInt OSQPVectorf_in_reccone(const OSQPVectorf* y,
                               const OSQPVectorf* l,
                               const OSQPVectorf* u,
                               const OSQPVectorf* alpha2,
                               const OSQPVectori* type,
                               OSQPInt            default_type,
                               OSQPFloat          infval,
                               OSQPFloat          tol);


/* Soft-constraint penalty operations.  In all three routines the type vector
   may be OSQP_NULL, in which case every row takes default_type.  The penalty
   on row i is

     OSQP_PENALTY_NONE : hard row, phi = indicator of {0}
     OSQP_PENALTY_L1L2 : phi(s) = alpha1*|s| + (alpha2/2)*s^2
     OSQP_PENALTY_HUBER: phi(s) = alpha1*h_delta(s), where h_delta(s) is
                         s^2/2 for |s| <= delta and delta*|s| - delta^2/2 above

   The parameters are the internally scaled ones and may be IEEE infinity,
   which makes the row behave as hard.
 */

/* Elementwise generalized projection z = vbar + prox_{phi/rho}(v - vbar),
   where vbar = min(max(v,l),u).  Reduces to OSQPVectorf_ew_bound_vec when
   every row is hard.  rho_vec may be OSQP_NULL, in which case the scalar rho
   applies to every row.  It is acceptable to assign z == v.
 */
void OSQPVectorf_ew_prox_penalty(OSQPVectorf*       z,
                                 const OSQPVectorf* v,
                                 const OSQPVectorf* l,
                                 const OSQPVectorf* u,
                                 const OSQPVectorf* rho_vec,
                                 OSQPFloat          rho,
                                 const OSQPVectorf* alpha1,
                                 const OSQPVectorf* alpha2,
                                 const OSQPVectorf* delta,
                                 const OSQPVectori* type,
                                 OSQPInt            default_type);

/* Penalty contribution to the objective, sum_i phi_i(R(z)_i), where the
   residual is R(z) = z - min(max(z,l),u).
   scratch is a preallocated one-element float vector for backend reductions.
 */
OSQPFloat OSQPVectorf_penalty_value(const OSQPVectorf* z,
                                    const OSQPVectorf* l,
                                    const OSQPVectorf* u,
                                    const OSQPVectorf* alpha1,
                                    const OSQPVectorf* alpha2,
                                    const OSQPVectorf* delta,
                                    const OSQPVectori* type,
                                    OSQPInt            default_type,
                                    OSQPVectorf*       scratch);

/* Rate at which the penalty grows along the direction w, that is
   sum_i slope_i * dist(w_i, rec[l_i,u_i]) over the rows whose penalty grows
   exactly linearly:

     L1L2 with alpha2 == 0 : slope = alpha1
     HUBER                 : slope = alpha1*delta, the asymptotic slope

   Such a row does not forbid a recession direction, it charges for it, so the
   dual-infeasibility test needs this term on top of q'dx. Returns OSQP_INFTY
   when a hard or superlinear row leaves its recession cone by more than tol.
   Linear rows always charge their exact distance. Values of +/- infval or
   larger are treated as infinite. scratch is a preallocated one-element float
   vector for backend reductions.
 */
OSQPFloat OSQPVectorf_penalty_reccone_rate(const OSQPVectorf* w,
                                           const OSQPVectorf* l,
                                           const OSQPVectorf* u,
                                           const OSQPVectorf* alpha1,
                                           const OSQPVectorf* alpha2,
                                           const OSQPVectorf* delta,
                                           const OSQPVectori* type,
                                           OSQPInt            default_type,
                                           OSQPFloat          infval,
                                           OSQPFloat          tol,
                                           OSQPVectorf*       scratch);

/* Conjugate sum_i phi*_i(y_i), needed by the duality gap.  Returns OSQP_INFTY
   if any row lies outside dom phi*.
   scratch is a preallocated one-element float vector for backend reductions.
 */
OSQPFloat OSQPVectorf_penalty_conj_value(const OSQPVectorf* y,
                                         const OSQPVectorf* alpha1,
                                         const OSQPVectorf* alpha2,
                                         const OSQPVectorf* delta,
                                         const OSQPVectori* type,
                                         OSQPInt            default_type,
                                         OSQPVectorf*       scratch);


/* Non-separable penalty groups.
 *
 * Group g owns the rows group_rows[group_ptr[g] .. group_ptr[g+1]-1] and
 * carries a single penalty

   OSQP_PENALTY_NORM2  : Phi(s_G) = alpha1 * ||s_G||_2
   OSQP_PENALTY_NORMINF: Phi(s_G) = alpha1 * ||s_G||_inf

 * whose type and weight are read from the group's first row; every row of a
 * group is kept in agreement by OSQPVectorf_penalty_groups_check.  All four
 * routines leave rows outside every group untouched, so they compose with the
 * elementwise routines above over a disjoint row set.  Both norms are
 * positively homogeneous, so a group always grows exactly linearly.
 *
 * rho must be constant on a group, which set_rho_vec guarantees by classifying
 * every grouped row as an inequality.
 */

/* Group part of the generalized projection z = vbar + prox_{Phi/rho}(v - vbar),
   applied after the elementwise pass has written vbar + prox_{phi/rho} on the
   ungrouped rows.  z and v may alias.  sort_tmp is a preallocated float vector
   of length 2*ngrouped: the residuals are staged in its first half so that the
   l_inf projection can sort their magnitudes in the second half without losing
   the signs.
 */
void OSQPVectorf_group_prox_penalty(OSQPVectorf*       z,
                                    const OSQPVectorf* v,
                                    const OSQPVectorf* l,
                                    const OSQPVectorf* u,
                                    OSQPFloat          rho,
                                    const OSQPVectorf* alpha1,
                                    const OSQPVectori* type,
                                    OSQPInt            default_type,
                                    const OSQPVectori* group_ptr,
                                    const OSQPVectori* group_rows,
                                    OSQPInt            ngroups,
                                    OSQPVectorf*       sort_tmp);

/* Group part of the objective, sum_g alpha1_g * ||R(z)_G||, where the residual
   is R(z) = z - min(max(z,l),u).
 */
OSQPFloat OSQPVectorf_group_penalty_value(const OSQPVectorf* z,
                                          const OSQPVectorf* l,
                                          const OSQPVectorf* u,
                                          const OSQPVectorf* alpha1,
                                          const OSQPVectori* type,
                                          OSQPInt            default_type,
                                          const OSQPVectori* group_ptr,
                                          const OSQPVectori* group_rows,
                                          OSQPInt            ngroups,
                                          OSQPVectorf*       scratch);

/* Group part of the recession rate along w, that is
   sum_g alpha1_g * || (dist(w_i, rec[l_i,u_i]))_{i in G} ||.
   Values of +/- infval or larger are treated as infinite.
 */
OSQPFloat OSQPVectorf_group_penalty_reccone_rate(const OSQPVectorf* w,
                                                 const OSQPVectorf* l,
                                                 const OSQPVectorf* u,
                                                 const OSQPVectorf* alpha1,
                                                 const OSQPVectori* type,
                                                 OSQPInt            default_type,
                                                 const OSQPVectori* group_ptr,
                                                 const OSQPVectori* group_rows,
                                                 OSQPInt            ngroups,
                                                 OSQPFloat          infval,
                                                 OSQPVectorf*       scratch);

/* Group part of the conjugate.  Both group penalties are norms, so the
   conjugate is the indicator of the dual-norm ball: ||y_G||_2 <= alpha1 for
   NORM2 and ||y_G||_1 <= alpha1 for NORMINF.  Returns OSQP_INFTY outside it
   and zero otherwise.
 */
OSQPFloat OSQPVectorf_group_penalty_conj_value(const OSQPVectorf* y,
                                               const OSQPVectorf* alpha1,
                                               const OSQPVectori* type,
                                               OSQPInt            default_type,
                                               const OSQPVectori* group_ptr,
                                               const OSQPVectori* group_rows,
                                               OSQPInt            ngroups,
                                               OSQPVectorf*       scratch);

# if OSQP_EMBEDDED_MODE != 1

/* Vector elementwise reciprocal b = 1./a (needed for scaling)*/
void OSQPVectorf_ew_reciprocal(OSQPVectorf*       b,
                               const OSQPVectorf* a);

/* elementwise sqrt of the vector elements */
void OSQPVectorf_ew_sqrt(OSQPVectorf* a);

/* Elementwise maximum between vectors c = max(a, b) */
void OSQPVectorf_ew_max_vec(OSQPVectorf*       c,
                            const OSQPVectorf* a,
                            const OSQPVectorf* b);

/* Elementwise minimum between vectors c = min(a, b) */
void OSQPVectorf_ew_min_vec(OSQPVectorf*       c,
                            const OSQPVectorf* a,
                            const OSQPVectorf* b);

/* Elementwise check for constraint type.
   if u[i] - l[i] < tol, iseq[i] = 1 otherwise iseq[i] = 0,
   unless values exceed +/- infval, in which case marked
   as iseq[i] = -1.

   A soft row is never an equality however tight its bounds are, so rows whose
   penalty type is not OSQP_PENALTY_NONE are classified 0 rather than 1.  They
   can still be -1.  type may be OSQP_NULL, in which case every row takes
   default_type; passing OSQP_PENALTY_NONE for it disables the softening.

   Returns 1 if any value in iseq has been modified.   O otherwise.
 */
OSQPInt OSQPVectorf_ew_bounds_type(OSQPVectori*       iseq,
                                   const OSQPVectorf* l,
                                   const OSQPVectorf* u,
                                   const OSQPVectori* type,
                                   OSQPInt            default_type,
                                   OSQPFloat          tol,
                                   OSQPFloat          infval);


/* Replace E on each penalty group by the group's geometric mean, which is what
   makes alpha*||E_G s_G|| a group penalty again: a norm only survives the
   constraint scaling when E is constant on the group.  The correction factor
   Ehat/E is written into corr, 1 on every ungrouped row, so that the caller
   can apply the same factor to the rows of A.  E itself is left untouched.
 */
void OSQPVectorf_group_equalize_scaling(OSQPVectorf*       corr,
                                        const OSQPVectorf* E,
                                        const OSQPVectori* group_ptr,
                                        const OSQPVectori* group_rows,
                                        OSQPInt            ngroups);


/* Scale (invert = 0) or unscale (invert = 1) soft-constraint penalty
   parameters in place, for a problem scaled by the objective factor c and the
   constraint scaling E:

     OSQP_PENALTY_L1L2 : alpha1 *= c/E,    alpha2 *= c/E^2
     OSQP_PENALTY_HUBER: alpha1 *= c/E^2,  delta  *= E

   Rows of type OSQP_PENALTY_NONE are left untouched.  Weights are required to
   be finite (see OSQPVectorf_penalty_params_check), so this is a plain
   multiply.  E may be NULL for unit scaling.  If type is OSQP_NULL every row
   takes default_type.
 */
void OSQPVectorf_ew_scale_penalty(OSQPVectorf*       alpha1,
                                  OSQPVectorf*       alpha2,
                                  OSQPVectorf*       delta,
                                  const OSQPVectori* type,
                                  OSQPInt            default_type,
                                  OSQPFloat          c,
                                  const OSQPVectorf* E,
                                  OSQPInt            invert);


/* Flags returned by OSQPVectorf_penalty_params_check */
#define OSQP_PENALTY_ERR_NEGATIVE  (0x01)   /* alpha1 or alpha2 negative or NaN */
#define OSQP_PENALTY_ERR_INFINITE  (0x02)   /* a weight at or beyond OSQP_INFTY */
#define OSQP_PENALTY_ERR_HUBER_W   (0x04)   /* Huber row with alpha1 <= 0 */
#define OSQP_PENALTY_ERR_HUBER_D   (0x08)   /* Huber row with delta <= 0 */

/* Flags returned by OSQPVectorf_penalty_groups_check */
#define OSQP_PENALTY_ERR_GROUP_TYPE   (0x10)  /* a group type outside any group, or a grouped row without one */
#define OSQP_PENALTY_ERR_GROUP_MIXED  (0x20)  /* rows of one group disagree on type or on alpha1 */
#define OSQP_PENALTY_ERR_GROUP_WEIGHT (0x40)  /* group weight not finite and positive, or alpha2/delta set */

/* Check every row's parameters against its penalty type. Returns 0 if all rows
   are valid, otherwise the OR of the OSQP_PENALTY_ERR_* flags above.
   scratch is a preallocated one-element integer vector for backend reductions.
 */
OSQPInt OSQPVectorf_penalty_params_check(const OSQPVectorf* alpha1,
                                         const OSQPVectorf* alpha2,
                                         const OSQPVectorf* delta,
                                         const OSQPVectori* type,
                                         OSQPInt            default_type,
                                         OSQPVectori*       scratch);

/* Check the group layout against the per-row types and weights.  Verifies that
   group types and group membership name exactly the same rows, that the rows
   of a group agree on their type and on alpha1, and that a group's weight is
   finite and positive with alpha2 and delta unused.  Valid with ngroups = 0,
   where it only rejects a group type outside any group.  Returns 0 if
   everything is consistent, otherwise the OR of the flags above.
   scratch is a preallocated one-element integer vector for backend reductions.
 */
OSQPInt OSQPVectorf_penalty_groups_check(const OSQPVectorf* alpha1,
                                         const OSQPVectorf* alpha2,
                                         const OSQPVectorf* delta,
                                         const OSQPVectori* type,
                                         OSQPInt            default_type,
                                         const OSQPVectori* group_ptr,
                                         const OSQPVectori* group_rows,
                                         OSQPInt            ngroups,
                                         OSQPVectori*       scratch);

/* Elementwise replacement based on lt comparison.
   x[i] = z[i] < testval ? newval : z[i];
*/
void OSQPVectorf_set_scalar_if_lt(OSQPVectorf*       x,
                                  const OSQPVectorf* z,
                                  OSQPFloat          testval,
                                  OSQPFloat          newval);

/* Elementwise replacement based on gt comparison.
 * x[i] = z[i] > testval ? newval : z[i];
 */
void OSQPVectorf_set_scalar_if_gt(OSQPVectorf*       x,
                                  const OSQPVectorf* z,
                                  OSQPFloat          testval,
                                  OSQPFloat          newval);

# endif /* if OSQP_EMBEDDED_MODE != 1 */

#ifdef __cplusplus
}
#endif

#endif /* ifndef ALGEBRA_VECTOR_H */
