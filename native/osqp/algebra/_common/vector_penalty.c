#include <math.h>
#include "osqp.h"
#include "algebra_vector.h"
#include "algebra_impl.h"

/*******************************************************************************
 * Soft-constraint penalty vector operations                                   *
 *                                                                             *
 * Branchy element-wise work with no BLAS equivalent, so the builtin and MKL    *
 * backends share one implementation rather than each carrying a copy.          *
 * Uniform soft penalties deliberately use the same loop as mixed penalties;   *
 * only uniform hard penalties bypass it. CUDA specializes uniform kernels.    *
 *******************************************************************************/

/* prox_{phi/rho}(r) for a single row */
static OSQPFloat ew_prox_penalty_row(OSQPFloat r,
                                     OSQPFloat rho,
                                     OSQPInt   type,
                                     OSQPFloat a1,
                                     OSQPFloat a2,
                                     OSQPFloat d) {

  OSQPFloat t;

  switch (type) {
    case OSQP_PENALTY_L1L2:
      /* rho/(rho + alpha2) * S_{alpha1/rho}(r) */
      t = c_absval(r) - a1 / rho;
      if (t <= 0.0) return 0.0;
      return (r > 0.0 ? t : -t) * rho / (rho + a2);

    case OSQP_PENALTY_HUBER:
      /* Test the quadratic candidate instead of forming (1 + a1/rho)*d,
         whose intermediate ratio can overflow for finite scaled weights.
         Divide r first for large weights, so rho/(rho+a1) cannot underflow;
         otherwise form the bounded ratio first to avoid enlarging r. */
      t = a1 > 1.0 ? (r / (rho + a1)) * rho
                   : (rho / (rho + a1)) * r;
      if (c_absval(t) <= d) return t;
      /* Divide first only when rho reduces the product. */
      t = rho >= 1.0 ? (a1 / rho) * d : (a1 * d) / rho;
      return r - t * (r > 0.0 ? 1.0 : -1.0);

    default:
      return 0.0;   // Hard row
  }
}

/* phi(s) for a single row */
static OSQPFloat penalty_value_row(OSQPFloat s,
                                   OSQPInt   type,
                                   OSQPFloat a1,
                                   OSQPFloat a2,
                                   OSQPFloat d) {

  OSQPFloat as = c_absval(s);

  /* NB: s is exactly zero on a hard row and on any row whose weight is
     infinite, so returning here also avoids forming 0 * inf */
  if (as == 0.0) return 0.0;

  switch (type) {
    case OSQP_PENALTY_L1L2:
      return a1 * as + 0.5 * a2 * s * s;

    case OSQP_PENALTY_HUBER:
      return as <= d ? 0.5 * a1 * s * s : a1 * (d * as - 0.5 * d * d);

    default:
      return 0.0;
  }
}

/* Membership in a conjugate domain |y| <= lim, up to the roundoff tolerance.
   The dual iterate satisfies y in dphi(s) by construction, so it sits exactly
   on the boundary whenever the penalty is linear there; testing lim exactly
   would call the solution dual infeasible and report an infinite gap. */
static OSQPInt penalty_conj_in_domain(OSQPFloat ay,
                                      OSQPFloat lim) {

  return ay <= lim + OSQP_PENALTY_CONJ_TOL * (1.0 + lim);
}

/* phi*(y) for a single row, OSQP_INFTY outside the domain */
static OSQPFloat penalty_conj_row(OSQPFloat y,
                                  OSQPInt   type,
                                  OSQPFloat a1,
                                  OSQPFloat a2,
                                  OSQPFloat d) {

  OSQPFloat ay = c_absval(y);
  OSQPFloat t;

  switch (type) {
    case OSQP_PENALTY_L1L2:
      /* Pure L1 has an indicator conjugate; an infinite quadratic weight
         has a zero conjugate. Handle both before any division or square. */
      if (a2 <= 0.0) return penalty_conj_in_domain(ay, a1) ? 0.0 : OSQP_INFTY;
      t = ay - a1;
      return t <= 0.0 ? 0.0 : (0.5 * t) * (t / a2);

    case OSQP_PENALTY_HUBER:
      t = a1 * d;
      if (!penalty_conj_in_domain(ay, t)) return OSQP_INFTY;
      /* Evaluate at the boundary rather than past it, so the tolerated
         overshoot cannot report more than the true conjugate value */
      ay = c_min(ay, t);
      return (0.5 * ay) * (ay / a1);

    default:
      return 0.0;   // Hard row, phi* = 0
  }
}

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
                                 OSQPInt            default_type) {

  OSQPInt    i;
  OSQPInt    length = z->length;
  OSQPFloat* zv     = z->values;
  OSQPFloat* vv     = v->values;
  OSQPFloat* lv     = l->values;
  OSQPFloat* uv     = u->values;
  OSQPFloat* a1     = alpha1->values;
  OSQPFloat* a2     = alpha2->values;
  OSQPFloat* d      = delta->values;
  OSQPFloat* rv     = rho_vec ? rho_vec->values : OSQP_NULL;
  OSQPInt*   tv     = type ? type->values : OSQP_NULL;

  /* The all-hard case is the plain projection, bit for bit */
  if (!tv && (default_type == OSQP_PENALTY_NONE)) {
    OSQPVectorf_ew_bound_vec(z, v, l, u);
    return;
  }

  for (i = 0; i < length; i++) {
    OSQPInt   t = tv ? tv[i] : default_type;
    OSQPFloat vbar;

    /* Left to OSQPVectorf_group_prox_penalty, which runs first and has already
       written this row. Skipping rather than overwriting is what lets z alias
       v: the group pass needs v intact on its own rows, and this pass needs it
       intact on the rest, so neither may touch the other's. */
    if ((t == OSQP_PENALTY_NORM2) || (t == OSQP_PENALTY_NORMINF)) continue;

    vbar = c_min(c_max(vv[i], lv[i]), uv[i]);

    zv[i] = vbar + ew_prox_penalty_row(vv[i] - vbar, rv ? rv[i] : rho,
                                       t, a1[i], a2[i], d[i]);
  }
}

OSQPFloat OSQPVectorf_penalty_value(const OSQPVectorf* z,
                                    const OSQPVectorf* l,
                                    const OSQPVectorf* u,
                                    const OSQPVectorf* alpha1,
                                    const OSQPVectorf* alpha2,
                                    const OSQPVectorf* delta,
                                    const OSQPVectori* type,
                                    OSQPInt            default_type,
                                    OSQPVectorf*       scratch) {

  OSQPInt    i;
  OSQPFloat  val    = 0.0;
  OSQPInt    length = z->length;
  OSQPFloat* zv     = z->values;
  OSQPFloat* lv     = l->values;
  OSQPFloat* uv     = u->values;
  OSQPFloat* a1     = alpha1->values;
  OSQPFloat* a2     = alpha2->values;
  OSQPFloat* d      = delta->values;
  OSQPInt*   tv     = type ? type->values : OSQP_NULL;

  (void)scratch; /* Only device backends need reduction storage. */

  if (!tv && (default_type == OSQP_PENALTY_NONE)) return 0.0;

  for (i = 0; i < length; i++) {
    OSQPFloat zbar = c_min(c_max(zv[i], lv[i]), uv[i]);

    val += penalty_value_row(zv[i] - zbar, tv ? tv[i] : default_type,
                             a1[i], a2[i], d[i]);
  }

  return val;
}

/* Recession rate of one row along w. Hard and superlinear rows return infinity
 * when the direction leaves the bound recession cone. */
static OSQPFloat penalty_reccone_rate_row(OSQPFloat w,
                                          OSQPFloat l,
                                          OSQPFloat u,
                                          OSQPInt   type,
                                          OSQPFloat a1,
                                          OSQPFloat a2,
                                          OSQPFloat d,
                                          OSQPFloat infval,
                                          OSQPFloat tol) {

  OSQPFloat slope;
  OSQPFloat dist;

  /* Distance from rec[l,u]. Linear penalties charge the exact distance;
     tol only relaxes the membership test for blocking rows. */
  if ((u < +infval) && (w > 0.0))      dist = w;
  else if ((l > -infval) && (w < 0.0)) dist = -w;
  else                                 return 0.0;

  switch (type) {
    case OSQP_PENALTY_L1L2:
      if (a2 > 0.0) return dist > tol ? OSQP_INFTY : 0.0;
      slope = a1;
      break;

    case OSQP_PENALTY_HUBER:
      /* NB: the asymptotic slope of alpha1*h_delta is alpha1*delta */
      slope = a1 * d;
      break;

    case OSQP_PENALTY_NORM2:
    case OSQP_PENALTY_NORMINF:
      /* Charged by OSQPVectorf_group_penalty_reccone_rate, which needs the
         whole group at once. Blocking nothing here is not an approximation:
         both norms are positively homogeneous, so a group never blocks. */
      return 0.0;

    default:
      return dist > tol ? OSQP_INFTY : 0.0;
  }

  return slope * dist;
}

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
                                           OSQPVectorf*       scratch) {

  OSQPInt    i;
  OSQPFloat  rate   = 0.0;
  OSQPInt    length = w->length;
  OSQPFloat* wv     = w->values;
  OSQPFloat* lv     = l->values;
  OSQPFloat* uv     = u->values;
  OSQPFloat* a1     = alpha1->values;
  OSQPFloat* a2     = alpha2->values;
  OSQPFloat* d      = delta->values;
  OSQPInt*   tv     = type ? type->values : OSQP_NULL;

  (void)scratch; /* Only device backends need reduction storage. */

  for (i = 0; i < length; i++) {
    OSQPFloat row_rate = penalty_reccone_rate_row(wv[i], lv[i], uv[i],
                                                  tv ? tv[i] : default_type,
                                                  a1[i], a2[i], d[i], infval, tol);

    if (row_rate >= OSQP_INFTY) return OSQP_INFTY;
    rate += row_rate;
  }

  return rate;
}

OSQPFloat OSQPVectorf_penalty_conj_value(const OSQPVectorf* y,
                                         const OSQPVectorf* alpha1,
                                         const OSQPVectorf* alpha2,
                                         const OSQPVectorf* delta,
                                         const OSQPVectori* type,
                                         OSQPInt            default_type,
                                         OSQPVectorf*       scratch) {

  OSQPInt    i;
  OSQPFloat  val    = 0.0;
  OSQPInt    length = y->length;
  OSQPFloat* yv     = y->values;
  OSQPFloat* a1     = alpha1->values;
  OSQPFloat* a2     = alpha2->values;
  OSQPFloat* d      = delta->values;
  OSQPInt*   tv     = type ? type->values : OSQP_NULL;

  (void)scratch; /* Only device backends need reduction storage. */

  if (!tv && (default_type == OSQP_PENALTY_NONE)) return 0.0;

  for (i = 0; i < length; i++) {
    OSQPFloat c = penalty_conj_row(yv[i], tv ? tv[i] : default_type,
                                   a1[i], a2[i], d[i]);

    if (c >= OSQP_INFTY) return OSQP_INFTY;

    val += c;
  }

  return val;
}

/*******************************************************************************
 * Non-separable penalty groups                                                *
 *                                                                             *
 * Each routine walks the groups in CSR order and touches only their rows, so  *
 * it composes with the elementwise pass above over a disjoint row set.  The    *
 * CUDA backend does not share this code: it runs its own device kernels, in    *
 * algebra/cuda/src/cuda_lin_alg.cu.                                            *
 *******************************************************************************/

/* Size at which the l_inf prox stops insertion-sorting and starts heapsorting.
   |r_G| barely moves between ADMM iterations, so the array arrives nearly
   sorted and insertion sort runs in O(|G|) on the common path; heapsort only
   bounds the rare large group, without recursing. */
#define OSQP_PENALTY_SORT_SMALL (32)

/* Type of a group, which validation keeps equal across its rows */
static OSQPInt group_type(const OSQPInt* tv,
                          OSQPInt        default_type,
                          const OSQPInt* rows,
                          OSQPInt        start) {

  return tv ? tv[rows[start]] : default_type;
}

static void penalty_insertion_sort_desc(OSQPFloat* a,
                                        OSQPInt    n) {

  OSQPInt i, j;

  for (i = 1; i < n; i++) {
    OSQPFloat key = a[i];

    for (j = i - 1; (j >= 0) && (a[j] < key); j--) a[j + 1] = a[j];

    a[j + 1] = key;
  }
}

/* Restore the min-heap rooted at start over a[0..n-1] */
static void penalty_sift_down_min(OSQPFloat* a,
                                  OSQPInt    start,
                                  OSQPInt    n) {

  OSQPInt root = start;

  while (2 * root + 1 < n) {
    OSQPInt   child = 2 * root + 1;
    OSQPFloat tmp;

    if ((child + 1 < n) && (a[child + 1] < a[child])) child++;

    if (a[root] <= a[child]) return;

    tmp = a[root]; a[root] = a[child]; a[child] = tmp;
    root = child;
  }
}

/* Heapsort descending: a min-heap repeatedly moves its smallest element to the
   back. Iterative, in place, O(n log n) worst case. */
static void penalty_heap_sort_desc(OSQPFloat* a,
                                   OSQPInt    n) {

  OSQPInt i;

  for (i = n / 2 - 1; i >= 0; i--) penalty_sift_down_min(a, i, n);

  for (i = n - 1; i > 0; i--) {
    OSQPFloat tmp = a[0]; a[0] = a[i]; a[i] = tmp;

    penalty_sift_down_min(a, 0, i);
  }
}

static void penalty_sort_desc(OSQPFloat* a,
                              OSQPInt    n) {

  if (n <= OSQP_PENALTY_SORT_SMALL) penalty_insertion_sort_desc(a, n);
  else                              penalty_heap_sort_desc(a, n);
}

/* The threshold lambda > 0 solving sum_i (|r_i| - lambda)_+ = tau, given |r|
   sorted descending in a[0..n-1]. Exact in finitely many operations: the
   largest k with a[k] > (sum_{j<=k} a[j] - tau)/(k+1) identifies the entries
   that stay above lambda, and lambda follows in closed form.
   The caller guarantees sum_i |r_i| > tau, so such a k exists. */
static OSQPFloat penalty_l1ball_threshold(const OSQPFloat* a,
                                          OSQPInt          n,
                                          OSQPFloat        tau) {

  OSQPFloat cum    = 0.0;
  OSQPFloat lambda = 0.0;
  OSQPInt   k;

  for (k = 0; k < n; k++) {
    OSQPFloat cand;

    cum += a[k];
    cand = (cum - tau) / (OSQPFloat)(k + 1);

    /* NB: tested as !(a[k] > cand) so that a NaN stops the scan */
    if (!(a[k] > cand)) break;

    lambda = cand;
  }

  return lambda;
}

/* Group part of z = vbar + prox_{Phi/rho}(v - vbar).  Runs before the
   elementwise pass, which then leaves these rows alone; that split is what
   lets z alias v, each pass needing v intact on the other's rows.
   sort_tmp holds 2*ptr[ngroups] floats: the staged residuals, then a sortable
   copy of their magnitudes. */
static void group_prox_penalty(OSQPFloat*       zv,
                               const OSQPFloat* vv,
                               const OSQPFloat* lv,
                               const OSQPFloat* uv,
                               OSQPFloat        rho,
                               const OSQPFloat* a1,
                               const OSQPInt*   tv,
                               OSQPInt          default_type,
                               const OSQPInt*   gp,
                               const OSQPInt*   rows,
                               OSQPInt          ngroups,
                               OSQPFloat*       sort_tmp) {

  OSQPInt    g;
  OSQPFloat* res = sort_tmp;                  /* signed residuals */
  OSQPFloat* mag = sort_tmp + gp[ngroups];    /* sortable copy    */

  for (g = 0; g < ngroups; g++) {
    OSQPInt   start = gp[g];
    OSQPInt   stop  = gp[g + 1];
    OSQPFloat tau   = a1[rows[start]] / rho;
    OSQPInt   k;

    /* Stage the residual r = v - Proj_[l,u](v) before writing anything back,
       since z is allowed to alias v, and write the projection into z so that
       the prox below only has to add its slack to it. */
    for (k = start; k < stop; k++) {
      OSQPInt   i    = rows[k];
      OSQPFloat vbar = c_min(c_max(vv[i], lv[i]), uv[i]);

      res[k] = vv[i] - vbar;
      zv[i]  = vbar;
    }

    switch (group_type(tv, default_type, rows, start)) {
      case OSQP_PENALTY_NORM2: {
        /* prox = (1 - tau/||r||_2)_+ * r */
        OSQPFloat nrm   = 0.0;
        OSQPFloat scale = 0.0;

        for (k = start; k < stop; k++) nrm += res[k] * res[k];

        nrm = c_sqrt(nrm);

        if (nrm > tau) scale = 1.0 - tau / nrm;

        for (k = start; k < stop; k++) zv[rows[k]] += scale * res[k];
        break;
      }

      case OSQP_PENALTY_NORMINF: {
        /* prox = r - Proj_{||.||_1 <= tau}(r) by Moreau, which is zero inside
           the ball and sign(r_i)*min(|r_i|, lambda) outside it */
        OSQPFloat sum = 0.0;
        OSQPFloat lambda;
        OSQPInt   n   = stop - start;

        for (k = start; k < stop; k++) {
          mag[k] = c_absval(res[k]);
          sum   += mag[k];
        }

        /* Inside the dual ball the group's whole slack is driven to zero */
        if (sum <= tau) break;

        /* Sorting permutes mag, so the signs stay in res */
        penalty_sort_desc(mag + start, n);

        lambda = penalty_l1ball_threshold(mag + start, n, tau);

        for (k = start; k < stop; k++) {
          OSQPFloat s = c_min(c_absval(res[k]), lambda);

          zv[rows[k]] += (res[k] > 0.0 ? s : -s);
        }
        break;
      }

      default:
        break;   /* Not a group type; validation rules this out */
    }
  }
}

/* sum_g alpha1_g * ||R(z)_G||, with R(z) = z - min(max(z,l),u) */
static OSQPFloat group_penalty_value(const OSQPFloat* zv,
                                     const OSQPFloat* lv,
                                     const OSQPFloat* uv,
                                     const OSQPFloat* a1,
                                     const OSQPInt*   tv,
                                     OSQPInt          default_type,
                                     const OSQPInt*   gp,
                                     const OSQPInt*   rows,
                                     OSQPInt          ngroups) {

  OSQPInt   g;
  OSQPFloat val = 0.0;

  for (g = 0; g < ngroups; g++) {
    OSQPInt   start = gp[g];
    OSQPInt   stop  = gp[g + 1];
    OSQPInt   gt    = group_type(tv, default_type, rows, start);
    OSQPFloat acc   = 0.0;
    OSQPInt   k;

    for (k = start; k < stop; k++) {
      OSQPInt   i    = rows[k];
      OSQPFloat zbar = c_min(c_max(zv[i], lv[i]), uv[i]);
      OSQPFloat s    = zv[i] - zbar;

      if (gt == OSQP_PENALTY_NORM2) acc += s * s;
      else                          acc  = c_max(acc, c_absval(s));
    }

    if (gt == OSQP_PENALTY_NORM2) acc = c_sqrt(acc);

    val += a1[rows[start]] * acc;
  }

  return val;
}

/* sum_g alpha1_g * || (dist(w_i, rec[l_i,u_i]))_{i in G} ||.  Both group
   penalties are positively homogeneous, so a group always grows exactly
   linearly and never blocks a recession direction. */
static OSQPFloat group_penalty_reccone_rate(const OSQPFloat* wv,
                                            const OSQPFloat* lv,
                                            const OSQPFloat* uv,
                                            const OSQPFloat* a1,
                                            const OSQPInt*   tv,
                                            OSQPInt          default_type,
                                            const OSQPInt*   gp,
                                            const OSQPInt*   rows,
                                            OSQPInt          ngroups,
                                            OSQPFloat        infval) {

  OSQPInt   g;
  OSQPFloat rate = 0.0;

  for (g = 0; g < ngroups; g++) {
    OSQPInt   start = gp[g];
    OSQPInt   stop  = gp[g + 1];
    OSQPInt   gt    = group_type(tv, default_type, rows, start);
    OSQPFloat acc   = 0.0;
    OSQPInt   k;

    for (k = start; k < stop; k++) {
      OSQPInt   i = rows[k];
      OSQPFloat dist;

      if ((uv[i] < +infval) && (wv[i] > 0.0))      dist = wv[i];
      else if ((lv[i] > -infval) && (wv[i] < 0.0)) dist = -wv[i];
      else                                         continue;

      if (gt == OSQP_PENALTY_NORM2) acc += dist * dist;
      else                          acc  = c_max(acc, dist);
    }

    if (gt == OSQP_PENALTY_NORM2) acc = c_sqrt(acc);

    rate += a1[rows[start]] * acc;
  }

  return rate;
}

/* A norm's conjugate is the indicator of the dual-norm ball, so a group
   contributes nothing to the dual objective unless it is infeasible.
   in_domain applies the caller's roundoff tolerance. */
static OSQPFloat group_penalty_conj_value(const OSQPFloat* yv,
                                          const OSQPFloat* a1,
                                          const OSQPInt*   tv,
                                          OSQPInt          default_type,
                                          const OSQPInt*   gp,
                                          const OSQPInt*   rows,
                                          OSQPInt          ngroups) {

  OSQPInt g;

  for (g = 0; g < ngroups; g++) {
    OSQPInt   start = gp[g];
    OSQPInt   stop  = gp[g + 1];
    OSQPFloat lim   = a1[rows[start]];
    OSQPFloat acc   = 0.0;
    OSQPInt   k;

    switch (group_type(tv, default_type, rows, start)) {
      case OSQP_PENALTY_NORM2:
        /* dual of ||.||_2 is ||.||_2 */
        for (k = start; k < stop; k++) acc += yv[rows[k]] * yv[rows[k]];
        acc = c_sqrt(acc);
        break;

      case OSQP_PENALTY_NORMINF:
        /* dual of ||.||_inf is ||.||_1 */
        for (k = start; k < stop; k++) acc += c_absval(yv[rows[k]]);
        break;

      default:
        continue;
    }

    if (acc > lim + OSQP_PENALTY_CONJ_TOL * (1.0 + lim)) return OSQP_INFTY;
  }

  return 0.0;
}

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
                                    OSQPVectorf*       sort_tmp) {

  group_prox_penalty(z->values, v->values, l->values, u->values,
                          rho, alpha1->values,
                          type ? type->values : OSQP_NULL, default_type,
                          group_ptr->values, group_rows->values, ngroups,
                          sort_tmp->values);
}

OSQPFloat OSQPVectorf_group_penalty_value(const OSQPVectorf* z,
                                          const OSQPVectorf* l,
                                          const OSQPVectorf* u,
                                          const OSQPVectorf* alpha1,
                                          const OSQPVectori* type,
                                          OSQPInt            default_type,
                                          const OSQPVectori* group_ptr,
                                          const OSQPVectori* group_rows,
                                          OSQPInt            ngroups,
                                          OSQPVectorf*       scratch) {

  (void)scratch; /* Only device backends need reduction storage. */

  return group_penalty_value(z->values, l->values, u->values,
                             alpha1->values,
                             type ? type->values : OSQP_NULL, default_type,
                             group_ptr->values, group_rows->values, ngroups);
}

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
                                                 OSQPVectorf*       scratch) {

  (void)scratch; /* Only device backends need reduction storage. */

  return group_penalty_reccone_rate(w->values, l->values, u->values,
                                    alpha1->values,
                                    type ? type->values : OSQP_NULL,
                                    default_type,
                                    group_ptr->values, group_rows->values,
                                    ngroups, infval);
}

OSQPFloat OSQPVectorf_group_penalty_conj_value(const OSQPVectorf* y,
                                               const OSQPVectorf* alpha1,
                                               const OSQPVectori* type,
                                               OSQPInt            default_type,
                                               const OSQPVectori* group_ptr,
                                               const OSQPVectori* group_rows,
                                               OSQPInt            ngroups,
                                               OSQPVectorf*       scratch) {

  (void)scratch; /* Only device backends need reduction storage. */

  return group_penalty_conj_value(y->values, alpha1->values,
                                  type ? type->values : OSQP_NULL,
                                  default_type,
                                  group_ptr->values, group_rows->values,
                                  ngroups);
}

#if OSQP_EMBEDDED_MODE != 1



/* Replace E on each group by the group's geometric mean, writing the
   correction Ehat/E into corr and 1 on every ungrouped row. */
static void group_equalize_scaling(OSQPFloat*       cv,
                                   const OSQPFloat* Ev,
                                   OSQPInt          length,
                                   const OSQPInt*   gp,
                                   const OSQPInt*   rows,
                                   OSQPInt          ngroups) {

  OSQPInt i, g;

  for (i = 0; i < length; i++) cv[i] = 1.0;

  for (g = 0; g < ngroups; g++) {
    OSQPInt   start = gp[g];
    OSQPInt   stop  = gp[g + 1];
    OSQPInt   n     = stop - start;
    OSQPFloat mean  = 0.0;
    OSQPInt   k;

    /* Averaged in the log so that a long group cannot overflow the product.
       E is strictly positive, being a Ruiz factor passed through
       limit_scaling_vector. */
    for (k = start; k < stop; k++) mean += c_log(Ev[rows[k]]);

    mean = c_exp(mean / (OSQPFloat)n);

    for (k = start; k < stop; k++) cv[rows[k]] = mean / Ev[rows[k]];
  }
}

/* Check the group layout against the per-row types and weights; see
   OSQPVectorf_penalty_groups_check. */
static OSQPInt group_penalty_check(const OSQPFloat* a1,
                                   const OSQPFloat* a2,
                                   const OSQPFloat* d,
                                   const OSQPInt*   tv,
                                   OSQPInt          default_type,
                                   OSQPInt          length,
                                   const OSQPInt*   gp,
                                   const OSQPInt*   rows,
                                   OSQPInt          ngroups) {

  OSQPInt i, g;
  OSQPInt flags       = 0;
  OSQPInt ngroup_rows = 0;

  /* Count the rows that declare a group type. Comparing this against the rows
     the groups actually contain is what rules out a group type outside any
     group, without needing the inverse of the membership map. */
  for (i = 0; i < length; i++) {
    OSQPInt t = tv ? tv[i] : default_type;

    if ((t == OSQP_PENALTY_NORM2) || (t == OSQP_PENALTY_NORMINF)) ngroup_rows++;
  }

  for (g = 0; g < ngroups; g++) {
    OSQPInt   start = gp[g];
    OSQPInt   stop  = gp[g + 1];
    OSQPInt   gt;
    OSQPFloat gw;
    OSQPInt   k;

    /* An empty group has no type or weight to read, so nothing below applies */
    if (start >= stop) {
      flags |= OSQP_PENALTY_ERR_GROUP_TYPE;
      continue;
    }

    gt = tv ? tv[rows[start]] : default_type;
    gw = a1[rows[start]];

    if ((gt != OSQP_PENALTY_NORM2) && (gt != OSQP_PENALTY_NORMINF)) {
      flags |= OSQP_PENALTY_ERR_GROUP_TYPE;
      continue;
    }

    /* NB: tested as !(gw > 0) rather than (gw <= 0) so that NaN is rejected */
    if (!(gw > 0.0) || (gw >= OSQP_INFTY)) flags |= OSQP_PENALTY_ERR_GROUP_WEIGHT;

    for (k = start; k < stop; k++) {
      OSQPInt i_row = rows[k];

      if ((tv ? tv[i_row] : default_type) != gt) flags |= OSQP_PENALTY_ERR_GROUP_MIXED;
      if (a1[i_row] != gw)                       flags |= OSQP_PENALTY_ERR_GROUP_MIXED;

      /* Neither norm uses the remaining slots; a nonzero there is a weight the
         caller believes is active, so it is rejected rather than ignored */
      if ((a2[i_row] != 0.0) || (d[i_row] != 0.0)) flags |= OSQP_PENALTY_ERR_GROUP_WEIGHT;
    }

    ngroup_rows -= (stop - start);
  }

  /* Nonzero either way: a group type on an ungrouped row, or a grouped row
     whose type is separable. The per-group loop above catches the latter, so
     what survives here is the former. */
  if (ngroup_rows != 0) flags |= OSQP_PENALTY_ERR_GROUP_TYPE;

  return flags;
}

void OSQPVectorf_group_equalize_scaling(OSQPVectorf*       corr,
                                        const OSQPVectorf* E,
                                        const OSQPVectori* group_ptr,
                                        const OSQPVectori* group_rows,
                                        OSQPInt            ngroups) {

  group_equalize_scaling(corr->values, E->values, corr->length,
                         group_ptr->values, group_rows->values, ngroups);
}

void OSQPVectorf_ew_scale_penalty(OSQPVectorf*       alpha1,
                                  OSQPVectorf*       alpha2,
                                  OSQPVectorf*       delta,
                                  const OSQPVectori* type,
                                  OSQPInt            default_type,
                                  OSQPFloat          c,
                                  const OSQPVectorf* E,
                                  OSQPInt            invert) {

  OSQPInt    i;
  OSQPInt    length = alpha1->length;
  OSQPFloat* a1     = alpha1->values;
  OSQPFloat* a2     = alpha2->values;
  OSQPFloat* d      = delta->values;
  OSQPFloat* Ev     = E ? E->values : OSQP_NULL;
  OSQPInt*   tv     = type ? type->values : OSQP_NULL;

  for (i = 0; i < length; i++) {
    OSQPFloat e = Ev ? Ev[i] : 1.0;
    OSQPFloat f1, f2, fd;

    switch (tv ? tv[i] : default_type) {
      case OSQP_PENALTY_L1L2:
        f1 = c / e;       f2 = c / (e * e); fd = 1.0;
        break;

      case OSQP_PENALTY_HUBER:
        f1 = c / (e * e); f2 = 1.0;         fd = e;
        break;

      case OSQP_PENALTY_NORM2:
      case OSQP_PENALTY_NORMINF:
        /* Both are 1-homogeneous, so c*alpha*||s/E|| = (c*alpha/E)*||s||.
           This needs E constant on the group, which scale_data enforces. */
        f1 = c / e;       f2 = 1.0;         fd = 1.0;
        break;

      default:
        continue;   // Hard row, parameters unused
    }

    if (invert) {
      a1[i] /= f1;  a2[i] /= f2;  d[i] /= fd;
    }
    else {
      a1[i] *= f1;  a2[i] *= f2;  d[i] *= fd;
    }
  }
}
OSQPInt OSQPVectorf_penalty_params_check(const OSQPVectorf* alpha1,
                                         const OSQPVectorf* alpha2,
                                         const OSQPVectorf* delta,
                                         const OSQPVectori* type,
                                         OSQPInt            default_type,
                                         OSQPVectori*       scratch) {

  OSQPInt    i;
  OSQPInt    flags  = 0;
  OSQPInt    length = alpha1->length;
  OSQPFloat* a1     = alpha1->values;
  OSQPFloat* a2     = alpha2->values;
  OSQPFloat* d      = delta->values;
  OSQPInt*   tv     = type ? type->values : OSQP_NULL;

  (void)scratch; /* Only device backends need reduction storage. */

  /* NB: tested as !(x >= 0) rather than (x < 0) so that NaN is rejected too */
  for (i = 0; i < length; i++) {
    switch (tv ? tv[i] : default_type) {
      case OSQP_PENALTY_L1L2:
        if (!(a1[i] >= 0.0) || !(a2[i] >= 0.0))          flags |= OSQP_PENALTY_ERR_NEGATIVE;
        if ((a1[i] >= OSQP_INFTY) || (a2[i] >= OSQP_INFTY)) flags |= OSQP_PENALTY_ERR_INFINITE;
        break;

      case OSQP_PENALTY_HUBER:
        if (!(a1[i] > 0.0)) flags |= OSQP_PENALTY_ERR_HUBER_W;
        if (!(d[i]  > 0.0)) flags |= OSQP_PENALTY_ERR_HUBER_D;
        if ((a1[i] >= OSQP_INFTY) || (d[i] >= OSQP_INFTY)) flags |= OSQP_PENALTY_ERR_INFINITE;
        break;

      default:
        break;
    }
  }

  return flags;
}

OSQPInt OSQPVectorf_penalty_groups_check(const OSQPVectorf* alpha1,
                                         const OSQPVectorf* alpha2,
                                         const OSQPVectorf* delta,
                                         const OSQPVectori* type,
                                         OSQPInt            default_type,
                                         const OSQPVectori* group_ptr,
                                         const OSQPVectori* group_rows,
                                         OSQPInt            ngroups,
                                         OSQPVectori*       scratch) {

  (void)scratch; /* Only device backends need reduction storage. */

  return group_penalty_check(alpha1->values, alpha2->values, delta->values,
                             type ? type->values : OSQP_NULL, default_type,
                             alpha1->length,
                             ngroups ? group_ptr->values  : OSQP_NULL,
                             ngroups ? group_rows->values : OSQP_NULL,
                             ngroups);
}

#endif /* OSQP_EMBEDDED_MODE != 1 */
