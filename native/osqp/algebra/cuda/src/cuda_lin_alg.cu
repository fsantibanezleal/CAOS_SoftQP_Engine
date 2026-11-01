/**
 *  Copyright (c) 2019-2021 ETH Zurich, Automatic Control Lab,
 *  Michel Schubiger, Goran Banjac.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <float.h>
#include <math.h>
#include "cuda_lin_alg.h"
#include "cuda_configure.h"
#include "cuda_handler.h"
#include "cuda_memory.h"
#include "cuda_wrapper.h"
#include "helper_cuda.h"    /* --> checkCudaErrors */

#include "csr_type.h"
#include "algebra_vector.h"
#include "glob_opts.h"

#include <thrust/gather.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

extern CUDA_Handle_t *CUDA_handle;


/*******************************************************************************
 *                              GPU Kernels                                    *
 *******************************************************************************/
 __global__ void vec_round_kernel(OSQPFloat* a,
                                  OSQPFloat  tol,
                                  OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    if(c_absval(a[i]) < tol) {
      a[i] = (OSQPFloat)0.0;
    }
  }
}

 __global__ void vec_set_sc_kernel(OSQPFloat* a,
                                   OSQPFloat  sc,
                                   OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    a[i] = sc;
  }
}

__global__ void vec_set_sc_cond_kernel(OSQPFloat*     a,
                                       const OSQPInt* test,
                                       OSQPFloat      sc_if_neg,
                                       OSQPFloat      sc_if_zero,
                                       OSQPFloat      sc_if_pos,
                                       OSQPInt        n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    if (test[i] == 0)      a[i] = sc_if_zero;
    else if (test[i] > 0)  a[i] = sc_if_pos;
    else                   a[i] = sc_if_neg;
  }
}

__global__ void vec_scale_penalty_kernel(OSQPFloat*       a1,
                                         OSQPFloat*       a2,
                                         OSQPFloat*       d,
                                         const OSQPInt*   type,
                                         OSQPInt          default_type,
                                         OSQPFloat        c,
                                         const OSQPFloat* E,
                                         OSQPInt          invert,
                                         OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    OSQPFloat e = E ? E[i] : 1.0;
    OSQPFloat f1, f2, fd;

    switch (type ? type[i] : default_type) {
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

__global__ void vec_penalty_check_kernel(const OSQPFloat* a1,
                                         const OSQPFloat* a2,
                                         const OSQPFloat* d,
                                         const OSQPInt*   type,
                                         OSQPInt          default_type,
                                         OSQPInt*         res,
                                         OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    switch (type ? type[i] : default_type) {
      case OSQP_PENALTY_L1L2:
        if (!(a1[i] >= 0.0) || !(a2[i] >= 0.0))            atomicOr(res, OSQP_PENALTY_ERR_NEGATIVE);
        if ((a1[i] >= OSQP_INFTY) || (a2[i] >= OSQP_INFTY)) atomicOr(res, OSQP_PENALTY_ERR_INFINITE);
        break;

      case OSQP_PENALTY_HUBER:
        if (!(a1[i] > 0.0)) atomicOr(res, OSQP_PENALTY_ERR_HUBER_W);
        if (!(d[i]  > 0.0)) atomicOr(res, OSQP_PENALTY_ERR_HUBER_D);
        if ((a1[i] >= OSQP_INFTY) || (d[i] >= OSQP_INFTY)) atomicOr(res, OSQP_PENALTY_ERR_INFINITE);
        break;

      default:
        break;
    }
  }
}

/* prox_{phi/rho}(r) for a single row */
__device__ static OSQPFloat prox_penalty_row(OSQPFloat r,
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
__device__ static OSQPFloat penalty_value_row(OSQPFloat s,
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
__device__ static OSQPInt penalty_conj_in_domain(OSQPFloat ay,
                                                 OSQPFloat lim) {

  return ay <= lim + OSQP_PENALTY_CONJ_TOL * (1.0 + lim);
}

/* phi*(y) for a single row, OSQP_INFTY outside the domain */
__device__ static OSQPFloat penalty_conj_row(OSQPFloat y,
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

__device__ static OSQPFloat penalty_reccone_rate_row(OSQPFloat w,
                                                      OSQPFloat l,
                                                      OSQPFloat u,
                                                      OSQPInt   type,
                                                      OSQPFloat a1,
                                                      OSQPFloat a2,
                                                      OSQPFloat d,
                                                      OSQPFloat infval,
                                                      OSQPFloat tol) {
  OSQPFloat dist;

  if ((u < +infval) && (w > 0.0))      dist = w;
  else if ((l > -infval) && (w < 0.0)) dist = -w;
  else                                 return 0.0;

  switch (type) {
    case OSQP_PENALTY_L1L2:
      return a2 > 0.0 ? (dist > tol ? OSQP_INFTY : 0.0) : a1 * dist;

    case OSQP_PENALTY_HUBER:
      return (a1 * d) * dist;

    case OSQP_PENALTY_NORM2:
    case OSQP_PENALTY_NORMINF:
      /* Charged by the group kernel, which needs the whole group at once.
         Both norms are positively homogeneous, so a group never blocks. */
      return 0.0;

    default:
      return dist > tol ? OSQP_INFTY : 0.0;
  }
}

/* TYPE < 0 reads the per-row type vector, otherwise every row is TYPE and the
   switches above fold away at compile time */
template<OSQPInt TYPE>
__global__ void vec_prox_penalty_kernel(      OSQPFloat* z,
                                        const OSQPFloat* v,
                                        const OSQPFloat* l,
                                        const OSQPFloat* u,
                                        const OSQPFloat* rho_vec,
                                              OSQPFloat  rho,
                                        const OSQPFloat* a1,
                                        const OSQPFloat* a2,
                                        const OSQPFloat* d,
                                        const OSQPInt*   type,
                                              OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    OSQPInt   t = TYPE < 0 ? type[i] : TYPE;
    OSQPFloat vbar;

    /* Left to group_prox_penalty_kernel, which runs first and has already
       written this row; overwriting it would also break z aliasing v */
    if ((t == OSQP_PENALTY_NORM2) || (t == OSQP_PENALTY_NORMINF)) continue;

    vbar = c_min(c_max(v[i], l[i]), u[i]);

    z[i] = vbar + prox_penalty_row(v[i] - vbar, rho_vec ? rho_vec[i] : rho,
                                   t, a1[i], a2[i], d[i]);
  }
}

template<OSQPInt TYPE>
__global__ void vec_penalty_value_kernel(const OSQPFloat* z,
                                         const OSQPFloat* l,
                                         const OSQPFloat* u,
                                         const OSQPFloat* a1,
                                         const OSQPFloat* a2,
                                         const OSQPFloat* d,
                                         const OSQPInt*   type,
                                               OSQPFloat* res,
                                               OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  OSQPFloat res_kernel = 0.0;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    OSQPFloat zbar = c_min(c_max(z[i], l[i]), u[i]);

    res_kernel += penalty_value_row(z[i] - zbar, TYPE < 0 ? type[i] : TYPE,
                                    a1[i], a2[i], d[i]);
  }
  atomicAdd(res, res_kernel);
}

template<OSQPInt TYPE>
__global__ void vec_penalty_conj_value_kernel(const OSQPFloat* y,
                                              const OSQPFloat* a1,
                                              const OSQPFloat* a2,
                                              const OSQPFloat* d,
                                              const OSQPInt*   type,
                                                    OSQPFloat* res,
                                                    OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  OSQPFloat res_kernel = 0.0;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    res_kernel += penalty_conj_row(y[i], TYPE < 0 ? type[i] : TYPE,
                                   a1[i], a2[i], d[i]);
  }
  atomicAdd(res, res_kernel);
}

template<OSQPInt TYPE>
__global__ void vec_penalty_reccone_rate_kernel(const OSQPFloat* w,
                                                const OSQPFloat* l,
                                                const OSQPFloat* u,
                                                const OSQPFloat* a1,
                                                const OSQPFloat* a2,
                                                const OSQPFloat* d,
                                                const OSQPInt*   type,
                                                OSQPFloat        infval,
                                                OSQPFloat        tol,
                                                OSQPFloat*       res,
                                                OSQPInt          n) {
  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;
  OSQPFloat res_kernel = 0.0;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    res_kernel += penalty_reccone_rate_row(w[i], l[i], u[i],
                                           TYPE < 0 ? type[i] : TYPE,
                                           a1[i], a2[i], d[i], infval, tol);
  }
  atomicAdd(res, res_kernel);
}

__global__ void vec_prod_pos_kernel(const OSQPFloat* a,
                                    const OSQPFloat* b,
                                          OSQPFloat* res,
                                          OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  OSQPFloat res_kernel = 0.0;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    res_kernel += a[i] * c_max(b[i], 0.0);
  }
  atomicAdd(res, res_kernel);
}

__global__ void vec_prod_neg_kernel(const OSQPFloat* a,
                                    const OSQPFloat* b,
                                          OSQPFloat* res,
                                          OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  OSQPFloat res_kernel = 0.0;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    res_kernel += a[i] * c_min(b[i], 0.0);
  }
  atomicAdd(res, res_kernel);
}

/*******************************************************************************
 * Non-separable penalty groups                                                *
 *                                                                             *
 * One block per group. A group's work is a handful of block-wide reductions    *
 * over its rows, so the whole pass stays on the device and the group vectors   *
 * are never staged on the host.                                                *
 *******************************************************************************/

/* Threads per group block. A power of two, so the reduction tree below is a
   plain halving, and well under THREADS_PER_BLOCK because a block is spent per
   group rather than per element: groups are typically small. */
#define GROUP_THREADS (256)

/* Block-wide sum, broadcast to every thread. s is shared scratch of
   GROUP_THREADS floats, reused across calls within a kernel; the leading
   barrier is what makes that reuse safe. */
__device__ static OSQPFloat group_reduce_sum(OSQPFloat* s,
                                             OSQPFloat  val) {

  OSQPInt tid = threadIdx.x;

  __syncthreads();
  s[tid] = val;
  __syncthreads();

  for (OSQPInt off = GROUP_THREADS / 2; off > 0; off >>= 1) {
    if (tid < off) s[tid] += s[tid + off];
    __syncthreads();
  }

  return s[0];
}

/* Block-wide maximum, broadcast to every thread */
__device__ static OSQPFloat group_reduce_max(OSQPFloat* s,
                                             OSQPFloat  val) {

  OSQPInt tid = threadIdx.x;

  __syncthreads();
  s[tid] = val;
  __syncthreads();

  for (OSQPInt off = GROUP_THREADS / 2; off > 0; off >>= 1) {
    if (tid < off) s[tid] = c_max(s[tid], s[tid + off]);
    __syncthreads();
  }

  return s[0];
}

/* Type of a group, which validation keeps equal across its rows */
__device__ static OSQPInt group_type_of(const OSQPInt* type,
                                        OSQPInt        default_type,
                                        const OSQPInt* rows,
                                        OSQPInt        start) {

  return type ? type[rows[start]] : default_type;
}

/* The threshold lambda > 0 solving sum_i (|r_i| - lambda)_+ = tau.
 *
 * Michelot's algorithm rather than the host's sort: starting from lambda = 0,
 * each sweep recomputes lambda from the rows still above it, which is exactly
 * the closed form once the active set stops shrinking. It is exact and
 * finitely terminating like the sorted scan, but its sweeps are block-wide
 * reductions, which is what the device has cheaply and a sort is not. The
 * caller guarantees sum_i |r_i| > tau, so lambda > 0 exists.
 *
 * Every thread reduces the same values and so leaves with the same lambda and
 * the same loop decision; the loop itself never diverges.
 */
__device__ static OSQPFloat group_l1ball_threshold(OSQPFloat*       s,
                                                   const OSQPFloat* res,
                                                   OSQPInt          start,
                                                   OSQPInt          stop,
                                                   OSQPFloat        tau) {

  OSQPInt   n      = stop - start;
  OSQPFloat lambda = 0.0;
  OSQPInt   sweep;

  /* The active set strictly shrinks on every sweep that moves lambda, so n
     sweeps is a hard bound and the counter is a guard, not the exit */
  for (sweep = 0; sweep <= n; sweep++) {
    OSQPFloat sum = 0.0;
    OSQPFloat cnt = 0.0;
    OSQPFloat cand;

    for (OSQPInt k = start + threadIdx.x; k < stop; k += GROUP_THREADS) {
      OSQPFloat a = c_absval(res[k]);

      if (a > lambda) { sum += a; cnt += 1.0; }
    }

    sum = group_reduce_sum(s, sum);
    cnt = group_reduce_sum(s, cnt);

    if (cnt <= 0.0) break;

    cand = (sum - tau) / cnt;

    /* lambda only ever increases; anything else is the fixed point */
    if (!(cand > lambda)) break;

    lambda = cand;
  }

  return lambda;
}

__global__ void group_prox_penalty_kernel(      OSQPFloat* z,
                                          const OSQPFloat* v,
                                          const OSQPFloat* l,
                                          const OSQPFloat* u,
                                                OSQPFloat  rho,
                                          const OSQPFloat* a1,
                                          const OSQPInt*   type,
                                          const OSQPInt*   gp,
                                          const OSQPInt*   rows,
                                                OSQPInt    default_type,
                                                OSQPFloat* res) {

  __shared__ OSQPFloat s[GROUP_THREADS];

  OSQPInt   g     = blockIdx.x;
  OSQPInt   start = gp[g];
  OSQPInt   stop  = gp[g + 1];
  OSQPFloat tau   = a1[rows[start]] / rho;

  /* Stage the residual r = v - Proj_[l,u](v) and leave the projection in z, so
     that the prox below only has to add its slack. Each thread reads v and
     writes z at the same index, so z may alias v. */
  for (OSQPInt k = start + threadIdx.x; k < stop; k += GROUP_THREADS) {
    OSQPInt   i    = rows[k];
    OSQPFloat vbar = c_min(c_max(v[i], l[i]), u[i]);

    res[k] = v[i] - vbar;
    z[i]   = vbar;
  }

  __syncthreads();

  /* The group type is the same for every thread of the block, so the switch
     below never diverges and the barriers inside the reductions are safe */
  switch (group_type_of(type, default_type, rows, start)) {
    case OSQP_PENALTY_NORM2: {
      /* prox = (1 - tau/||r||_2)_+ * r */
      OSQPFloat acc   = 0.0;
      OSQPFloat nrm;
      OSQPFloat scale = 0.0;

      for (OSQPInt k = start + threadIdx.x; k < stop; k += GROUP_THREADS)
        acc += res[k] * res[k];

      nrm = c_sqrt(group_reduce_sum(s, acc));

      if (nrm > tau) scale = 1.0 - tau / nrm;

      for (OSQPInt k = start + threadIdx.x; k < stop; k += GROUP_THREADS)
        z[rows[k]] += scale * res[k];
      break;
    }

    case OSQP_PENALTY_NORMINF: {
      /* prox = r - Proj_{||.||_1 <= tau}(r) by Moreau, which is zero inside
         the ball and sign(r_i)*min(|r_i|, lambda) outside it */
      OSQPFloat acc = 0.0;
      OSQPFloat sum;
      OSQPFloat lambda;

      for (OSQPInt k = start + threadIdx.x; k < stop; k += GROUP_THREADS)
        acc += c_absval(res[k]);

      sum = group_reduce_sum(s, acc);

      /* Inside the dual ball the group's whole slack is driven to zero, and z
         keeps the projection already written above */
      if (sum <= tau) break;

      lambda = group_l1ball_threshold(s, res, start, stop, tau);

      for (OSQPInt k = start + threadIdx.x; k < stop; k += GROUP_THREADS) {
        OSQPFloat r = res[k];
        OSQPFloat t = c_min(c_absval(r), lambda);

        z[rows[k]] += (r > 0.0 ? t : -t);
      }
      break;
    }

    default:
      break;   /* Not a group type; validation rules this out */
  }
}

__global__ void group_penalty_value_kernel(const OSQPFloat* z,
                                           const OSQPFloat* l,
                                           const OSQPFloat* u,
                                           const OSQPFloat* a1,
                                           const OSQPInt*   type,
                                           const OSQPInt*   gp,
                                           const OSQPInt*   rows,
                                                 OSQPInt    default_type,
                                                 OSQPFloat* res) {

  __shared__ OSQPFloat s[GROUP_THREADS];

  OSQPInt   g     = blockIdx.x;
  OSQPInt   start = gp[g];
  OSQPInt   stop  = gp[g + 1];
  OSQPInt   gt    = group_type_of(type, default_type, rows, start);
  OSQPFloat acc   = 0.0;
  OSQPFloat val;

  for (OSQPInt k = start + threadIdx.x; k < stop; k += GROUP_THREADS) {
    OSQPInt   i    = rows[k];
    OSQPFloat zbar = c_min(c_max(z[i], l[i]), u[i]);
    OSQPFloat sl   = z[i] - zbar;

    if (gt == OSQP_PENALTY_NORM2) acc += sl * sl;
    else                          acc  = c_max(acc, c_absval(sl));
  }

  val = (gt == OSQP_PENALTY_NORM2) ? c_sqrt(group_reduce_sum(s, acc))
                                   : group_reduce_max(s, acc);

  if (threadIdx.x == 0) atomicAdd(res, a1[rows[start]] * val);
}

__global__ void group_penalty_reccone_rate_kernel(const OSQPFloat* w,
                                                  const OSQPFloat* l,
                                                  const OSQPFloat* u,
                                                  const OSQPFloat* a1,
                                                  const OSQPInt*   type,
                                                  const OSQPInt*   gp,
                                                  const OSQPInt*   rows,
                                                        OSQPInt    default_type,
                                                        OSQPFloat  infval,
                                                        OSQPFloat* res) {

  __shared__ OSQPFloat s[GROUP_THREADS];

  OSQPInt   g     = blockIdx.x;
  OSQPInt   start = gp[g];
  OSQPInt   stop  = gp[g + 1];
  OSQPInt   gt    = group_type_of(type, default_type, rows, start);
  OSQPFloat acc   = 0.0;
  OSQPFloat val;

  for (OSQPInt k = start + threadIdx.x; k < stop; k += GROUP_THREADS) {
    OSQPInt   i = rows[k];
    OSQPFloat dist;

    if ((u[i] < +infval) && (w[i] > 0.0))      dist = w[i];
    else if ((l[i] > -infval) && (w[i] < 0.0)) dist = -w[i];
    else                                       continue;

    if (gt == OSQP_PENALTY_NORM2) acc += dist * dist;
    else                          acc  = c_max(acc, dist);
  }

  val = (gt == OSQP_PENALTY_NORM2) ? c_sqrt(group_reduce_sum(s, acc))
                                   : group_reduce_max(s, acc);

  if (threadIdx.x == 0) atomicAdd(res, a1[rows[start]] * val);
}

__global__ void group_penalty_conj_value_kernel(const OSQPFloat* y,
                                                const OSQPFloat* a1,
                                                const OSQPInt*   type,
                                                const OSQPInt*   gp,
                                                const OSQPInt*   rows,
                                                      OSQPInt    default_type,
                                                      OSQPFloat* res) {

  __shared__ OSQPFloat s[GROUP_THREADS];

  OSQPInt   g     = blockIdx.x;
  OSQPInt   start = gp[g];
  OSQPInt   stop  = gp[g + 1];
  OSQPInt   gt    = group_type_of(type, default_type, rows, start);
  OSQPFloat lim   = a1[rows[start]];
  OSQPFloat acc   = 0.0;
  OSQPFloat nrm;

  /* A norm's conjugate is the indicator of the dual-norm ball: ||.||_2 for
     NORM2 and ||.||_1 for NORMINF */
  for (OSQPInt k = start + threadIdx.x; k < stop; k += GROUP_THREADS) {
    OSQPFloat yi = y[rows[k]];

    if (gt == OSQP_PENALTY_NORM2) acc += yi * yi;
    else                          acc += c_absval(yi);
  }

  nrm = group_reduce_sum(s, acc);

  if (gt == OSQP_PENALTY_NORM2) nrm = c_sqrt(nrm);

  if (threadIdx.x == 0) {
    if (nrm > lim + OSQP_PENALTY_CONJ_TOL * (1.0 + lim)) atomicAdd(res, OSQP_INFTY);
  }
}

__global__ void group_equalize_scaling_kernel(      OSQPFloat* corr,
                                              const OSQPFloat* E,
                                              const OSQPInt*   gp,
                                              const OSQPInt*   rows) {

  __shared__ OSQPFloat s[GROUP_THREADS];

  OSQPInt   g     = blockIdx.x;
  OSQPInt   start = gp[g];
  OSQPInt   stop  = gp[g + 1];
  OSQPFloat acc   = 0.0;
  OSQPFloat mean;

  /* Averaged in the log so that a long group cannot overflow the product.
     E is strictly positive, being a Ruiz factor passed through
     limit_scaling_vector. */
  for (OSQPInt k = start + threadIdx.x; k < stop; k += GROUP_THREADS)
    acc += c_log(E[rows[k]]);

  mean = c_exp(group_reduce_sum(s, acc) / (OSQPFloat)(stop - start));

  for (OSQPInt k = start + threadIdx.x; k < stop; k += GROUP_THREADS)
    corr[rows[k]] = mean / E[rows[k]];
}

/* Count the rows declaring a group type. Comparing this against the rows the
   groups actually contain, which the caller knows as ngrouped, is what rules
   out a group type outside any group, without needing the inverse of the
   membership map.

   NB: the counter is unsigned long long rather than OSQPInt because atomicAdd
   has no overload for a signed 64-bit integer, which is what OSQPInt is under
   OSQP_USE_LONG. */
__global__ void group_type_count_kernel(const OSQPInt*      type,
                                              OSQPInt       default_type,
                                              unsigned long long* count,
                                              OSQPInt       n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  unsigned long long local = 0;

  for (OSQPInt i = idx; i < n; i += grid_size) {
    OSQPInt t = type ? type[i] : default_type;

    if ((t == OSQP_PENALTY_NORM2) || (t == OSQP_PENALTY_NORMINF)) local++;
  }

  atomicAdd(count, local);
}

__global__ void group_penalty_check_kernel(const OSQPFloat* a1,
                                           const OSQPFloat* a2,
                                           const OSQPFloat* d,
                                           const OSQPInt*   type,
                                           const OSQPInt*   gp,
                                           const OSQPInt*   rows,
                                                 OSQPInt    default_type,
                                                 OSQPInt*   flags) {

  OSQPInt   g     = blockIdx.x;
  OSQPInt   start = gp[g];
  OSQPInt   stop  = gp[g + 1];
  OSQPInt   gt;
  OSQPFloat gw;

  /* An empty group has no type or weight to read, so nothing below applies */
  if (start >= stop) {
    if (threadIdx.x == 0) atomicOr(flags, OSQP_PENALTY_ERR_GROUP_TYPE);
    return;
  }

  gt = type ? type[rows[start]] : default_type;
  gw = a1[rows[start]];

  if ((gt != OSQP_PENALTY_NORM2) && (gt != OSQP_PENALTY_NORMINF)) {
    if (threadIdx.x == 0) atomicOr(flags, OSQP_PENALTY_ERR_GROUP_TYPE);
    return;
  }

  /* NB: tested as !(gw > 0) rather than (gw <= 0) so that NaN is rejected */
  if (threadIdx.x == 0) {
    if (!(gw > 0.0) || (gw >= OSQP_INFTY)) atomicOr(flags, OSQP_PENALTY_ERR_GROUP_WEIGHT);
  }

  for (OSQPInt k = start + threadIdx.x; k < stop; k += GROUP_THREADS) {
    OSQPInt i = rows[k];

    if ((type ? type[i] : default_type) != gt) atomicOr(flags, OSQP_PENALTY_ERR_GROUP_MIXED);
    if (a1[i] != gw)                           atomicOr(flags, OSQP_PENALTY_ERR_GROUP_MIXED);

    /* Neither norm uses the remaining slots; a nonzero there is a weight the
       caller believes is active, so it is rejected rather than ignored */
    if ((a2[i] != 0.0) || (d[i] != 0.0)) atomicOr(flags, OSQP_PENALTY_ERR_GROUP_WEIGHT);
  }
}

__global__ void vec_ew_prod_kernel(OSQPFloat*       c,
                                   const OSQPFloat* a,
                                   const OSQPFloat* b,
                                   OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
#ifdef OSQP_USE_FLOAT
    c[i] = __fmul_rn(a[i], b[i]);
#else
    c[i] = __dmul_rn(a[i], b[i]);
#endif
  }
}

__global__ void vec_eq_kernel(const OSQPFloat* a,
                              const OSQPFloat* b,
                                    OSQPFloat  tol,
                                    OSQPInt*   res,
                                    OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  *res = 1;

  for(OSQPInt i = idx; i < n; i += grid_size) {
      if (c_absval(a[i] - b[i]) > tol) {
          *res = 0;
          break;
      }
  }
}

__global__ void vec_int_eq_kernel(const OSQPInt* a,
                                  const OSQPInt* b,
                                        OSQPInt* res,
                                        OSQPInt  n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  *res = 1;

  for(OSQPInt i = idx; i < n; i += grid_size) {
      if (a[i] != b[i]) {
          *res = 0;
          break;
      }
  }
}

__global__ void vec_leq_kernel(const OSQPFloat* l,
                               const OSQPFloat* u,
                                     OSQPInt*   res,
                                     OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    if (l[i] > u[i]) atomicAnd(res, 0);
  }
}

__global__ void vec_bound_kernel(OSQPFloat*       x,
                                 const OSQPFloat* z,
                                 const OSQPFloat* l,
                                 const OSQPFloat* u,
                                 OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    x[i] = c_min(c_max(z[i], l[i]), u[i]);
  }
}

__global__ void vec_project_polar_reccone_kernel(OSQPFloat*       y,
                                                 const OSQPFloat* l,
                                                 const OSQPFloat* u,
                                                 const OSQPInt*   type,
                                                 OSQPInt          default_type,
                                                 OSQPFloat        infval,
                                                 OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    if ((type ? type[i] : default_type) != OSQP_PENALTY_NONE) {
      /* Soft row: free, so the polar of its recession cone is {0} */
      y[i] = 0.0;
    }
    else if (u[i] > +infval) {
      if (l[i] < -infval) {
        /* Both bounds infinite */
        y[i] = 0.0;
      }
      else {
        /* Only upper bound infinite */
        y[i] = c_min(y[i], 0.0);
      }
    }
    else if (l[i] < -infval) {
      /* Only lower bound infinite */
      y[i] = c_max(y[i], 0.0);
    }
  }
}

__global__ void vec_in_reccone_kernel(const OSQPFloat* y,
                                      const OSQPFloat* l,
                                      const OSQPFloat* u,
                                      const OSQPFloat* alpha2,
                                      const OSQPInt*   type,
                                      OSQPInt          default_type,
                                      OSQPFloat        infval,
                                      OSQPFloat        tol,
                                      OSQPInt*         res,
                                      OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    OSQPInt row_type = type ? type[i] : default_type;

    /* Only a quadratic L1L2 penalty constrains recession directions. */
    if (row_type != OSQP_PENALTY_NONE &&
        (row_type != OSQP_PENALTY_L1L2 || !alpha2 || alpha2[i] <= 0.0)) continue;

    if ( (u[i] < +infval && y[i] > +tol) ||
         (l[i] > -infval && y[i] < -tol) )
      atomicAnd(res, 0);
  }
}

__global__ void vec_reciprocal_kernel(OSQPFloat*       b,
                                      const OSQPFloat* a,
                                      OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
#ifdef OSQP_USE_FLOAT
    b[i] = __frcp_rn(a[i]);
#else
    b[i] = __drcp_rn(a[i]);
#endif
  }
}

__global__ void vec_sqrt_kernel(OSQPFloat* a,
                                OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
#ifdef OSQP_USE_FLOAT
    a[i] = __fsqrt_rn(a[i]);
#else
    a[i] = __dsqrt_rn(a[i]);
#endif
  }
}

__global__ void vec_max_kernel(OSQPFloat*       c,
                               const OSQPFloat* a,
                               const OSQPFloat* b,
                               OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    c[i] = c_max(a[i], b[i]);
  }
}

__global__ void vec_min_kernel(OSQPFloat*       c,
                               const OSQPFloat* a,
                               const OSQPFloat* b,
                               OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    c[i] = c_min(a[i], b[i]);
  }
}

__global__ void vec_bounds_type_kernel(OSQPInt*         iseq,
                                       const OSQPFloat* l,
                                       const OSQPFloat* u,
                                       const OSQPInt*   type,
                                       OSQPInt          default_type,
                                       OSQPFloat        infval,
                                       OSQPFloat        tol,
                                       OSQPInt*         has_changed,
                                       OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    OSQPInt t = type ? type[i] : default_type;

    /* A soft row is never an equality however tight its bounds are */
    if ((u[i] - l[i] < tol) && (t == OSQP_PENALTY_NONE)) {
      /* Equality constraints */
      if (iseq[i] != 1) {
        iseq[i] = 1;
        atomicOr(has_changed, 1);
      }
    }
    else if ( (l[i] < -infval) && (u[i] > infval) &&
              (t != OSQP_PENALTY_NORM2) && (t != OSQP_PENALTY_NORMINF) ) {
      /* Loose bounds. A grouped row is excluded: a group prox takes a scalar
         rho, so every row of a group must land on the same one, and keeping
         that true after any later bound update is not something a setup-time
         check could do. Such a row is inert anyway, its bound residual being
         identically zero. */
      if (iseq[i] != -1) {
        iseq[i] = -1;
        atomicOr(has_changed, 1);
      }
    }
    else {
      /* Inequality constraints */
      if (iseq[i] != 0) {
        iseq[i] = 0;
        atomicOr(has_changed, 1);
      }
    }
  }
}

__global__ void vec_set_sc_if_lt_kernel(OSQPFloat*       x,
                                        const OSQPFloat* z,
                                        OSQPFloat        testval,
                                        OSQPFloat        newval,
                                        OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    x[i] = z[i] < testval ? newval : z[i];
  }
}

__global__ void vec_set_sc_if_gt_kernel(OSQPFloat*       x,
                                        const OSQPFloat* z,
                                        OSQPFloat        testval,
                                        OSQPFloat        newval,
                                        OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    x[i] = z[i] > testval ? newval : z[i];
  }
}

__global__ void mat_lmult_diag_kernel(const OSQPInt*   row_ind,
                                      const OSQPFloat* diag,
                                            OSQPFloat* data,
                                            OSQPInt    nnz) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < nnz; i += grid_size) {
    OSQPInt row = row_ind[i];
    data[i] *= diag[row];
  }
}

__global__ void mat_rmult_diag_kernel(const OSQPInt*   col_ind,
                                      const OSQPFloat* diag,
                                            OSQPFloat* data,
                                            OSQPInt    nnz) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < nnz; i += grid_size) {
    OSQPInt column = col_ind[i];
    data[i] *= diag[column];
  }
}

__global__ void mat_rmult_diag_new_kernel(const OSQPInt*   col_ind,
                                          const OSQPFloat* diag,
                                          const OSQPFloat* data_in,
                                                OSQPFloat* data_out,
                                                OSQPInt    nnz) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < nnz; i += grid_size) {
    OSQPInt column = col_ind[i];
    data_out[i] = data_in[i] * diag[column];
  }
}

__global__ void vec_abs_kernel(OSQPFloat* a,
                               OSQPInt    n) {

  OSQPInt i  = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < n) {
#ifdef OSQP_USE_FLOAT
    a[i] = fabsf(a[i]);
#else
    a[i] = fabs(a[i]);
#endif
  }
}

__global__ void scatter_kernel(OSQPFloat*       out,
                               const OSQPFloat* in,
                               const OSQPInt*   ind,
                               OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    OSQPInt j = ind[i];
    out[j] = in[i];
  }
}

/*
 * This code complements the cublasITamax routine which only returns the
 * one-based index to the maximum absolute value in d_x.
*/
__global__ void abs_kernel(const OSQPInt*   index_one_based,
                           const OSQPFloat* d_x,
                                 OSQPFloat* res) {

  /* cublasITamax returns one-based index */
  (*res) = abs(d_x[(*index_one_based)-1]);
}


/*******************************************************************************
 *                         Private functions                                   *
 *******************************************************************************/

/*
 *  out[j] = in[i], where j = ind[i] for i in [0,n-1]
 */
void scatter(OSQPFloat*       out,
             const OSQPFloat* in,
             const OSQPInt*   ind,
             OSQPInt          n) {

  OSQPInt num_blocks = (n / THREADS_PER_BLOCK) + 1;
  scatter_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(out, in, ind, n);
}


/*******************************************************************************
 *                          Thrust-related functions                           *
 *******************************************************************************/

template<typename BinaryFunction>
void Segmented_reduce(const OSQPInt*       key_start,
                            OSQPInt        number_of_keys,
                            OSQPInt        num_segments,
                      const OSQPFloat*     values,
                            void*          buffer,
                            OSQPFloat*     result,
                            BinaryFunction binary_op) {

  OSQPInt num_nnz_rows;

 /*  Memory layout of buffer:
  *  [ m*sizeof(OSQPFloat) Bytes | m*sizeof(OSQPInt) Bytes]
  *  where m = "number of rows"
  */
  OSQPFloat* intermediate_result = (OSQPFloat*) buffer;
  OSQPInt*   nnz_rows            = (OSQPInt*) (&intermediate_result[num_segments]);

  thrust::pair<OSQPInt*,OSQPFloat*> new_end;
  thrust::equal_to<OSQPInt> binary_pred;

  new_end = thrust::reduce_by_key(thrust::device,
                                  key_start,
                                  key_start + number_of_keys,
                                  values,
                                  nnz_rows,
                                  intermediate_result,
                                  binary_pred,
                                  binary_op);

  num_nnz_rows = new_end.first - nnz_rows;
  checkCudaErrors(cudaMemset(result, 0, num_segments * sizeof(OSQPFloat)));
  scatter(result, intermediate_result, nnz_rows, num_nnz_rows);
}

template<typename T>
struct abs_maximum {
  typedef T first_argument_type;
  typedef T second_argument_type;
  typedef T result_type;
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return max(abs(lhs), abs(rhs));}
 };

template void Segmented_reduce<abs_maximum<OSQPFloat>>(const OSQPInt*                key_start,
                                                             OSQPInt                 number_of_keys,
                                                             OSQPInt                 number_of_segments,
                                                       const OSQPFloat*              values,
                                                             void*                   buffer,
                                                             OSQPFloat*              result,
                                                             abs_maximum<OSQPFloat>  binary_op);


/*******************************************************************************
 *                           API Functions                                     *
 *******************************************************************************/

void cuda_vec_create(cusparseDnVecDescr_t* vec,
                     const OSQPFloat*      d_x,
                     OSQPInt               n) {

  /* Some versions of CUDA don't accept n=0 as a valid size (e.g. can't accept a
   * zero-length vector), so don't create the vector in that case.
   */
  if (n > 0)
    checkCudaErrors(cusparseCreateDnVec(vec, n, (void*)d_x, CUDA_FLOAT));
  else
    vec = NULL;
}

void cuda_vec_destroy(cusparseDnVecDescr_t vec) {

  if (vec) checkCudaErrors(cusparseDestroyDnVec(vec));
}

void cuda_vec_copy_d2d(OSQPFloat*       d_y,
                       const OSQPFloat* d_x,
                       OSQPInt          n) {

  checkCudaErrors(cudaMemcpy(d_y, d_x, n * sizeof(OSQPFloat), cudaMemcpyDeviceToDevice));
}

void cuda_vec_copy_h2d(OSQPFloat*       d_y,
                       const OSQPFloat* h_x,
                       OSQPInt          n) {

  checkCudaErrors(cudaMemcpy(d_y, h_x, n * sizeof(OSQPFloat), cudaMemcpyHostToDevice));
}

void cuda_vec_copy_d2h(OSQPFloat*       h_y,
                       const OSQPFloat* d_x,
                       OSQPInt          n) {

  checkCudaErrors(cudaMemcpy(h_y, d_x, n * sizeof(OSQPFloat), cudaMemcpyDeviceToHost));
}

void cuda_vec_int_copy_d2d(OSQPInt*       d_y,
                           const OSQPInt* d_x,
                           OSQPInt        n) {

  checkCudaErrors(cudaMemcpy(d_y, d_x, n * sizeof(OSQPInt), cudaMemcpyDeviceToDevice));
}

void cuda_vec_int_copy_h2d(OSQPInt*       d_y,
                           const OSQPInt* h_x,
                           OSQPInt        n) {

  checkCudaErrors(cudaMemcpy(d_y, h_x, n * sizeof(OSQPInt), cudaMemcpyHostToDevice));
}

void cuda_vec_int_copy_d2h(OSQPInt*       h_y,
                           const OSQPInt* d_x,
                           OSQPInt        n) {

  checkCudaErrors(cudaMemcpy(h_y, d_x, n * sizeof(OSQPInt), cudaMemcpyDeviceToHost));
}

void cuda_vec_set_sc(OSQPFloat* d_a,
                     OSQPFloat  sc,
                     OSQPInt    n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;
  vec_set_sc_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, sc, n);
}

void cuda_vec_set_sc_cond(OSQPFloat*     d_a,
                          const OSQPInt* d_test,
                          OSQPFloat      sc_if_neg,
                          OSQPFloat      sc_if_zero,
                          OSQPFloat      sc_if_pos,
                          OSQPInt        n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_set_sc_cond_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, d_test, sc_if_neg, sc_if_zero, sc_if_pos, n);
}

void cuda_vec_scale_penalty(OSQPFloat*       d_a1,
                            OSQPFloat*       d_a2,
                            OSQPFloat*       d_d,
                            const OSQPInt*   d_type,
                            OSQPInt          default_type,
                            OSQPFloat        c,
                            const OSQPFloat* d_E,
                            OSQPInt          invert,
                            OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_scale_penalty_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a1, d_a2, d_d, d_type, default_type, c, d_E, invert, n);
}

void cuda_vec_penalty_check(const OSQPFloat* d_a1,
                            const OSQPFloat* d_a2,
                            const OSQPFloat* d_d,
                            const OSQPInt*   d_type,
                            OSQPInt          default_type,
                            OSQPInt          n,
                            OSQPInt*         h_res,
                            OSQPInt*         d_res) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  *h_res = 0;
  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(OSQPInt), cudaMemcpyHostToDevice));

  vec_penalty_check_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a1, d_a2, d_d, d_type, default_type, d_res, n);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPInt), cudaMemcpyDeviceToHost));
}

/* Launch KERNEL specialized on the uniform type when d_type is NULL, so that a
   uniform problem runs a branch-free kernel, otherwise the per-row variant */
#define CUDA_PENALTY_DISPATCH(KERNEL, ...)                                     \
  do {                                                                         \
    if (d_type) {                                                              \
      KERNEL<-1><<<number_of_blocks, THREADS_PER_BLOCK>>>(__VA_ARGS__);        \
    }                                                                          \
    else {                                                                     \
      switch (default_type) {                                                  \
        case OSQP_PENALTY_L1L2:                                                \
          KERNEL<OSQP_PENALTY_L1L2><<<number_of_blocks, THREADS_PER_BLOCK>>>(__VA_ARGS__);  \
          break;                                                               \
        case OSQP_PENALTY_HUBER:                                               \
          KERNEL<OSQP_PENALTY_HUBER><<<number_of_blocks, THREADS_PER_BLOCK>>>(__VA_ARGS__); \
          break;                                                               \
        case OSQP_PENALTY_NORM2:                                               \
          KERNEL<OSQP_PENALTY_NORM2><<<number_of_blocks, THREADS_PER_BLOCK>>>(__VA_ARGS__); \
          break;                                                               \
        case OSQP_PENALTY_NORMINF:                                             \
          KERNEL<OSQP_PENALTY_NORMINF><<<number_of_blocks, THREADS_PER_BLOCK>>>(__VA_ARGS__); \
          break;                                                               \
        default:                                                               \
          KERNEL<OSQP_PENALTY_NONE><<<number_of_blocks, THREADS_PER_BLOCK>>>(__VA_ARGS__);  \
          break;                                                               \
      }                                                                        \
    }                                                                          \
  } while (0)

void cuda_vec_prox_penalty(      OSQPFloat* d_z,
                           const OSQPFloat* d_v,
                           const OSQPFloat* d_l,
                           const OSQPFloat* d_u,
                           const OSQPFloat* d_rho_vec,
                                 OSQPFloat  rho,
                           const OSQPFloat* d_a1,
                           const OSQPFloat* d_a2,
                           const OSQPFloat* d_d,
                           const OSQPInt*   d_type,
                                 OSQPInt    default_type,
                                 OSQPInt    n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  /* The all-hard case is the plain projection, bit for bit */
  if (!d_type && (default_type == OSQP_PENALTY_NONE)) {
    cuda_vec_bound(d_z, d_v, d_l, d_u, n);
    return;
  }

  CUDA_PENALTY_DISPATCH(vec_prox_penalty_kernel, d_z, d_v, d_l, d_u,
                        d_rho_vec, rho, d_a1, d_a2, d_d, d_type, n);
}

void cuda_vec_penalty_value(const OSQPFloat* d_z,
                            const OSQPFloat* d_l,
                            const OSQPFloat* d_u,
                            const OSQPFloat* d_a1,
                            const OSQPFloat* d_a2,
                            const OSQPFloat* d_d,
                            const OSQPInt*   d_type,
                                  OSQPInt    default_type,
                                  OSQPInt    n,
                                  OSQPFloat* h_res,
                                  OSQPFloat* d_res) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  *h_res = 0.0;

  if (!d_type && (default_type == OSQP_PENALTY_NONE)) return;

  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(OSQPFloat), cudaMemcpyHostToDevice));

  CUDA_PENALTY_DISPATCH(vec_penalty_value_kernel, d_z, d_l, d_u,
                        d_a1, d_a2, d_d, d_type, d_res, n);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPFloat), cudaMemcpyDeviceToHost));
}

void cuda_vec_penalty_reccone_rate(const OSQPFloat* d_w,
                                   const OSQPFloat* d_l,
                                   const OSQPFloat* d_u,
                                   const OSQPFloat* d_a1,
                                   const OSQPFloat* d_a2,
                                   const OSQPFloat* d_d,
                                   const OSQPInt*   d_type,
                                   OSQPInt          default_type,
                                   OSQPFloat        infval,
                                   OSQPFloat        tol,
                                   OSQPInt          n,
                                   OSQPFloat*       h_res,
                                   OSQPFloat*       d_res) {
  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  *h_res = 0.0;
  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(OSQPFloat), cudaMemcpyHostToDevice));

  CUDA_PENALTY_DISPATCH(vec_penalty_reccone_rate_kernel, d_w, d_l, d_u,
                        d_a1, d_a2, d_d, d_type, infval, tol, d_res, n);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPFloat), cudaMemcpyDeviceToHost));
  if (*h_res >= OSQP_INFTY) *h_res = OSQP_INFTY;
}

void cuda_vec_penalty_conj_value(const OSQPFloat* d_y,
                                 const OSQPFloat* d_a1,
                                 const OSQPFloat* d_a2,
                                 const OSQPFloat* d_d,
                                 const OSQPInt*   d_type,
                                       OSQPInt    default_type,
                                       OSQPInt    n,
                                       OSQPFloat* h_res,
                                       OSQPFloat* d_res) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  *h_res = 0.0;

  if (!d_type && (default_type == OSQP_PENALTY_NONE)) return;

  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(OSQPFloat), cudaMemcpyHostToDevice));

  CUDA_PENALTY_DISPATCH(vec_penalty_conj_value_kernel, d_y,
                        d_a1, d_a2, d_d, d_type, d_res, n);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPFloat), cudaMemcpyDeviceToHost));

  /* Rows outside dom phi* each contribute OSQP_INFTY to the sum */
  if (*h_res >= OSQP_INFTY) *h_res = OSQP_INFTY;
}

void cuda_vec_group_prox_penalty(      OSQPFloat* d_z,
                                 const OSQPFloat* d_v,
                                 const OSQPFloat* d_l,
                                 const OSQPFloat* d_u,
                                       OSQPFloat  rho,
                                 const OSQPFloat* d_a1,
                                 const OSQPInt*   d_type,
                                       OSQPInt    default_type,
                                 const OSQPInt*   d_gp,
                                 const OSQPInt*   d_rows,
                                       OSQPInt    ngroups,
                                       OSQPFloat* d_sort_tmp) {

  if (ngroups <= 0) return;

  group_prox_penalty_kernel<<<ngroups, GROUP_THREADS>>>(d_z, d_v, d_l, d_u, rho,
                                                        d_a1, d_type, d_gp, d_rows,
                                                        default_type, d_sort_tmp);
}

void cuda_vec_group_penalty_value(const OSQPFloat* d_z,
                                  const OSQPFloat* d_l,
                                  const OSQPFloat* d_u,
                                  const OSQPFloat* d_a1,
                                  const OSQPInt*   d_type,
                                        OSQPInt    default_type,
                                  const OSQPInt*   d_gp,
                                  const OSQPInt*   d_rows,
                                        OSQPInt    ngroups,
                                        OSQPFloat* h_res,
                                        OSQPFloat* d_res) {

  *h_res = 0.0;

  if (ngroups <= 0) return;

  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(OSQPFloat), cudaMemcpyHostToDevice));

  group_penalty_value_kernel<<<ngroups, GROUP_THREADS>>>(d_z, d_l, d_u, d_a1,
                                                         d_type, d_gp, d_rows,
                                                         default_type, d_res);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPFloat), cudaMemcpyDeviceToHost));
}

void cuda_vec_group_penalty_reccone_rate(const OSQPFloat* d_w,
                                         const OSQPFloat* d_l,
                                         const OSQPFloat* d_u,
                                         const OSQPFloat* d_a1,
                                         const OSQPInt*   d_type,
                                               OSQPInt    default_type,
                                         const OSQPInt*   d_gp,
                                         const OSQPInt*   d_rows,
                                               OSQPInt    ngroups,
                                               OSQPFloat  infval,
                                               OSQPFloat* h_res,
                                               OSQPFloat* d_res) {

  *h_res = 0.0;

  if (ngroups <= 0) return;

  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(OSQPFloat), cudaMemcpyHostToDevice));

  group_penalty_reccone_rate_kernel<<<ngroups, GROUP_THREADS>>>(d_w, d_l, d_u, d_a1,
                                                                d_type, d_gp, d_rows,
                                                                default_type, infval,
                                                                d_res);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPFloat), cudaMemcpyDeviceToHost));
}

void cuda_vec_group_penalty_conj_value(const OSQPFloat* d_y,
                                       const OSQPFloat* d_a1,
                                       const OSQPInt*   d_type,
                                             OSQPInt    default_type,
                                       const OSQPInt*   d_gp,
                                       const OSQPInt*   d_rows,
                                             OSQPInt    ngroups,
                                             OSQPFloat* h_res,
                                             OSQPFloat* d_res) {

  *h_res = 0.0;

  if (ngroups <= 0) return;

  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(OSQPFloat), cudaMemcpyHostToDevice));

  group_penalty_conj_value_kernel<<<ngroups, GROUP_THREADS>>>(d_y, d_a1, d_type,
                                                              d_gp, d_rows,
                                                              default_type, d_res);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPFloat), cudaMemcpyDeviceToHost));

  /* A group outside its dual ball contributes OSQP_INFTY to the sum */
  if (*h_res >= OSQP_INFTY) *h_res = OSQP_INFTY;
}

void cuda_vec_group_equalize_scaling(      OSQPFloat* d_corr,
                                     const OSQPFloat* d_E,
                                           OSQPInt    n,
                                     const OSQPInt*   d_gp,
                                     const OSQPInt*   d_rows,
                                           OSQPInt    ngroups) {

  /* Ungrouped rows keep their scaling, so the correction is 1 there */
  cuda_vec_set_sc(d_corr, 1.0, n);

  if (ngroups <= 0) return;

  group_equalize_scaling_kernel<<<ngroups, GROUP_THREADS>>>(d_corr, d_E, d_gp, d_rows);
}

void cuda_vec_group_penalty_check(const OSQPFloat* d_a1,
                                  const OSQPFloat* d_a2,
                                  const OSQPFloat* d_d,
                                  const OSQPInt*   d_type,
                                        OSQPInt    default_type,
                                        OSQPInt    n,
                                  const OSQPInt*   d_gp,
                                  const OSQPInt*   d_rows,
                                        OSQPInt    ngroups,
                                        OSQPInt    ngrouped,
                                        OSQPInt*   h_res,
                                        OSQPInt*   d_res) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  unsigned long long  h_count = 0;
  unsigned long long* d_count;

  *h_res = 0;
  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(OSQPInt), cudaMemcpyHostToDevice));

  /* The count is a second reduction target, so it cannot share d_res */
  checkCudaErrors(cudaMalloc(&d_count, sizeof(unsigned long long)));
  checkCudaErrors(cudaMemcpy(d_count, &h_count, sizeof(unsigned long long),
                             cudaMemcpyHostToDevice));

  group_type_count_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_type, default_type,
                                                                   d_count, n);

  if (ngroups > 0) {
    group_penalty_check_kernel<<<ngroups, GROUP_THREADS>>>(d_a1, d_a2, d_d, d_type,
                                                           d_gp, d_rows, default_type,
                                                           d_res);
  }

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPInt), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&h_count, d_count, sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_count));

  /* Disagreeing either way: a group type on an ungrouped row, or a grouped row
     whose type is separable. The per-group kernel catches the latter as
     GROUP_MIXED, so what this adds is the former. */
  if (h_count != (unsigned long long)ngrouped) *h_res |= OSQP_PENALTY_ERR_GROUP_TYPE;
}

void cuda_vec_round(OSQPFloat* d_a,
                    OSQPFloat  tol,
                    OSQPInt    n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;
  vec_round_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, tol, n);
}

void cuda_vec_mult_sc(OSQPFloat* d_a,
                      OSQPFloat  sc,
                      OSQPInt    n) {

  checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, n, &sc, d_a, 1));
}

void cuda_vec_add_scaled(OSQPFloat*       d_x,
                         const OSQPFloat* d_a,
                         const OSQPFloat* d_b,
                         OSQPFloat        sca,
                         OSQPFloat        scb,
                         OSQPInt          n) {

  if (d_x != d_a || sca != 1.0) {
    if (sca == 1.0) {
      /* d_x = d_a */
      checkCudaErrors(cudaMemcpy(d_x, d_a, n * sizeof(OSQPFloat), cudaMemcpyDeviceToDevice));
    }
    else if (d_x == d_a) {
      /* d_x *= sca */
      checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, n, &sca, d_x, 1));
    }
    else {
      /* d_x = 0 */
      checkCudaErrors(cudaMemset(d_x, 0, n * sizeof(OSQPFloat)));

      /* d_x += sca * d_a */
      checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &sca, d_a, 1, d_x, 1));
    }
  }

  /* d_x += scb * d_b */
  checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &scb, d_b, 1, d_x, 1));
}

void cuda_vec_add_scaled3(OSQPFloat*       d_x,
                          const OSQPFloat* d_a,
                          const OSQPFloat* d_b,
                          const OSQPFloat* d_c,
                          OSQPFloat        sca,
                          OSQPFloat        scb,
                          OSQPFloat        scc,
                          OSQPInt          n) {

  if (d_x != d_a || sca != 1.0) {
    if (sca == 1.0) {
      /* d_x = d_a */
      checkCudaErrors(cudaMemcpy(d_x, d_a, n * sizeof(OSQPFloat), cudaMemcpyDeviceToDevice));
    }
    else if (d_x == d_a) {
      /* d_x *= sca */
      checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, n, &sca, d_x, 1));
    }
    else {
      /* d_x = 0 */
      checkCudaErrors(cudaMemset(d_x, 0, n * sizeof(OSQPFloat)));

      /* d_x += sca * d_a */
      checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &sca, d_a, 1, d_x, 1));
    }
  }

  /* d_x += scb * d_b */
  checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &scb, d_b, 1, d_x, 1));

  /* d_x += scc * d_c */
  checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &scc, d_c, 1, d_x, 1));
}

void cuda_vec_norm_inf(const OSQPFloat* d_x,
                             OSQPInt    n,
                             OSQPFloat* h_res) {

  cublasPointerMode_t mode;
  checkCudaErrors(cublasGetPointerMode(CUDA_handle->cublasHandle, &mode));

  if (mode == CUBLAS_POINTER_MODE_DEVICE) {
    checkCudaErrors(cublasITamax(CUDA_handle->cublasHandle, n, d_x, 1, CUDA_handle->d_index));
    abs_kernel<<<1,1>>>(CUDA_handle->d_index, d_x, h_res);  /* d_res actually */
  }
  else {
    OSQPInt idx;
    checkCudaErrors(cublasITamax(CUDA_handle->cublasHandle, n, d_x, 1, &idx));
    checkCudaErrors(cudaMemcpy(h_res, d_x + (idx-1), sizeof(OSQPFloat), cudaMemcpyDeviceToHost));
    (*h_res) = abs(*h_res);
  }
}

void cuda_vec_norm_2(const OSQPFloat* d_x,
                           OSQPInt    n,
                           OSQPFloat* h_res) {

  cublasTnrm2(CUDA_handle->cublasHandle, n, d_x, 1, h_res);
}

void cuda_vec_scaled_norm_inf(const OSQPFloat* d_S,
                              const OSQPFloat* d_v,
                                    OSQPInt    n,
                                    OSQPFloat* h_res) {

  OSQPFloat *d_v_scaled;

  cuda_malloc((void **) &d_v_scaled, n * sizeof(OSQPFloat));

  /* d_v_scaled = d_S * d_v */
  cuda_vec_ew_prod(d_v_scaled, d_S, d_v, n);

  /* (*h_res) = |d_v_scaled|_inf */
  cuda_vec_norm_inf(d_v_scaled, n, h_res);

  cuda_free((void **) &d_v_scaled);
}

void cuda_vec_diff_norm_inf(const OSQPFloat* d_a,
                            const OSQPFloat* d_b,
                                  OSQPInt    n,
                                  OSQPFloat* h_res) {

  OSQPFloat *d_diff;

  cuda_malloc((void **) &d_diff, n * sizeof(OSQPFloat));

  /* d_diff = d_a - d_b */
  cuda_vec_add_scaled(d_diff, d_a, d_b, 1.0, -1.0, n);

  /* (*h_res) = |d_diff|_inf */
  cuda_vec_norm_inf(d_diff, n, h_res);

  cuda_free((void **) &d_diff);
}

void cuda_vec_norm_1(const OSQPFloat* d_x,
                           OSQPInt    n,
                           OSQPFloat* h_res) {

  checkCudaErrors(cublasTasum(CUDA_handle->cublasHandle, n, d_x, 1, h_res));
}

void cuda_vec_prod(const OSQPFloat* d_a,
                   const OSQPFloat* d_b,
                         OSQPInt    n,
                         OSQPFloat* h_res) {

  checkCudaErrors(cublasTdot(CUDA_handle->cublasHandle, n, d_a, 1, d_b, 1, h_res));
}

void cuda_vec_prod_signed(const OSQPFloat* d_a,
                          const OSQPFloat* d_b,
                                OSQPInt    sign,
                                OSQPInt    n,
                                OSQPFloat* h_res) {

  OSQPFloat *d_res;
  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  cuda_calloc((void **) &d_res, sizeof(OSQPFloat));

  if (sign == 1) {
    vec_prod_pos_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_res, n);
    checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPFloat), cudaMemcpyDeviceToHost));
  }
  else if (sign == -1) {
    vec_prod_neg_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_res, n);
    checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPFloat), cudaMemcpyDeviceToHost));
  }
  else {
    checkCudaErrors(cublasTdot(CUDA_handle->cublasHandle, n, d_a, 1, d_b, 1, h_res));
  }

  cuda_free((void **) &d_res);
}

void cuda_vec_ew_prod(OSQPFloat*       d_c,
                      const OSQPFloat* d_a,
                      const OSQPFloat* d_b,
                      OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_ew_prod_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_c, d_a, d_b, n);
}

void cuda_vec_eq(const OSQPFloat* a,
                 const OSQPFloat* b,
                       OSQPFloat  tol,
                       OSQPInt    n,
                       OSQPInt*   h_res) {

  OSQPInt *d_res;
  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  cuda_malloc((void **) &d_res, sizeof(OSQPInt));

  /* Initialize d_res to 1 */
  *h_res = 1;
  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(OSQPInt), cudaMemcpyHostToDevice));

  vec_eq_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(a, b, tol, d_res, n);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPInt), cudaMemcpyDeviceToHost));

  cuda_free((void **) &d_res);
}

void cuda_vec_int_eq(const OSQPInt* a,
                     const OSQPInt* b,
                           OSQPInt  n,
                           OSQPInt* h_res) {

  OSQPInt *d_res;
  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  cuda_malloc((void **) &d_res, sizeof(OSQPInt));

  /* Initialize d_res to 1 */
  *h_res = 1;
  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(OSQPInt), cudaMemcpyHostToDevice));

  vec_int_eq_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(a, b, d_res, n);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPInt), cudaMemcpyDeviceToHost));

  cuda_free((void **) &d_res);
}

void cuda_vec_leq(const OSQPFloat* d_l,
                  const OSQPFloat* d_u,
                        OSQPInt    n,
                        OSQPInt*   h_res) {

  OSQPInt *d_res;
  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  cuda_malloc((void **) &d_res, sizeof(OSQPInt));

  /* Initialize d_res to 1 */
  *h_res = 1;
  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(OSQPInt), cudaMemcpyHostToDevice));

  vec_leq_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_l, d_u, d_res, n);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPInt), cudaMemcpyDeviceToHost));

  cuda_free((void **) &d_res);
}

void cuda_vec_bound(OSQPFloat*       d_x,
                    const OSQPFloat* d_z,
                    const OSQPFloat* d_l,
                    const OSQPFloat* d_u,
                    OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_bound_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_x, d_z, d_l, d_u, n);
}

void cuda_vec_project_polar_reccone(OSQPFloat*       d_y,
                                    const OSQPFloat* d_l,
                                    const OSQPFloat* d_u,
                                    const OSQPInt*   d_type,
                                    OSQPInt          default_type,
                                    OSQPFloat        infval,
                                    OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_project_polar_reccone_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_y, d_l, d_u, d_type, default_type, infval, n);
}

void cuda_vec_in_reccone(const OSQPFloat* d_y,
                         const OSQPFloat* d_l,
                         const OSQPFloat* d_u,
                         const OSQPFloat* d_alpha2,
                         const OSQPInt*   d_type,
                         OSQPInt          default_type,
                         OSQPFloat        infval,
                         OSQPFloat        tol,
                         OSQPInt          n,
                         OSQPInt*         h_res) {

  OSQPInt *d_res;
  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  cuda_malloc((void **) &d_res, sizeof(OSQPInt));

  /* Initialize d_res to 1 */
  *h_res = 1;
  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(OSQPInt), cudaMemcpyHostToDevice));

  vec_in_reccone_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_y, d_l, d_u, d_alpha2, d_type, default_type, infval, tol, d_res, n);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPInt), cudaMemcpyDeviceToHost));

  cuda_free((void **) &d_res);
}

void cuda_vec_reciprocal(OSQPFloat*       d_b,
                         const OSQPFloat* d_a,
                         OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_reciprocal_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_b, d_a, n);
}

void cuda_vec_sqrt(OSQPFloat* d_a,
                   OSQPInt    n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_sqrt_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, n);
}

void cuda_vec_max(OSQPFloat*       d_c,
                  const OSQPFloat* d_a,
                  const OSQPFloat* d_b,
                  OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_max_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_c, d_a, d_b, n);
}

void cuda_vec_min(OSQPFloat*       d_c,
                  const OSQPFloat* d_a,
                  const OSQPFloat* d_b,
                  OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_min_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_c, d_a, d_b, n);
}

void cuda_vec_bounds_type(OSQPInt*         d_iseq,
                          const OSQPFloat* d_l,
                          const OSQPFloat* d_u,
                          const OSQPInt*   d_type,
                          OSQPInt          default_type,
                          OSQPFloat        infval,
                          OSQPFloat        tol,
                          OSQPInt          n,
                          OSQPInt*         h_has_changed) {

  OSQPInt *d_has_changed;
  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  /* Initialize d_has_changed to zero */
  cuda_calloc((void **) &d_has_changed, sizeof(OSQPInt));

  vec_bounds_type_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_iseq, d_l, d_u, d_type, default_type, infval, tol, d_has_changed, n);

  checkCudaErrors(cudaMemcpy(h_has_changed, d_has_changed, sizeof(OSQPInt), cudaMemcpyDeviceToHost));

  cuda_free((void **) &d_has_changed);
}

void cuda_vec_set_sc_if_lt(OSQPFloat*       d_x,
                           const OSQPFloat* d_z,
                           OSQPFloat        testval,
                           OSQPFloat        newval,
                           OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_set_sc_if_lt_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_x, d_z, testval, newval, n);
}

void cuda_vec_set_sc_if_gt(OSQPFloat*       d_x,
                           const OSQPFloat* d_z,
                           OSQPFloat        testval,
                           OSQPFloat        newval,
                           OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_set_sc_if_gt_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_x, d_z, testval, newval, n);
}

void cuda_vec_segmented_sum(const OSQPFloat* d_values,
                            const OSQPInt*   d_keys,
                                  OSQPFloat* d_res,
                                  void*      d_buffer,
                                  OSQPInt    num_segments,
                                  OSQPInt    num_elements) {

  thrust::plus<OSQPFloat> binary_op;
  Segmented_reduce(d_keys, num_elements, num_segments, d_values, d_buffer, d_res, binary_op);
}

void cuda_vec_gather(OSQPInt          nnz,
                     const OSQPFloat* d_y,
                     OSQPFloat*       d_xVal,
                     const OSQPInt*   d_xInd) {

  thrust::gather(thrust::device, d_xInd, d_xInd + nnz, d_y, d_xVal);
}

void cuda_mat_mult_sc(csr*      S,
                      csr*      At,
                      OSQPFloat sc) {

  checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, S->nnz, &sc, S->val, 1));

  if (At) {
    /* Update At as well */
    checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, At->nnz, &sc, At->val, 1));
  }
}

void cuda_mat_lmult_diag(csr*             S,
                         csr*             At,
                         const OSQPFloat* d_diag) {

  OSQPInt nnz = S->nnz;
  OSQPInt number_of_blocks = (nnz / THREADS_PER_BLOCK) / ELEMENTS_PER_THREAD + 1;

  mat_lmult_diag_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(S->row_ind, d_diag, S->val, nnz);

  if (At) {
    /* Multiply At from right */
    mat_rmult_diag_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(At->col_ind, d_diag, At->val, nnz);
  }
}

void cuda_mat_rmult_diag(csr*             S,
                         csr*             At,
                         const OSQPFloat* d_diag) {

  OSQPInt nnz = S->nnz;
  OSQPInt number_of_blocks = (nnz / THREADS_PER_BLOCK) / ELEMENTS_PER_THREAD + 1;

  mat_rmult_diag_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(S->col_ind, d_diag, S->val, nnz);

  if (At) {
    /* Multiply At from left */
    mat_lmult_diag_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(At->row_ind, d_diag, At->val, nnz);
  }
}

void cuda_mat_rmult_diag_new(const csr*       S,
                                   OSQPFloat* d_buffer,
                             const OSQPFloat* d_diag) {

  OSQPInt nnz = S->nnz;
  OSQPInt number_of_blocks = (nnz / THREADS_PER_BLOCK) / ELEMENTS_PER_THREAD + 1;

  mat_rmult_diag_new_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(S->col_ind, d_diag, S->val, d_buffer, nnz);
}

void cuda_mat_Axpy(const csr*                 A,
                   const cusparseDnVecDescr_t vecx,
                         cusparseDnVecDescr_t vecy,
                         OSQPFloat            alpha,
                         OSQPFloat            beta) {

  checkCudaErrors(cusparseSpMV(
    CUDA_handle->cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, A->SpMatDescr, vecx, &beta, vecy,
    CUDA_FLOAT, CUSPARSE_SPMV_ALGORITHM_DEFAULT, A->SpMatBuffer));
}

void cuda_mat_row_norm_inf(const csr*       S,
                                 OSQPFloat* d_res) {

  OSQPInt nnz      = S->nnz;
  OSQPInt num_rows = S->m;

  if (nnz == 0) return;

  abs_maximum<OSQPFloat> binary_op;
  void *d_buffer;
  cuda_malloc(&d_buffer, num_rows * (sizeof(OSQPFloat) + sizeof(OSQPInt)));

  /*
  *  For rows with only one element, the element itself is returned.
  *  Therefore, we have to take the absolute value to get the inf-norm.
  */
  Segmented_reduce(S->row_ind, nnz, num_rows, S->val, d_buffer, d_res, binary_op);
  vec_abs_kernel<<<num_rows/THREADS_PER_BLOCK+1,THREADS_PER_BLOCK>>>(d_res, num_rows);

  cuda_free(&d_buffer);
}
