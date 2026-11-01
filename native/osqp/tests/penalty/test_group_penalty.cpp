#include <catch2/catch.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */
#include "penalty.h"     /* penalty_project, Phi(R(z)) and Phi*(y) */

/* Non-separable penalties, which pool the slacks of a group of rows under a
 * norm. The prox is checked against its closed form, the infinity-norm case
 * end to end against the equivalent hard QP, and the Euclidean case against
 * the KKT conditions it is supposed to satisfy. */

namespace {

const OSQPInt MAX_ITER = 20000;

#ifdef OSQP_USE_FLOAT
const OSQPFloat SOLVE_EPS   = 1e-5;
const OSQPFloat COMPARE_TOL = 1e-3;
#else
const OSQPFloat SOLVE_EPS   = 1e-9;
const OSQPFloat COMPARE_TOL = 1e-5;
#endif

std::vector<OSQPFloat> to_vector(const OSQPVectorf* vec, OSQPInt len) {
  std::vector<OSQPFloat> out(len);
  OSQPVectorf_to_raw(out.data(), vec);
  return out;
}

OSQPFloat norm2(const std::vector<OSQPFloat>& v) {
  OSQPFloat acc = 0.0;
  for (OSQPFloat x : v) acc += x * x;
  return std::sqrt(acc);
}

OSQPFloat norminf(const std::vector<OSQPFloat>& v) {
  OSQPFloat acc = 0.0;
  for (OSQPFloat x : v) acc = std::max(acc, std::abs(x));
  return acc;
}

OSQPFloat norm1(const std::vector<OSQPFloat>& v) {
  OSQPFloat acc = 0.0;
  for (OSQPFloat x : v) acc += std::abs(x);
  return acc;
}

/* prox_{alpha*||.||_inf/rho}(r), by bisecting the threshold rather than
   sorting, so that the reference shares no code with the implementation */
std::vector<OSQPFloat> ref_prox_norminf(const std::vector<OSQPFloat>& r,
                                        OSQPFloat                     alpha,
                                        OSQPFloat                     rho) {

  const OSQPFloat tau = alpha / rho;
  std::vector<OSQPFloat> out(r.size(), 0.0);

  if (norm1(r) <= tau) return out;

  OSQPFloat lo = 0.0, hi = norminf(r);

  for (int it = 0; it < 200; it++) {
    OSQPFloat mid = 0.5 * (lo + hi);
    OSQPFloat sum = 0.0;

    for (OSQPFloat x : r) sum += std::max(std::abs(x) - mid, (OSQPFloat)0.0);

    if (sum > tau) lo = mid;
    else           hi = mid;
  }

  const OSQPFloat lambda = 0.5 * (lo + hi);

  for (size_t i = 0; i < r.size(); i++) {
    OSQPFloat s = std::min(std::abs(r[i]), lambda);
    out[i] = r[i] > 0.0 ? s : -s;
  }

  return out;
}

std::vector<OSQPFloat> ref_prox_norm2(const std::vector<OSQPFloat>& r,
                                      OSQPFloat                     alpha,
                                      OSQPFloat                     rho) {

  const OSQPFloat tau = alpha / rho;
  const OSQPFloat nrm = norm2(r);
  std::vector<OSQPFloat> out(r.size(), 0.0);

  if (nrm <= tau) return out;

  for (size_t i = 0; i < r.size(); i++) out[i] = (1.0 - tau / nrm) * r[i];

  return out;
}

/* A diagonal QP whose rows are all soft, so a group can pool any subset. */
class group_test_fixture : public OSQPTestFixture {
public:
  group_test_fixture() {
    data = std::make_unique<OSQPTestData>();

    data->n = 4;
    data->m = 4;

    OSQPFloat A_diag[4] = {1.0, 1.0, 1.0, 1.0};

    data->P = OSQPCscMatrix_identity(data->n);
    data->A = OSQPCscMatrix_diag_vec(data->m, data->n, A_diag);

    data->q = (OSQPFloat*)c_malloc(data->n * sizeof(OSQPFloat));
    data->l = (OSQPFloat*)c_malloc(data->m * sizeof(OSQPFloat));
    data->u = (OSQPFloat*)c_malloc(data->m * sizeof(OSQPFloat));

    /* q pulls x away from the box, so the slack is genuinely nonzero */
    for (OSQPInt i = 0; i < data->n; i++) data->q[i] = -2.0 - 0.5 * i;
    for (OSQPInt i = 0; i < data->m; i++) {
      data->l[i] = -1.0;
      data->u[i] = 1.0;
    }

    settings->verbose  = 0;
    settings->eps_abs  = SOLVE_EPS;
    settings->eps_rel  = SOLVE_EPS;
    settings->max_iter = MAX_ITER;
  }

  void setup_solver() {
    OSQPInt exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                                  data->A, data->l, data->u,
                                  data->m, data->n, settings.get());
    solver.reset(tmpSolver);

    mu_assert("Group penalty: setup error!", exitflag == 0);
  }

  /* Rebuild the problem at a different size, keeping it diagonal */
  void resize(OSQPInt m) {
    c_free(data->q); c_free(data->l); c_free(data->u);
    OSQPCscMatrix_free(data->P);
    OSQPCscMatrix_free(data->A);

    std::vector<OSQPFloat> A_diag(m, 1.0);

    data->n = m;
    data->m = m;
    data->P = OSQPCscMatrix_identity(m);
    data->A = OSQPCscMatrix_diag_vec(m, m, A_diag.data());
    data->q = (OSQPFloat*)c_malloc(m * sizeof(OSQPFloat));
    data->l = (OSQPFloat*)c_malloc(m * sizeof(OSQPFloat));
    data->u = (OSQPFloat*)c_malloc(m * sizeof(OSQPFloat));

    for (OSQPInt i = 0; i < m; i++) {
      data->q[i] = -1.0;
      data->l[i] = -1.0;
      data->u[i] =  1.0;
    }
  }

  /* One group over every row, with a uniform type and weight */
  OSQPInt setup_one_group(OSQPInt type, OSQPFloat alpha) {
    std::vector<OSQPFloat> a1(data->m, alpha), a2(data->m, 0.0), d(data->m, 0.0);
    std::vector<OSQPInt>   gid(data->m, 0);

    return osqp_setup_penalty(solver.get(), type, OSQP_NULL,
                              a1.data(), a2.data(), d.data(),
                              1, gid.data());
  }
};

} // namespace


TEST_CASE_METHOD(group_test_fixture, "Group penalty: prox matches its closed form", "[penalty],[group]")
{
  settings->scaling    = 0;    /* Weights stay in the caller's units */
  settings->rho_is_vec = 1;
  settings->rho        = 0.7;
  setup_solver();

  const OSQPFloat alpha = 1.3;

  /* v is pushed well outside [l,u] on some rows and left inside on others, so
     that the residual the prox sees has mixed signs and magnitudes */
  std::vector<OSQPFloat> v = {3.0, -2.5, 0.5, 1.4};
  std::vector<OSQPFloat> r(data->m);

  for (OSQPInt i = 0; i < data->m; i++) {
    OSQPFloat vbar = std::min(std::max(v[i], data->l[i]), data->u[i]);
    r[i] = v[i] - vbar;
  }

  auto project = [&](OSQPInt type) {
    REQUIRE(setup_one_group(type, alpha) == 0);

    OSQPVectorf_ptr vvec{OSQPVectorf_new(v.data(), data->m)};
    OSQPVectorf_ptr zvec{OSQPVectorf_malloc(data->m)};

    penalty_project(solver.get(), zvec.get(), vvec.get());

    return to_vector(zvec.get(), data->m);
  };

  SECTION("Euclidean norm") {
    auto z   = project(OSQP_PENALTY_NORM2);
    auto ref = ref_prox_norm2(r, alpha, settings->rho);

    for (OSQPInt i = 0; i < data->m; i++) {
      OSQPFloat vbar = std::min(std::max(v[i], data->l[i]), data->u[i]);
      REQUIRE(z[i] == Approx(vbar + ref[i]).margin(COMPARE_TOL));
    }
  }

  SECTION("Infinity norm") {
    auto z   = project(OSQP_PENALTY_NORMINF);
    auto ref = ref_prox_norminf(r, alpha, settings->rho);

    for (OSQPInt i = 0; i < data->m; i++) {
      OSQPFloat vbar = std::min(std::max(v[i], data->l[i]), data->u[i]);
      REQUIRE(z[i] == Approx(vbar + ref[i]).margin(COMPARE_TOL));
    }
  }

  SECTION("Infinity norm inside the dual ball") {
    /* ||r||_1 <= alpha/rho drives the whole group's slack to zero */
    const OSQPFloat big = 10.0 * norm1(r) * settings->rho;

    REQUIRE(setup_one_group(OSQP_PENALTY_NORMINF, big) == 0);

    OSQPVectorf_ptr vvec{OSQPVectorf_new(v.data(), data->m)};
    OSQPVectorf_ptr zvec{OSQPVectorf_malloc(data->m)};

    penalty_project(solver.get(), zvec.get(), vvec.get());
    auto z = to_vector(zvec.get(), data->m);

    for (OSQPInt i = 0; i < data->m; i++) {
      OSQPFloat vbar = std::min(std::max(v[i], data->l[i]), data->u[i]);
      REQUIRE(z[i] == Approx(vbar).margin(COMPARE_TOL));
    }
  }
}

TEST_CASE_METHOD(group_test_fixture, "Group penalty: the prox sorts long groups", "[penalty],[group]")
{
  /* Past OSQP_PENALTY_SORT_SMALL the l_inf prox swaps insertion sort for
     heapsort, so the threshold has to come out the same either side of it. */
  const OSQPInt   m     = 70;
  const OSQPFloat alpha = 4.0;

  resize(m);

  settings->scaling = 0;
  settings->rho     = 1.0;
  setup_solver();

  REQUIRE(setup_one_group(OSQP_PENALTY_NORMINF, alpha) == 0);

  /* Deliberately unsorted magnitudes, including ties at the maximum */
  std::vector<OSQPFloat> v(m), r(m);

  for (OSQPInt i = 0; i < m; i++) {
    v[i] = 1.0 + ((i * 37) % 23) * 0.5;
    if (i % 11 == 0) v[i] = 12.0;          /* ties */
    if (i % 5 == 0)  v[i] = -v[i];
    r[i] = v[i] - std::min(std::max(v[i], data->l[i]), data->u[i]);
  }

  OSQPVectorf_ptr vvec{OSQPVectorf_new(v.data(), m)};
  OSQPVectorf_ptr zvec{OSQPVectorf_malloc(m)};

  penalty_project(solver.get(), zvec.get(), vvec.get());

  auto z   = to_vector(zvec.get(), m);
  auto ref = ref_prox_norminf(r, alpha, settings->rho);

  for (OSQPInt i = 0; i < m; i++) {
    OSQPFloat vbar = std::min(std::max(v[i], data->l[i]), data->u[i]);
    REQUIRE(z[i] == Approx(vbar + ref[i]).margin(COMPARE_TOL));
  }
}

TEST_CASE_METHOD(group_test_fixture, "Group penalty: a group of one is a plain L1 row", "[penalty],[group]")
{
  /* Both norms of a scalar are the absolute value, so a size-1 group must
     reproduce OSQP_PENALTY_L1L2 with alpha2 == 0 exactly. */
  const OSQPFloat alpha = 0.8;

  settings->scaling = 0;
  setup_solver();

  std::vector<OSQPFloat> a1(data->m, alpha), a2(data->m, 0.0), d(data->m, 0.0);

  REQUIRE(osqp_setup_penalty(solver.get(), OSQP_PENALTY_L1L2, OSQP_NULL,
                             a1.data(), a2.data(), d.data(), 0, OSQP_NULL) == 0);
  REQUIRE(osqp_solve(solver.get()) == 0);

  std::vector<OSQPFloat> x_ref(solver->solution->x, solver->solution->x + data->n);
  const OSQPFloat obj_ref = solver->info->obj_val;

  for (OSQPInt type : {OSQP_PENALTY_NORM2, OSQP_PENALTY_NORMINF}) {
    solver.reset();
    setup_solver();

    std::vector<OSQPInt> types(data->m, type);
    std::vector<OSQPInt> gid(data->m);

    /* Every row its own group */
    for (OSQPInt i = 0; i < data->m; i++) gid[i] = i;

    REQUIRE(osqp_setup_penalty(solver.get(), OSQP_PENALTY_NONE, types.data(),
                               a1.data(), a2.data(), d.data(),
                               data->m, gid.data()) == 0);
    REQUIRE(osqp_solve(solver.get()) == 0);

    for (OSQPInt i = 0; i < data->n; i++)
      REQUIRE(solver->solution->x[i] == Approx(x_ref[i]).margin(COMPARE_TOL));

    REQUIRE(solver->info->obj_val == Approx(obj_ref).margin(COMPARE_TOL));
  }
}

TEST_CASE_METHOD(group_test_fixture, "Group penalty: the infinity norm matches its epigraph lift", "[penalty],[group]")
{
  /* alpha*||xi||_inf is LP-representable: introduce t with -t <= xi_i <= t and
     charge alpha*t. Solving that hard QP with stock OSQP gives a reference
     that shares no code with the penalty path. */
  const OSQPFloat alpha = 1.5;

  setup_solver();
  REQUIRE(setup_one_group(OSQP_PENALTY_NORMINF, alpha) == 0);
  REQUIRE(osqp_solve(solver.get()) == 0);

  std::vector<OSQPFloat> x_soft(solver->solution->x, solver->solution->x + data->n);
  const OSQPFloat obj_soft = solver->info->obj_val;

  /* Lifted variables: (x, xi_0..xi_{m-1}, t) */
  const OSQPInt n_lift = data->n + data->m + 1;
  const OSQPInt m_lift = data->m + 2 * data->m;

  std::vector<OSQPFloat> P_dense(n_lift * n_lift, 0.0);
  for (OSQPInt i = 0; i < data->n; i++) P_dense[i * n_lift + i] = 1.0;

  std::vector<OSQPFloat> q_lift(n_lift, 0.0);
  for (OSQPInt i = 0; i < data->n; i++) q_lift[i] = data->q[i];
  q_lift[n_lift - 1] = alpha;                       /* alpha * t */

  std::vector<OSQPFloat> A_dense(m_lift * n_lift, 0.0);
  std::vector<OSQPFloat> l_lift(m_lift), u_lift(m_lift);

  /* l <= A x + xi <= u */
  for (OSQPInt i = 0; i < data->m; i++) {
    A_dense[i * n_lift + i]              = 1.0;     /* A is the identity here */
    A_dense[i * n_lift + data->n + i]    = 1.0;
    l_lift[i] = data->l[i];
    u_lift[i] = data->u[i];
  }

  /* xi_i - t <= 0 and -xi_i - t <= 0 */
  for (OSQPInt i = 0; i < data->m; i++) {
    OSQPInt rl = data->m + 2 * i;
    OSQPInt ru = data->m + 2 * i + 1;

    A_dense[rl * n_lift + data->n + i] =  1.0;
    A_dense[rl * n_lift + n_lift - 1]  = -1.0;
    l_lift[rl] = -OSQP_INFTY; u_lift[rl] = 0.0;

    A_dense[ru * n_lift + data->n + i] = -1.0;
    A_dense[ru * n_lift + n_lift - 1]  = -1.0;
    l_lift[ru] = -OSQP_INFTY; u_lift[ru] = 0.0;
  }

  /* Dense to CSC, dropping exact zeros */
  auto to_csc = [](OSQPInt m, OSQPInt n, const std::vector<OSQPFloat>& dense) {
    std::vector<OSQPFloat> x;
    std::vector<OSQPInt>   ind, ptr(n + 1, 0);

    for (OSQPInt j = 0; j < n; j++) {
      for (OSQPInt i = 0; i < m; i++) {
        OSQPFloat val = dense[i * n + j];
        if (val != 0.0) { x.push_back(val); ind.push_back(i); }
      }
      ptr[j + 1] = (OSQPInt)x.size();
    }

    OSQPCscMatrix* M = (OSQPCscMatrix*)c_malloc(sizeof(OSQPCscMatrix));
    OSQPFloat* xv = (OSQPFloat*)c_malloc(std::max<size_t>(x.size(), 1) * sizeof(OSQPFloat));
    OSQPInt*   iv = (OSQPInt*)  c_malloc(std::max<size_t>(ind.size(), 1) * sizeof(OSQPInt));
    OSQPInt*   pv = (OSQPInt*)  c_malloc((n + 1) * sizeof(OSQPInt));

    std::copy(x.begin(), x.end(), xv);
    std::copy(ind.begin(), ind.end(), iv);
    std::copy(ptr.begin(), ptr.end(), pv);

    OSQPCscMatrix_set_data(M, m, n, (OSQPInt)x.size(), xv, iv, pv);
    return M;
  };

  OSQPCscMatrix* P_lift = to_csc(n_lift, n_lift, P_dense);
  OSQPCscMatrix* A_lift = to_csc(m_lift, n_lift, A_dense);

  OSQPSolver* ref_solver = nullptr;
  REQUIRE(osqp_setup(&ref_solver, P_lift, q_lift.data(), A_lift,
                     l_lift.data(), u_lift.data(),
                     m_lift, n_lift, settings.get()) == 0);
  REQUIRE(osqp_solve(ref_solver) == 0);

  for (OSQPInt i = 0; i < data->n; i++)
    REQUIRE(x_soft[i] == Approx(ref_solver->solution->x[i]).margin(COMPARE_TOL));

  REQUIRE(obj_soft == Approx(ref_solver->info->obj_val).margin(COMPARE_TOL));

  osqp_cleanup(ref_solver);
  c_free(P_lift->x); c_free(P_lift->i); c_free(P_lift->p); c_free(P_lift);
  c_free(A_lift->x); c_free(A_lift->i); c_free(A_lift->p); c_free(A_lift);
}

TEST_CASE_METHOD(group_test_fixture, "Group penalty: the Euclidean norm satisfies its KKT conditions", "[penalty],[group]")
{
  /* alpha*||xi||_2 is not QP-representable, so the solution is checked against
     the optimality conditions instead: stationarity, and a dual that lies in
     alpha times the subdifferential of the norm at the slack. */
  const OSQPFloat alpha = 1.5;

  settings->polishing = 0;
  setup_solver();

  REQUIRE(setup_one_group(OSQP_PENALTY_NORM2, alpha) == 0);
  REQUIRE(osqp_solve(solver.get()) == 0);

  std::vector<OSQPFloat> x(solver->solution->x, solver->solution->x + data->n);
  std::vector<OSQPFloat> y(solver->solution->y, solver->solution->y + data->m);

  /* A and P are the identity, so stationarity is x + q + y = 0 */
  for (OSQPInt i = 0; i < data->n; i++)
    REQUIRE(x[i] + data->q[i] + y[i] == Approx(0.0).margin(COMPARE_TOL));

  /* xi = -(Ax - Proj(Ax)) */
  std::vector<OSQPFloat> xi(data->m);
  for (OSQPInt i = 0; i < data->m; i++) {
    OSQPFloat z = x[i];
    xi[i] = std::min(std::max(z, data->l[i]), data->u[i]) - z;
  }

  const OSQPFloat nrm = norm2(xi);

  mu_assert("The group slack should be active for this data", nrm > COMPARE_TOL);

  /* On a nonzero slack the subdifferential is the single point
     alpha*xi/||xi||, and y must match it up to sign convention. This also
     pins ||y||_2 to alpha, i.e. the dual sits on its ball. */
  for (OSQPInt i = 0; i < data->m; i++)
    REQUIRE(std::abs(y[i]) == Approx(alpha * std::abs(xi[i]) / nrm).margin(COMPARE_TOL));
}

TEST_CASE_METHOD(group_test_fixture, "Group penalty: objective and conjugate", "[penalty],[group]")
{
  settings->scaling = 0;
  setup_solver();

  const OSQPFloat alpha = 2.0;
  std::vector<OSQPFloat> z = {3.0, -2.0, 0.5, 1.0};
  std::vector<OSQPFloat> s(data->m);

  for (OSQPInt i = 0; i < data->m; i++)
    s[i] = z[i] - std::min(std::max(z[i], data->l[i]), data->u[i]);

  OSQPVectorf_ptr zvec{OSQPVectorf_new(z.data(), data->m)};

  SECTION("Euclidean norm") {
    REQUIRE(setup_one_group(OSQP_PENALTY_NORM2, alpha) == 0);
    REQUIRE(penalty_obj_value(solver.get(), zvec.get()) ==
            Approx(alpha * norm2(s)).epsilon(COMPARE_TOL));

    /* Phi* is the indicator of the dual ball, here ||y||_2 <= alpha */
    std::vector<OSQPFloat> y_in  = {alpha / 2, 0.0, 0.0, 0.0};
    std::vector<OSQPFloat> y_out = {alpha, alpha, 0.0, 0.0};

    OSQPVectorf_ptr in{OSQPVectorf_new(y_in.data(), data->m)};
    OSQPVectorf_ptr out{OSQPVectorf_new(y_out.data(), data->m)};

    REQUIRE(penalty_conj_value(solver.get(), in.get()) == Approx(0.0).margin(TESTS_TOL));
    mu_assert("The conjugate accepts a dual outside the 2-norm ball",
              penalty_conj_value(solver.get(), out.get()) >= OSQP_INFTY);
  }

  SECTION("Infinity norm") {
    REQUIRE(setup_one_group(OSQP_PENALTY_NORMINF, alpha) == 0);
    REQUIRE(penalty_obj_value(solver.get(), zvec.get()) ==
            Approx(alpha * norminf(s)).epsilon(COMPARE_TOL));

    /* The dual of ||.||_inf is ||.||_1 */
    std::vector<OSQPFloat> y_in  = {alpha / 4, -alpha / 4, 0.0, 0.0};
    std::vector<OSQPFloat> y_out = {alpha, alpha / 2, 0.0, 0.0};

    OSQPVectorf_ptr in{OSQPVectorf_new(y_in.data(), data->m)};
    OSQPVectorf_ptr out{OSQPVectorf_new(y_out.data(), data->m)};

    REQUIRE(penalty_conj_value(solver.get(), in.get()) == Approx(0.0).margin(TESTS_TOL));
    mu_assert("The conjugate accepts a dual outside the 1-norm ball",
              penalty_conj_value(solver.get(), out.get()) >= OSQP_INFTY);
  }
}

TEST_CASE_METHOD(group_test_fixture, "Group penalty: layout validation", "[penalty],[group]")
{
  setup_solver();

  std::vector<OSQPFloat> a1(data->m, 1.0), a2(data->m, 0.0), d(data->m, 0.0);
  std::vector<OSQPInt>   types(data->m, OSQP_PENALTY_NORM2);
  std::vector<OSQPInt>   gid(data->m, 0);

  /* A rejected setup must leave the scaled problem data untouched, which the
     group rescale makes a live concern rather than a formality */
  auto snapshot = [&]() {
    auto l = to_vector(solver->work->data->l, data->m);
    auto u = to_vector(solver->work->data->u, data->m);
    auto E = to_vector(solver->work->scaling->E, data->m);

    l.insert(l.end(), u.begin(), u.end());
    l.insert(l.end(), E.begin(), E.end());
    return l;
  };

  const std::vector<OSQPFloat> before = snapshot();

  auto setup = [&](OSQPInt ngroups, const OSQPInt* g, const OSQPInt* t,
                   const OSQPFloat* w1, const OSQPFloat* w2, const OSQPFloat* wd) {
    OSQPInt flag = osqp_setup_penalty(solver.get(), OSQP_PENALTY_NONE, t,
                                      w1, w2, wd, ngroups, g);

    if (flag) {
      auto after = snapshot();

      for (size_t i = 0; i < before.size(); i++)
        REQUIRE(after[i] == Approx(before[i]).epsilon(1e-12));

      mu_assert("A rejected setup still allocated",
                solver->work->data->penalty == OSQP_NULL);
    }

    return flag;
  };

  SECTION("A group type outside any group is rejected") {
    mu_assert("An ungrouped NORM2 row was accepted",
              setup(0, OSQP_NULL, types.data(), a1.data(), a2.data(), d.data()) != 0);
  }

  SECTION("A grouped row must carry a group type") {
    std::vector<OSQPInt> mixed = types;
    mixed[2] = OSQP_PENALTY_L1L2;

    mu_assert("A grouped L1L2 row was accepted",
              setup(1, gid.data(), mixed.data(), a1.data(), a2.data(), d.data()) != 0);
  }

  SECTION("Rows of a group must agree") {
    std::vector<OSQPInt> mixed = types;
    mixed[3] = OSQP_PENALTY_NORMINF;

    mu_assert("A group mixing NORM2 and NORMINF was accepted",
              setup(1, gid.data(), mixed.data(), a1.data(), a2.data(), d.data()) != 0);

    std::vector<OSQPFloat> mixed_w = a1;
    mixed_w[1] = 2.0;

    mu_assert("A group with two different weights was accepted",
              setup(1, gid.data(), types.data(), mixed_w.data(), a2.data(), d.data()) != 0);
  }

  SECTION("A group weight must be finite and positive") {
    std::vector<OSQPFloat> zero(data->m, 0.0);

    mu_assert("A zero group weight was accepted",
              setup(1, gid.data(), types.data(), zero.data(), a2.data(), d.data()) != 0);

    std::vector<OSQPFloat> inf(data->m, OSQP_INFTY);

    mu_assert("An infinite group weight was accepted",
              setup(1, gid.data(), types.data(), inf.data(), a2.data(), d.data()) != 0);
  }

  SECTION("The unused weights must be left at zero") {
    std::vector<OSQPFloat> nonzero(data->m, 1.0);

    mu_assert("A group row with alpha2 set was accepted",
              setup(1, gid.data(), types.data(), a1.data(), nonzero.data(), d.data()) != 0);

    mu_assert("A group row with delta set was accepted",
              setup(1, gid.data(), types.data(), a1.data(), a2.data(), nonzero.data()) != 0);
  }

  SECTION("Membership must name a valid, non-empty group") {
    std::vector<OSQPInt> bad = gid;
    bad[0] = 5;

    mu_assert("An out-of-range group_id was accepted",
              setup(1, bad.data(), types.data(), a1.data(), a2.data(), d.data()) != 0);

    mu_assert("An empty group was accepted",
              setup(2, gid.data(), types.data(), a1.data(), a2.data(), d.data()) != 0);

    mu_assert("ngroups > 0 without group_id was accepted",
              setup(1, OSQP_NULL, types.data(), a1.data(), a2.data(), d.data()) != 0);
  }
}

TEST_CASE_METHOD(group_test_fixture, "Group penalty: rho is uniform over a group", "[penalty],[group]")
{
  /* A free row inside a group must not fall into the loose-bound class, or the
     group would straddle two rho values and the prox would be wrong. */
  data->l[2] = -OSQP_INFTY;
  data->u[2] =  OSQP_INFTY;

  settings->rho_is_vec = 1;
  setup_solver();

  REQUIRE(setup_one_group(OSQP_PENALTY_NORM2, 1.5) == 0);

  auto rho = to_vector(solver->work->rho_vec, data->m);

  for (OSQPInt i = 1; i < data->m; i++)
    REQUIRE(rho[i] == Approx(rho[0]).epsilon(1e-12));
}

TEST_CASE_METHOD(group_test_fixture, "Group penalty: scaling is constant on a group", "[penalty],[group]")
{
  /* A group norm only survives the constraint scaling when E is constant on
     the group, which scale_data arranges by equalizing it there. */
  OSQPFloat A_diag[4] = {1.0, 10.0, 100.0, 1000.0};

  OSQPCscMatrix_free(data->A);
  data->A = OSQPCscMatrix_diag_vec(data->m, data->n, A_diag);

  settings->scaling = 10;
  setup_solver();

  std::vector<OSQPFloat> a1(data->m, 1.0), a2(data->m, 0.0), d(data->m, 0.0);
  std::vector<OSQPInt>   types(data->m, OSQP_PENALTY_NORM2);
  std::vector<OSQPInt>   gid = {0, 0, OSQP_NO_GROUP, OSQP_NO_GROUP};

  types[2] = OSQP_PENALTY_NONE;
  types[3] = OSQP_PENALTY_NONE;

  REQUIRE(osqp_setup_penalty(solver.get(), OSQP_PENALTY_NONE, types.data(),
                             a1.data(), a2.data(), d.data(),
                             1, gid.data()) == 0);

  auto E = to_vector(solver->work->scaling->E, data->m);

  REQUIRE(E[0] == Approx(E[1]).epsilon(1e-10));

  /* The ungrouped rows keep whatever Ruiz gave them, which for a diagonal A
     spanning three decades is nothing like the group's value */
  mu_assert("Equalization flattened the whole of E",
            std::abs(E[2] - E[3]) > 1e-8);

  /* The scaled weights stay equal too, which is what keeps the group valid */
  auto scaled = to_vector(solver->work->data->penalty->alpha1, data->m);

  REQUIRE(scaled[0] == Approx(scaled[1]).epsilon(1e-10));
}
