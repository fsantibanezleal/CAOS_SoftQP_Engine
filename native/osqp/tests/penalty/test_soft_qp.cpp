#include <catch2/catch.hpp>

#include <memory>
#include <vector>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

/* End-to-end checks of the z-update against a reference obtained by solving the
 * equivalent hard QP in lifted form with stock OSQP. The lift is exact, so the
 * two solutions must agree to solver tolerance rather than by inspection.
 *
 * Per soft row i the lift introduces, with the slack xi_i eliminated:
 *
 *   L1L2 : xi_i free, cost alpha1*|xi_i| + (alpha2/2)*xi_i^2
 *   HUBER: xi_i = a_i + b_i, cost alpha1*(a_i^2/2 + delta*|b_i|), which is
 *          exactly alpha1*h_delta(xi_i) after minimizing over the split
 *
 * and every |.| becomes an epigraph variable t with a pair of rows.
 */

namespace {

/* Solve and comparison accuracy, tracking the working precision */
#ifdef OSQP_USE_FLOAT
const OSQPFloat SOLVE_EPS   = 1e-5;
const OSQPFloat COMPARE_TOL = 1e-4;
#else
const OSQPFloat SOLVE_EPS   = 1e-9;
const OSQPFloat COMPARE_TOL = 1e-5;
#endif

const OSQPInt MAX_ITER = 20000;

#ifndef OSQP_USE_FLOAT
/* Tolerance of the double-precision cross-check of the stored references,
   which only has to sit above the reference solve's own residual */
const OSQPFloat REF_TOL = 1e-6;
#endif

/* CSC matrices built here own their arrays, unlike OSQPCscMatrix_ptr */
struct OwnedCsc_deleter {
  void operator()(OSQPCscMatrix* M) const {
    if (!M) return;
    c_free(M->x);
    c_free(M->i);
    c_free(M->p);
    c_free(M);
  }
};

using OwnedCsc_ptr = std::unique_ptr<OSQPCscMatrix, OwnedCsc_deleter>;

/* Column-compressed form of a dense row-major matrix, dropping exact zeros */
OwnedCsc_ptr to_csc(OSQPInt                       m,
                    OSQPInt                       n,
                    const std::vector<OSQPFloat>& dense) {

  std::vector<OSQPFloat> x;
  std::vector<OSQPInt>   i;
  std::vector<OSQPInt>   p(n + 1, 0);

  for (OSQPInt col = 0; col < n; col++) {
    for (OSQPInt row = 0; row < m; row++) {
      OSQPFloat val = dense[row * n + col];

      if (val != 0.0) {
        x.push_back(val);
        i.push_back(row);
      }
    }
    p[col + 1] = (OSQPInt)x.size();
  }

  OSQPFloat* Mx = (OSQPFloat*)c_malloc((x.size() + 1) * sizeof(OSQPFloat));
  OSQPInt*   Mi = (OSQPInt*)c_malloc((i.size() + 1) * sizeof(OSQPInt));
  OSQPInt*   Mp = (OSQPInt*)c_malloc((n + 1) * sizeof(OSQPInt));

  for (size_t k = 0; k < x.size(); k++) Mx[k] = x[k];
  for (size_t k = 0; k < i.size(); k++) Mi[k] = i[k];
  for (OSQPInt k = 0; k <= n; k++)      Mp[k] = p[k];

  return OwnedCsc_ptr{OSQPCscMatrix_new(m, n, (OSQPInt)x.size(), Mx, Mi, Mp)};
}

/* One row's penalty */
struct Penalty {
  OSQPInt   type;
  OSQPFloat alpha1;
  OSQPFloat alpha2;
  OSQPFloat delta;
};

/* How the soft problem is handed to the solver. The reference lift is the same
 * either way, so these pin the integration rather than the mathematics. */
struct Options {
  bool    uniform    = false;  ///< pass type == NULL instead of a type vector
  OSQPInt rho_is_vec = 1;
  bool    eq_row0    = false;  ///< make row 0 a hard equality
  OSQPInt polishing  = 1;
  enum osqp_linsys_solver_type linsys_solver = OSQP_UNKNOWN_SOLVER;
};

/* The shared problem: a strongly convex QP whose unconstrained minimizer is
 * (1, 1), so all three bounds are active with a nonzero multiplier and every
 * row's penalty actually binds. The weights below are small enough that the
 * soft solution is interior to the penalty, which check_against_lift asserts. */
struct Problem {
  static const OSQPInt n = 2;
  static const OSQPInt m = 3;

  std::vector<OSQPFloat> P_diag{2.0, 3.0};
  std::vector<OSQPFloat> q{-2.0, -3.0};
  std::vector<OSQPFloat> A{1.0, 0.0,
                           0.0, 1.0,
                           1.0, 1.0};
  std::vector<OSQPFloat> l{-0.5, -0.5, -0.4};
  std::vector<OSQPFloat> u{ 0.5,  0.5,  0.4};

  explicit Problem(const Options& opts = Options{}) {
    /* set_rho_vec classifies an equality row separately and gives it
       OSQP_RHO_EQ_OVER_RHO_INEQ times the rho of the others, so the prox then
       sees genuinely different rho values per row */
    if (opts.eq_row0) l[0] = u[0] = 0.3;
  }
};

/* Reference solution of a lifted problem, and the slack it eliminates.
 *
 * These are literals rather than a runtime solve because the lift is only
 * solvable to double accuracy: its epigraph block is degenerate enough that
 * stock OSQP stalls a little above 1e-3 on the primal residual in single
 * precision, no matter how many iterations it is given, which would cap the
 * comparison tolerance far above what the soft solve itself achieves.
 *
 * They cannot rot: the double build still solves the lift at runtime and
 * checks it against them, so any change to the problem or the lift fails
 * there. This mirrors the rest of the suite, where generate_problem.py
 * computes reference solutions in double and embeds them as data. */
struct Reference {
  OSQPFloat x[Problem::n];
  OSQPFloat y[Problem::m];
  OSQPFloat xi[Problem::m];
  OSQPFloat obj;              ///< primal objective, penalty term included
};

void tighten(OSQPSettings*  s,
             const Options& opts = Options{}) {

  s->verbose    = 0;
  s->eps_abs    = SOLVE_EPS;
  s->eps_rel    = SOLVE_EPS;
  s->max_iter   = MAX_ITER;
  s->polishing  = opts.polishing;
  s->rho_is_vec = opts.rho_is_vec;
  if (opts.linsys_solver != OSQP_UNKNOWN_SOLVER)
    s->linsys_solver = opts.linsys_solver;
  s->check_dualgap = 1;  /* Exercise the new gap as a termination criterion */
}

/* Solve the soft problem through the penalty API */
std::vector<OSQPFloat> solve_soft(const Problem&              prob,
                                  const std::vector<Penalty>& pen,
                                  const Options&              opts,
                                  std::vector<OSQPFloat>&     y_out,
                                  OSQPFloat*                  obj_out = OSQP_NULL,
                                  OSQPFloat*                  pen_out = OSQP_NULL,
                                  OSQPFloat*                  gap_out = OSQP_NULL,
                                  OSQPFloat*                  dual_out = OSQP_NULL) {

  OwnedCsc_ptr P = to_csc(prob.n, prob.n, {prob.P_diag[0], 0.0, 0.0, prob.P_diag[1]});
  OwnedCsc_ptr A = to_csc(prob.m, prob.n, prob.A);

  OSQPSettings_ptr settings{OSQPSettings_new()};
  tighten(settings.get(), opts);

  OSQPSolver* tmp = nullptr;
  OSQPInt exitflag = osqp_setup(&tmp, P.get(), prob.q.data(), A.get(),
                                prob.l.data(), prob.u.data(),
                                prob.m, prob.n, settings.get());
  OSQPSolver_ptr solver{tmp};
  mu_assert("Soft QP: setup error", exitflag == 0);

  std::vector<OSQPInt>   type(prob.m);
  std::vector<OSQPFloat> a1(prob.m), a2(prob.m), dl(prob.m);

  for (OSQPInt i = 0; i < prob.m; i++) {
    type[i] = pen[i].type;
    a1[i]   = pen[i].alpha1;
    a2[i]   = pen[i].alpha2;
    dl[i]   = pen[i].delta;

    /* A NULL type vector only means the same thing if the rows agree */
    if (opts.uniform)
      mu_assert("Soft QP: a uniform run needs one type", type[i] == type[0]);
  }

  mu_assert("Soft QP: penalty setup error",
            osqp_setup_penalty(solver.get(),
                               opts.uniform ? type[0] : OSQP_PENALTY_NONE,
                               opts.uniform ? OSQP_NULL : type.data(),
                               a1.data(), a2.data(), dl.data(), 0, OSQP_NULL) == 0);

  mu_assert("Soft QP: solve error", osqp_solve(solver.get()) == 0);
  mu_assert("Soft QP: not solved", solver->info->status_val == OSQP_SOLVED);

  std::vector<OSQPFloat> x(solver->solution->x, solver->solution->x + prob.n);
  y_out.assign(solver->solution->y, solver->solution->y + prob.m);

  if (obj_out) *obj_out = solver->info->obj_val;
  if (pen_out) *pen_out = solver->info->penalty_val;
  if (gap_out)  *gap_out  = solver->info->duality_gap;
  if (dual_out) *dual_out = solver->info->dual_obj_val;

  /* Polishing has to handle soft rows, not decline them: the reduced KKT
     carries a 1/kappa diagonal where a slack moves with its dual, and pins the
     dual where it does not. */
  if (opts.polishing)
    mu_assert("Soft QP: polishing did not succeed",
              solver->info->status_polish == OSQP_POLISH_SUCCESS);

  return x;
}

#ifndef OSQP_USE_FLOAT

/* Solve the equivalent hard QP in lifted form with stock OSQP. Only the double
 * build runs this; see Reference for why. */
std::vector<OSQPFloat> solve_lifted(const Problem&              prob,
                                    const std::vector<Penalty>& pen,
                                    std::vector<OSQPFloat>&     y_out,
                                    std::vector<OSQPFloat>&     xi_out,
                                    OSQPFloat*                  obj_out = OSQP_NULL) {

  const OSQPInt n = prob.n;
  const OSQPInt m = prob.m;

  /* Column of the slack (xi for L1L2, the quadratic part a for Huber), of the
   * Huber linear part b, and of the epigraph variable t, or -1 if absent */
  std::vector<OSQPInt> col_s(m, -1), col_b(m, -1), col_t(m, -1);

  OSQPInt N = n;

  for (OSQPInt i = 0; i < m; i++) {
    if (pen[i].type == OSQP_PENALTY_NONE) continue;

    col_s[i] = N++;

    if (pen[i].type == OSQP_PENALTY_HUBER) col_b[i] = N++;

    /* An L1 term is needed for a nonzero alpha1 on L1L2 and always on Huber */
    if ((pen[i].type == OSQP_PENALTY_HUBER) ||
        (pen[i].alpha1 > 0.0)) col_t[i] = N++;
  }

  OSQPInt M = m;
  for (OSQPInt i = 0; i < m; i++) if (col_t[i] >= 0) M += 2;

  std::vector<OSQPFloat> Pd((size_t)N * N, 0.0);
  std::vector<OSQPFloat> qd((size_t)N, 0.0);
  std::vector<OSQPFloat> Ad((size_t)M * N, 0.0);
  std::vector<OSQPFloat> ld((size_t)M), ud((size_t)M);

  for (OSQPInt j = 0; j < n; j++) {
    Pd[(size_t)j * N + j] = prob.P_diag[j];
    qd[j]                 = prob.q[j];
  }

  for (OSQPInt i = 0; i < m; i++) {
    for (OSQPInt j = 0; j < n; j++) Ad[(size_t)i * N + j] = prob.A[(size_t)i * n + j];

    ld[i] = prob.l[i];
    ud[i] = prob.u[i];

    if (col_s[i] >= 0) Ad[(size_t)i * N + col_s[i]] = 1.0;
    if (col_b[i] >= 0) Ad[(size_t)i * N + col_b[i]] = 1.0;

    if (pen[i].type == OSQP_PENALTY_L1L2) {
      Pd[(size_t)col_s[i] * N + col_s[i]] = pen[i].alpha2;
      if (col_t[i] >= 0) qd[col_t[i]] = pen[i].alpha1;
    }
    else if (pen[i].type == OSQP_PENALTY_HUBER) {
      Pd[(size_t)col_s[i] * N + col_s[i]] = pen[i].alpha1;
      qd[col_t[i]]                        = pen[i].alpha1 * pen[i].delta;
    }
  }

  /* The epigraph rows: -t <= zeta <= t, on xi for L1L2 and on b for Huber */
  OSQPInt row = m;
  for (OSQPInt i = 0; i < m; i++) {
    if (col_t[i] < 0) continue;

    OSQPInt zeta = (pen[i].type == OSQP_PENALTY_HUBER) ? col_b[i] : col_s[i];

    Ad[(size_t)row * N + zeta]     = 1.0;
    Ad[(size_t)row * N + col_t[i]] = -1.0;
    ld[row] = -OSQP_INFTY;
    ud[row] = 0.0;
    row++;

    Ad[(size_t)row * N + zeta]     = 1.0;
    Ad[(size_t)row * N + col_t[i]] = 1.0;
    ld[row] = 0.0;
    ud[row] = OSQP_INFTY;
    row++;
  }

  OwnedCsc_ptr P = to_csc(N, N, Pd);
  OwnedCsc_ptr A = to_csc(M, N, Ad);

  OSQPSettings_ptr settings{OSQPSettings_new()};
  tighten(settings.get(), Options{});

  OSQPSolver* tmp = nullptr;
  OSQPInt exitflag = osqp_setup(&tmp, P.get(), qd.data(), A.get(), ld.data(), ud.data(),
                                M, N, settings.get());
  OSQPSolver_ptr solver{tmp};
  mu_assert("Lifted QP: setup error", exitflag == 0);

  mu_assert("Lifted QP: solve error", osqp_solve(solver.get()) == 0);
  mu_assert("Lifted QP: not solved", solver->info->status_val == OSQP_SOLVED);

  std::vector<OSQPFloat> x(solver->solution->x, solver->solution->x + n);
  y_out.assign(solver->solution->y, solver->solution->y + m);

  if (obj_out) *obj_out = solver->info->obj_val;

  /* The eliminated slack, reassembled from the lift */
  xi_out.assign(m, 0.0);
  for (OSQPInt i = 0; i < m; i++) {
    if (col_s[i] >= 0) xi_out[i] += solver->solution->x[col_s[i]];
    if (col_b[i] >= 0) xi_out[i] += solver->solution->x[col_b[i]];
  }

  return x;
}

#endif /* ifndef OSQP_USE_FLOAT */

/* Check the soft solve against a stored reference. The double build also
 * recomputes that reference from the lift, so the literals stay honest. */
void check_against_lift(const std::vector<Penalty>& pen,
                        const Reference&            ref,
                        const Options&              opts = Options{}) {

  Problem prob{opts};
  std::vector<OSQPFloat> y_soft;
  OSQPFloat obj_soft = 0.0, pen_soft = 0.0, gap_soft = 0.0, dual_soft = 0.0;

  std::vector<OSQPFloat> x_soft = solve_soft(prob, pen, opts, y_soft, &obj_soft,
                                             &pen_soft, &gap_soft, &dual_soft);

#ifndef OSQP_USE_FLOAT
  std::vector<OSQPFloat> y_ref, xi_ref;
  OSQPFloat obj_ref = 0.0;
  std::vector<OSQPFloat> x_ref = solve_lifted(prob, pen, y_ref, xi_ref, &obj_ref);

  mu_assert("The stored reference objective no longer matches the lift",
            c_absval(obj_ref - ref.obj) < REF_TOL);

  mu_assert("The stored reference no longer matches the lifted problem",
            vec_norm_inf_diff(x_ref.data(), ref.x, prob.n) < REF_TOL);
  mu_assert("The stored reference dual no longer matches the lifted problem",
            vec_norm_inf_diff(y_ref.data(), ref.y, prob.m) < REF_TOL);
  mu_assert("The stored reference slack no longer matches the lifted problem",
            vec_norm_inf_diff(xi_ref.data(), ref.xi, prob.m) < REF_TOL);
#endif

  /* Without this the comparison is vacuous: weights large enough to drive every
     slack to zero make the soft problem identical to the hard one, and the prox
     is never exercised */
  for (OSQPInt i = 0; i < prob.m; i++) {
    if (pen[i].type == OSQP_PENALTY_NONE) continue;

    mu_assert("The penalty is inactive, so the comparison proves nothing",
              c_absval(ref.xi[i]) > 1e-3);
  }

  mu_assert("Soft solution differs from the lifted reference",
            vec_norm_inf_diff(x_soft.data(), ref.x, prob.n) < COMPARE_TOL);
  mu_assert("Soft dual differs from the lifted reference",
            vec_norm_inf_diff(y_soft.data(), ref.y, prob.m) < COMPARE_TOL);

  /* The lifted objective carries the penalty in its extra variables, so an
     obj_val missing Phi(R(z)) would not match it */
  mu_assert("Objective differs from the lifted reference",
            c_absval(obj_soft - ref.obj) < COMPARE_TOL);

  /* ... and the split reported in info has to add up */
  OSQPFloat quad = 0.0;
  for (OSQPInt j = 0; j < prob.n; j++)
    quad += 0.5 * prob.P_diag[j] * x_soft[j] * x_soft[j] + prob.q[j] * x_soft[j];

  mu_assert("penalty_val is not the penalty part of obj_val",
            c_absval((obj_soft - pen_soft) - quad) < COMPARE_TOL);

  /* The solve ran with check_dualgap on, so this also proves the gap is a
     usable termination criterion and not merely small after the fact */
  mu_assert("The duality gap does not close on a soft problem",
            c_absval(gap_soft) < COMPARE_TOL);

  /* The dual objective needs Phi*(y) to reach the primal one */
  mu_assert("The dual objective does not meet the primal one",
            c_absval(dual_soft - obj_soft) < COMPARE_TOL);
}

} // namespace


TEST_CASE("Soft QP: quadratic penalty", "[penalty],[solve]")
{
  // L1L2 with alpha1 == 0
  std::vector<Penalty> pen{{OSQP_PENALTY_L1L2, 0.0, 0.5, 1.0},
                           {OSQP_PENALTY_L1L2, 0.0, 0.5, 1.0},
                           {OSQP_PENALTY_L1L2, 0.0, 0.5, 1.0}};

  Reference ref{{0.687234042547, 0.776595744674},
                {0.093617021277, 0.138297872340, 0.531914893617},
                {-0.187234042549, -0.276595744677, -1.063829787231},
                -2.016489361696};

  SECTION("Every supported linear solver")
  {
    Options opts;
    opts.linsys_solver = GENERATE(filter(&isLinsysSupported,
                                  values({OSQP_DIRECT_SOLVER, OSQP_INDIRECT_SOLVER})));
    check_against_lift(pen, ref, opts);
  }

  // A homogeneous penalty has a second entry point: a NULL type vector takes
  // the specialized dispatch path, which must reach the same solution
  SECTION("Uniform (NULL) type vector")
  {
    Options opts;
    opts.uniform = true;
    check_against_lift(pen, ref, opts);
  }

  SECTION("Scalar rho")
  {
    Options opts;
    opts.rho_is_vec = 0;
    check_against_lift(pen, ref, opts);
  }
}

TEST_CASE("Soft QP: L1 penalty", "[penalty],[solve]")
{
  // L1L2 with alpha2 == 0, the exact-penalty case
  check_against_lift({{OSQP_PENALTY_L1L2, 0.1, 0.0, 1.0},
                      {OSQP_PENALTY_L1L2, 0.1, 0.0, 1.0},
                      {OSQP_PENALTY_L1L2, 0.1, 0.0, 1.0}},
                     {{0.8999999999999, 0.933333333333},
                      {0.100000000887, 0.100000000887, 0.099999999113},
                      {-0.399999997324, -0.433333330658, -1.433333},
                      -2.256666667068});
}

TEST_CASE("Soft QP: elastic net penalty", "[penalty],[solve]")
{
  check_against_lift({{OSQP_PENALTY_L1L2, 0.1, 0.5, 1.0},
                      {OSQP_PENALTY_L1L2, 0.1, 0.5, 1.0},
                      {OSQP_PENALTY_L1L2, 0.1, 0.5, 1.0}},
                     {{0.627659574469, 0.734042553190},
                      {0.163829787863, 0.217021277232, 0.580851063198},
                      {-0.127659574677, -0.234042553388, -0.961702},
                      -1.873936170210});
}

TEST_CASE("Soft QP: Huber penalty", "[penalty],[solve]")
{
  // The prox branch is selected by |s| against delta, so asserting the slack
  // against delta is what keeps each case in the regime it is named for
  SECTION("Linear branch")
  {
    const OSQPFloat delta = 0.05;

    Reference ref{{0.974999999994, 0.983333333325},
                  {0.024999999512, 0.024999999516, 0.025000000487},
                  {-0.474999998528, -0.483333331804, -1.558333334466},
                -2.437916666702};

    check_against_lift({{OSQP_PENALTY_HUBER, 0.5, 0.0, delta},
                        {OSQP_PENALTY_HUBER, 0.5, 0.0, delta},
                        {OSQP_PENALTY_HUBER, 0.5, 0.0, delta}}, ref);

    OSQPInt linear = 0;
    for (OSQPInt i = 0; i < Problem::m; i++)
      if (c_absval(ref.xi[i]) > delta) linear++;

    mu_assert("No row reaches the linear branch of the Huber prox", linear > 0);
  }

  SECTION("Quadratic branch")
  {
    // Larger than the largest slack the equivalent quadratic penalty produces
    const OSQPFloat delta = 2.0;

    // alpha1*h_delta collapses to (alpha1/2)*s^2 here, so this is the quadratic
    // case with alpha2 = 0.5 and must land on the same solution
    Reference ref{{0.687234042554, 0.776595744681},
                  {0.093617021276, 0.138297872340, 0.531914893617},
                  {-0.187234042554, -0.276595744681, -1.063829787234},
                -2.016489361913};

    check_against_lift({{OSQP_PENALTY_HUBER, 0.5, 0.0, delta},
                        {OSQP_PENALTY_HUBER, 0.5, 0.0, delta},
                        {OSQP_PENALTY_HUBER, 0.5, 0.0, delta}}, ref);

    for (OSQPInt i = 0; i < Problem::m; i++)
      mu_assert("A row left the quadratic branch of the Huber prox",
                c_absval(ref.xi[i]) <= delta);
  }
}

TEST_CASE("Soft QP: per-row penalty types", "[penalty],[solve]")
{
  // A hard equality, a quadratically softened row and an L1-softened one. The
  // equality gives row 0 a rho three orders of magnitude from the others, so a
  // prox handed the scalar rho by mistake disagrees with the reference here.
  Options opts;
  opts.eq_row0 = true;

  check_against_lift({{OSQP_PENALTY_NONE, 0.0, 0.0, 0.0},
                      {OSQP_PENALTY_L1L2, 0.0, 0.5, 1.0},
                      {OSQP_PENALTY_L1L2, 0.1, 0.0, 1.0}},
                     {{0.299999999999, 0.899999999970},
                      {1.299999999912, 0.200000000001, 0.100000000088},
                      {0.0, -0.400000000002, -0.799999999765},
                      -1.875000000023}, opts);

  // ... and with Huber in the mix
  check_against_lift({{OSQP_PENALTY_NONE,  0.0, 0.0, 0.0},
                      {OSQP_PENALTY_HUBER, 0.5, 0.0, 0.05},
                      {OSQP_PENALTY_L1L2,  0.1, 0.5, 1.0}},
                     {{0.299999999999, 0.835714285729},
                      {0.932142857218, 0.025000000031, 0.467857142782},
                      {0.0, -0.335714285781, -0.735714285531},
                      -1.752857143}, opts);
}

TEST_CASE("Soft QP: a large weight approaches the hard problem", "[penalty],[solve]")
{
  // Every prox degrades to zero as its weight grows, so penalty continuation is
  // well defined and its limit is the hard QP. Infinite weights are rejected --
  // a hard row is spelled OSQP_PENALTY_NONE -- so the limit is approached with
  // a large finite weight instead.
  const Reference hard{{0.04, 0.36}, {0.0, 0.0, 1.92}, {0.0, 0.0, 0.0}, -0.964};

  Options opts;
  Problem prob{opts};
  std::vector<OSQPFloat> y_soft;

#ifdef OSQP_USE_FLOAT
  const OSQPFloat big = 1e4;
  const OSQPFloat tol = 2e-3;
#else
  const OSQPFloat big = 1e8;
  const OSQPFloat tol = 1e-5;
#endif

  std::vector<OSQPFloat> x_soft = solve_soft(prob,
      {{OSQP_PENALTY_L1L2,  0.0, big, 1.0},
       {OSQP_PENALTY_HUBER, big, 0.0, 1.0},
       {OSQP_PENALTY_L1L2,  0.0, big, 1.0}}, opts, y_soft);

  mu_assert("A large weight does not approach the hard problem",
            vec_norm_inf_diff(x_soft.data(), hard.x, prob.n) < tol);
  mu_assert("A large weight does not approach the hard dual",
            vec_norm_inf_diff(y_soft.data(), hard.y, prob.m) < tol);
}
