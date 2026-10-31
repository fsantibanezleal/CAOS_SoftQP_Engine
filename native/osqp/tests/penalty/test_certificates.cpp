#include <catch2/catch.hpp>

#include <memory>
#include <vector>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

/* Chunk 4: the rho classification of soft rows, and the infeasibility
 * certificates once soft rows are treated as free. */

namespace {

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

OwnedCsc_ptr to_csc(OSQPInt                       m,
                    OSQPInt                       n,
                    const std::vector<OSQPFloat>& dense) {

  std::vector<OSQPFloat> x;
  std::vector<OSQPInt>   i;
  std::vector<OSQPInt>   p(n + 1, 0);

  for (OSQPInt col = 0; col < n; col++) {
    for (OSQPInt row = 0; row < m; row++) {
      OSQPFloat val = dense[row * n + col];

      if (val != 0.0) { x.push_back(val); i.push_back(row); }
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

std::vector<OSQPFloat> to_vector(const OSQPVectorf* vec, OSQPInt len) {
  std::vector<OSQPFloat> out(len);
  OSQPVectorf_to_raw(out.data(), vec);
  return out;
}

/* Weights are mandatory at setup; these are valid for every type */
OSQPInt setup_penalty(OSQPSolver*    solver,
                      OSQPInt        m,
                      OSQPInt        default_type,
                      const OSQPInt* type) {

  std::vector<OSQPFloat> a1(m, 1.0), a2(m, 1.0), d(m, 1.0);

  return osqp_setup_penalty(solver, default_type, type,
                            a1.data(), a2.data(), d.data(), 0, OSQP_NULL);
}

std::vector<OSQPInt> to_vector(const OSQPVectori* vec, OSQPInt len) {
  std::vector<OSQPInt> out(len);
  OSQPVectori_to_raw(out.data(), vec);
  return out;
}

} // namespace


TEST_CASE("Penalty: a soft row is never classified as an equality", "[penalty]")
{
  // Row 0 is an equality, row 1 a plain inequality, row 2 loose
  OwnedCsc_ptr P = to_csc(2, 2, {2.0, 0.0, 0.0, 2.0});
  OwnedCsc_ptr A = to_csc(3, 2, {1.0, 0.0,
                                 0.0, 1.0,
                                 1.0, 1.0});

  OSQPFloat q[2] = {-1.0, -1.0};
  OSQPFloat l[3] = {0.25, -0.5, -OSQP_INFTY};
  OSQPFloat u[3] = {0.25,  0.5,  OSQP_INFTY};

  OSQPSettings_ptr settings{OSQPSettings_new()};
  settings->verbose    = 0;
  settings->rho        = 0.1;
  settings->rho_is_vec = 1;
  settings->scaling    = 0;   /* so rho_vec is directly comparable to rho */

  OSQPSolver* tmp = nullptr;
  mu_assert("Certificates: setup error",
            osqp_setup(&tmp, P.get(), q, A.get(), l, u, 3, 2, settings.get()) == 0);
  OSQPSolver_ptr solver{tmp};

  const OSQPFloat rho     = solver->settings->rho;
  const OSQPFloat rho_eq  = OSQP_RHO_EQ_OVER_RHO_INEQ * rho;

  SECTION("All rows hard: the classification is unchanged")
  {
    auto ct = to_vector(solver->work->constr_type, 3);

    REQUIRE(ct[0] == 1);
    REQUIRE(ct[1] == 0);
    REQUIRE(ct[2] == -1);

    auto rv = to_vector(solver->work->rho_vec, 3);
    REQUIRE(rv[0] == Approx(rho_eq));
  }

  OSQPFloat a1[3] = {0.0, 0.0, 0.0};
  OSQPFloat a2[3] = {1.0, 1.0, 1.0};
  OSQPFloat dl[3] = {1.0, 1.0, 1.0};

  SECTION("Softening row 0 demotes it to an inequality")
  {
    OSQPInt type[3] = {OSQP_PENALTY_L1L2, OSQP_PENALTY_NONE, OSQP_PENALTY_NONE};

    mu_assert("Certificates: penalty setup error",
              setup_penalty(solver.get(), 3, OSQP_PENALTY_NONE, type) == 0);
    mu_assert("Certificates: penalty parameter error",
              osqp_update_penalty_params(solver.get(), a1, a2, dl) == 0);

    auto ct = to_vector(solver->work->constr_type, 3);
    auto rv = to_vector(solver->work->rho_vec, 3);

    mu_assert("A soft equality is still classified as an equality", ct[0] == 0);
    mu_assert("A soft equality still gets the rho boost", rv[0] == Approx(rho));

    // The hard rows are untouched
    REQUIRE(ct[1] == 0);
    REQUIRE(ct[2] == -1);
  }

  SECTION("A loose soft row stays loose, it does not become an inequality")
  {
    OSQPInt type[3] = {OSQP_PENALTY_NONE, OSQP_PENALTY_NONE, OSQP_PENALTY_L1L2};

    mu_assert("Certificates: penalty setup error",
              setup_penalty(solver.get(), 3, OSQP_PENALTY_NONE, type) == 0);
    mu_assert("Certificates: penalty parameter error",
              osqp_update_penalty_params(solver.get(), a1, a2, dl) == 0);

    auto ct = to_vector(solver->work->constr_type, 3);

    mu_assert("A loose soft row was promoted to an inequality", ct[2] == -1);
    REQUIRE(ct[0] == 1);
  }

  SECTION("An infinite weight is rejected, not silently treated as hard")
  {
    // A hard row is spelled OSQP_PENALTY_NONE, so infinite weights have no
    // meaning and are refused rather than quietly reproducing one
    OSQPInt   type[3]   = {OSQP_PENALTY_L1L2, OSQP_PENALTY_NONE, OSQP_PENALTY_NONE};
    OSQPFloat a2_inf[3] = {OSQP_INFTY, 1.0, 1.0};

    mu_assert("An infinite weight was accepted at setup",
              osqp_setup_penalty(solver.get(), OSQP_PENALTY_NONE, type,
                                 a1, a2_inf, dl, 0, OSQP_NULL) != 0);
    mu_assert("A rejected setup still allocated",
              solver->work->data->penalty == OSQP_NULL);

    mu_assert("Certificates: penalty setup error",
              setup_penalty(solver.get(), 3, OSQP_PENALTY_NONE, type) == 0);
    mu_assert("An infinite weight was accepted on update",
              osqp_update_penalty_params(solver.get(), OSQP_NULL, a2_inf, OSQP_NULL) != 0);

    // The rejected call left the penalty alone
    REQUIRE(to_vector(solver->work->constr_type, 3)[0] == 0);
  }

  SECTION("A type change is rejected when the stored weights do not fit it")
  {
    // Changing a row's type used to reset its weights, which left the row in a
    // state no solve could use. It is now the caller's job to set weights that
    // suit the new type first.
    OSQPInt l1l2[3]  = {OSQP_PENALTY_L1L2, OSQP_PENALTY_NONE, OSQP_PENALTY_NONE};
    OSQPInt huber[3] = {OSQP_PENALTY_HUBER, OSQP_PENALTY_NONE, OSQP_PENALTY_NONE};

    OSQPFloat zero_w[3] = {0.0, 0.0, 0.0};

    mu_assert("Certificates: penalty setup error",
              osqp_setup_penalty(solver.get(), OSQP_PENALTY_NONE, l1l2,
                                 zero_w, a2, dl, 0, OSQP_NULL) == 0);

    // Huber needs alpha1 > 0, and row 0 currently carries alpha1 == 0
    mu_assert("A type change with unusable weights was accepted",
              osqp_update_penalty_types(solver.get(), OSQP_PENALTY_NONE, huber) != 0);

    // The rejected call left the types alone
    REQUIRE(to_vector(solver->work->constr_type, 3)[0] == 0);

    OSQPFloat a1_pos[3] = {1.0, 1.0, 1.0};
    mu_assert("Certificates: penalty parameter error",
              osqp_update_penalty_params(solver.get(), a1_pos, OSQP_NULL, dl) == 0);
    mu_assert("A type change with usable weights was rejected",
              osqp_update_penalty_types(solver.get(), OSQP_PENALTY_NONE, huber) == 0);
  }

}


TEST_CASE("Penalty: primal infeasibility only follows from hard rows", "[penalty],[solve]")
{
  /* x0 >= 1 and x0 <= -1 are jointly infeasible; row 2 is a bystander that the
     test softens or hardens. */
  OwnedCsc_ptr P = to_csc(1, 1, {1.0});
  OwnedCsc_ptr A = to_csc(3, 1, {1.0, 1.0, 1.0});

  OSQPFloat q[1] = {0.0};

  OSQPSettings_ptr settings{OSQPSettings_new()};
  settings->verbose = 0;
  settings->max_iter = 20000;

  SECTION("Infeasible hard rows are still reported, with a soft row present")
  {
    OSQPFloat l[3] = { 1.0, -OSQP_INFTY, -0.1};
    OSQPFloat u[3] = { OSQP_INFTY, -1.0,  0.1};

    OSQPSolver* tmp = nullptr;
    mu_assert("setup error",
              osqp_setup(&tmp, P.get(), q, A.get(), l, u, 3, 1, settings.get()) == 0);
    OSQPSolver_ptr solver{tmp};

    OSQPInt   type[3] = {OSQP_PENALTY_NONE, OSQP_PENALTY_NONE, OSQP_PENALTY_L1L2};
    OSQPFloat a1[3]   = {0.0, 0.0, 1.0};
    OSQPFloat a2[3]   = {0.0, 0.0, 1.0};
    OSQPFloat dl[3]   = {1.0, 1.0, 1.0};

    REQUIRE(setup_penalty(solver.get(), 3, OSQP_PENALTY_NONE, type) == 0);
    REQUIRE(osqp_update_penalty_params(solver.get(), a1, a2, dl) == 0);

    osqp_solve(solver.get());

    mu_assert("Hard infeasibility was masked by a soft row",
              (solver->info->status_val == OSQP_PRIMAL_INFEASIBLE ||
               solver->info->status_val == OSQP_PRIMAL_INFEASIBLE_INACCURATE));
  }

  SECTION("Infeasibility confined to soft rows is solved, not reported")
  {
    /* The same two conflicting rows, now both softened: the problem becomes
       feasible because the slacks absorb the conflict */
    OSQPFloat l[3] = { 1.0, -OSQP_INFTY, -0.1};
    OSQPFloat u[3] = { OSQP_INFTY, -1.0,  0.1};

    OSQPSolver* tmp = nullptr;
    mu_assert("setup error",
              osqp_setup(&tmp, P.get(), q, A.get(), l, u, 3, 1, settings.get()) == 0);
    OSQPSolver_ptr solver{tmp};

    OSQPInt   type[3] = {OSQP_PENALTY_L1L2, OSQP_PENALTY_L1L2, OSQP_PENALTY_NONE};
    OSQPFloat a1[3]   = {0.0, 0.0, 0.0};
    OSQPFloat a2[3]   = {1.0, 1.0, 0.0};
    OSQPFloat dl[3]   = {1.0, 1.0, 1.0};

    REQUIRE(setup_penalty(solver.get(), 3, OSQP_PENALTY_NONE, type) == 0);
    REQUIRE(osqp_update_penalty_params(solver.get(), a1, a2, dl) == 0);

    osqp_solve(solver.get());

    mu_assert("A problem infeasible only in its soft rows was declared infeasible",
              (solver->info->status_val == OSQP_SOLVED ||
               solver->info->status_val == OSQP_SOLVED_INACCURATE));
  }
}


TEST_CASE("Penalty: a hard problem reports no penalty", "[penalty],[solve]")
{
  // info->penalty_val must stay at zero for a solver that never sets a penalty
  OwnedCsc_ptr P = to_csc(2, 2, {2.0, 0.0, 0.0, 2.0});
  OwnedCsc_ptr A = to_csc(3, 2, {1.0, 0.0,
                                 0.0, 1.0,
                                 1.0, 1.0});

  OSQPFloat q[2] = {-1.0, -1.0};
  OSQPFloat l[3] = {0.25, -0.5, -OSQP_INFTY};
  OSQPFloat u[3] = {0.25,  0.5,  OSQP_INFTY};

  OSQPSettings_ptr settings{OSQPSettings_new()};
  settings->verbose = 0;

  OSQPSolver* tmp = nullptr;
  mu_assert("setup error",
            osqp_setup(&tmp, P.get(), q, A.get(), l, u, 3, 2, settings.get()) == 0);
  OSQPSolver_ptr solver{tmp};

  /* Prove the objective update writes the hard-path value rather than merely
     observing OSQPInfo's zero initialization. */
  solver->info->penalty_val = 1.0;
  osqp_solve(solver.get());

  REQUIRE(solver->info->status_val == OSQP_SOLVED);
  mu_assert("A hard problem reported a nonzero penalty",
            solver->info->penalty_val == 0.0);
}

TEST_CASE("Penalty: dual infeasibility and penalty growth", "[penalty],[solve]")
{
  /* min x subject to a single row, unbounded below along x -> -inf */
  OwnedCsc_ptr P = to_csc(1, 1, {0.0});
  OwnedCsc_ptr A = to_csc(1, 1, {1.0});

  OSQPFloat q[1] = {1.0};
  OSQPFloat l[1] = {-OSQP_INFTY};
  OSQPFloat u[1] = {1.0};

  OSQPSettings_ptr settings{OSQPSettings_new()};
  settings->verbose  = 0;
  settings->max_iter = 5000;

  SECTION("Unbounded with all rows hard")
  {
    OSQPSolver* tmp = nullptr;
    mu_assert("setup error",
              osqp_setup(&tmp, P.get(), q, A.get(), l, u, 1, 1, settings.get()) == 0);
    OSQPSolver_ptr solver{tmp};

    osqp_solve(solver.get());

    mu_assert("A genuinely unbounded problem was not detected",
              (solver->info->status_val == OSQP_DUAL_INFEASIBLE ||
               solver->info->status_val == OSQP_DUAL_INFEASIBLE_INACCURATE));
  }

  SECTION("A superlinear soft row still allows the declaration")
  {
    OSQPSolver* tmp = nullptr;
    mu_assert("setup error",
              osqp_setup(&tmp, P.get(), q, A.get(), l, u, 1, 1, settings.get()) == 0);
    OSQPSolver_ptr solver{tmp};

    // L1L2 with alpha2 > 0 grows superlinearly, so its recession function is
    // the same as a hard row's and the existing test stays valid
    OSQPInt   type[1] = {OSQP_PENALTY_L1L2};
    OSQPFloat a1[1]   = {0.0};
    OSQPFloat a2[1]   = {1.0};
    OSQPFloat dl[1]   = {1.0};

    REQUIRE(setup_penalty(solver.get(), 3, OSQP_PENALTY_NONE, type) == 0);
    REQUIRE(osqp_update_penalty_params(solver.get(), a1, a2, dl) == 0);

    osqp_solve(solver.get());

    mu_assert("A superlinear soft row suppressed a valid declaration",
              (solver->info->status_val == OSQP_DUAL_INFEASIBLE ||
               solver->info->status_val == OSQP_DUAL_INFEASIBLE_INACCURATE));
  }

}

TEST_CASE("Penalty: the dual-infeasibility rate is exact", "[penalty],[solve]")
{
  /* min -x  s.t.  x <= 1 softened. Along dx = +1 the objective falls at rate 1
     and the penalty charges its asymptotic slope, so the problem is unbounded
     exactly while that slope is below 1.
     The backend-level test pins the numerical rate; these cases verify that
     the solver uses it when deciding whether to issue a certificate. */
  OwnedCsc_ptr P = to_csc(1, 1, {0.0});
  OwnedCsc_ptr A = to_csc(1, 1, {1.0});

  OSQPFloat q[1] = {-1.0};
  OSQPFloat l[1] = {-OSQP_INFTY};
  OSQPFloat u[1] = {1.0};

  auto unbounded = [&](OSQPInt type_val, OSQPFloat a1v, OSQPFloat dv) {
    OSQPSettings_ptr settings{OSQPSettings_new()};
    settings->verbose  = 0;
    settings->max_iter = 5000;

    OSQPSolver* tmp = nullptr;
    REQUIRE(osqp_setup(&tmp, P.get(), q, A.get(), l, u, 1, 1, settings.get()) == 0);
    OSQPSolver_ptr solver{tmp};

    OSQPInt   type[1] = {type_val};
    OSQPFloat a1[1]   = {a1v};
    OSQPFloat a2[1]   = {0.0};
    OSQPFloat dl[1]   = {dv};

    REQUIRE(osqp_setup_penalty(solver.get(), OSQP_PENALTY_NONE, type,
                               a1, a2, dl, 0, OSQP_NULL) == 0);
    osqp_solve(solver.get());

    return (OSQPInt)(solver->info->status_val == OSQP_DUAL_INFEASIBLE ||
                     solver->info->status_val == OSQP_DUAL_INFEASIBLE_INACCURATE);
  };

  SECTION("L1: the slope is alpha1")
  {
    mu_assert("a slope just under the rate did not give dual infeasible",
              unbounded(OSQP_PENALTY_L1L2, 0.9, 1.0) == 1);
    mu_assert("a slope above the rate still gave dual infeasible",
              unbounded(OSQP_PENALTY_L1L2, 2.0, 1.0) == 0);
  }

  SECTION("Huber: the slope is alpha1*delta, not alpha1")
  {
    // alpha1 is 1 in both, so a rate using alpha1 alone would sit exactly on
    // the boundary for each and could not separate them
    mu_assert("a slope just under the rate did not give dual infeasible",
              unbounded(OSQP_PENALTY_HUBER, 1.0, 0.9) == 1);
    mu_assert("a slope above the rate still gave dual infeasible",
              unbounded(OSQP_PENALTY_HUBER, 1.0, 2.0) == 0);
  }
}
