#include <catch2/catch.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */
#include "penalty.h"     /* Phi*(y), which the duality gap is built on */

namespace {

/* Read an OSQPVectorf out of the (possibly device-side) backend */
std::vector<OSQPFloat> to_vector(const OSQPVectorf* vec, OSQPInt len) {
  std::vector<OSQPFloat> out(len);
  OSQPVectorf_to_raw(out.data(), vec);
  return out;
}

/* A small diagonal QP. The diagonal of A spans several orders of magnitude so
 * that the scaling produces a non-constant E, which makes the different powers
 * of E in the parameter scaling observable. */
class penalty_test_fixture : public OSQPTestFixture {
public:
  penalty_test_fixture() {
    OSQPFloat A_diag[4] = {1.0, 10.0, 100.0, 1000.0};

    data = std::make_unique<OSQPTestData>();

    data->n = 4;
    data->m = 4;

    data->P = OSQPCscMatrix_identity(data->n);
    data->A = OSQPCscMatrix_diag_vec(data->m, data->n, A_diag);

    data->q = (OSQPFloat*)c_malloc(data->n * sizeof(OSQPFloat));
    data->l = (OSQPFloat*)c_malloc(data->m * sizeof(OSQPFloat));
    data->u = (OSQPFloat*)c_malloc(data->m * sizeof(OSQPFloat));

    for (OSQPInt i = 0; i < data->n; i++) data->q[i] = 1.0;
    for (OSQPInt i = 0; i < data->m; i++) {
      data->l[i] = -1.0;
      data->u[i] = 1.0;
    }

    settings->verbose = 0;
  }

  void setup_solver() {
    OSQPInt exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                                  data->A, data->l, data->u,
                                  data->m, data->n, settings.get());
    solver.reset(tmpSolver);

    mu_assert("Penalty: setup error!", exitflag == 0);
  }

  OSQPPenaltyData* penalty() { return solver->work->data->penalty; }

  /* Weights are mandatory at setup; these are valid for every type */
  OSQPInt setup_penalty(OSQPInt default_type, const OSQPInt* type) {
    std::vector<OSQPFloat> a1(data->m, 1.0), a2(data->m, 1.0), d(data->m, 1.0);

    return osqp_setup_penalty(solver.get(), default_type, type,
                              a1.data(), a2.data(), d.data(), 0, OSQP_NULL);
  }
};

} // namespace


TEST_CASE_METHOD(penalty_test_fixture, "Penalty: allocated only on setup", "[penalty]")
{
  setup_solver();

  mu_assert("Penalty: allocated without being asked for",
            penalty() == OSQP_NULL);

  mu_assert("Penalty: setup failed",
            setup_penalty(OSQP_PENALTY_NONE, OSQP_NULL) == 0);
  mu_assert("Penalty: setup did not allocate", penalty() != OSQP_NULL);

  // Setting up twice is an error, the update functions are for reconfiguring
  mu_assert("Penalty: setup accepted twice",
            setup_penalty(OSQP_PENALTY_L1L2, OSQP_NULL) != 0);
}

TEST_CASE_METHOD(penalty_test_fixture, "Penalty: setup requires weights", "[penalty]")
{
  // A soft row is never left without weights, so all three arrays are required
  std::vector<OSQPFloat> w(data->m, 1.0);

  setup_solver();

  mu_assert("Penalty: setup accepted a missing alpha1",
            osqp_setup_penalty(solver.get(), OSQP_PENALTY_L1L2, OSQP_NULL,
                               OSQP_NULL, w.data(), w.data(), 0, OSQP_NULL) != 0);
  mu_assert("Penalty: setup accepted a missing alpha2",
            osqp_setup_penalty(solver.get(), OSQP_PENALTY_L1L2, OSQP_NULL,
                               w.data(), OSQP_NULL, w.data(), 0, OSQP_NULL) != 0);
  mu_assert("Penalty: setup accepted a missing delta",
            osqp_setup_penalty(solver.get(), OSQP_PENALTY_HUBER, OSQP_NULL,
                               w.data(), w.data(), OSQP_NULL, 0, OSQP_NULL) != 0);

  mu_assert("Penalty: allocated by a rejected call", penalty() == OSQP_NULL);
}

TEST_CASE_METHOD(penalty_test_fixture, "Penalty: type validation", "[penalty]")
{
  setup_solver();

  SECTION("Invalid default type") {
    mu_assert("Penalty: out-of-range default type accepted",
              setup_penalty(999, OSQP_NULL) != 0);
    mu_assert("Penalty: negative default type accepted",
              setup_penalty(-1, OSQP_NULL) != 0);
  }

  SECTION("Invalid per-row type") {
    OSQPInt types[4] = {OSQP_PENALTY_L1L2, OSQP_PENALTY_NONE,
                        42, OSQP_PENALTY_NONE};

    mu_assert("Penalty: out-of-range row type accepted",
              setup_penalty(OSQP_PENALTY_NONE, types) != 0);
  }

  // A rejected call must not have allocated anything
  mu_assert("Penalty: allocated by a rejected call", penalty() == OSQP_NULL);
}

TEST_CASE_METHOD(penalty_test_fixture, "Penalty: parameters before setup", "[penalty]")
{
  setup_solver();

  OSQPFloat alpha2[4] = {1.0, 1.0, 1.0, 1.0};

  mu_assert("Penalty: parameters accepted before setup",
            osqp_update_penalty_params(solver.get(), OSQP_NULL, alpha2, OSQP_NULL) != 0);
}

TEST_CASE_METHOD(penalty_test_fixture, "Penalty: parameter validation", "[penalty]")
{
  setup_solver();

  OSQPFloat alpha1[4] = {1.0, 1.0, 1.0, 1.0};
  OSQPFloat alpha2[4] = {1.0, 1.0, 1.0, 1.0};
  OSQPFloat delta[4]  = {1.0, 1.0, 1.0, 1.0};

  SECTION("L1L2") {
    REQUIRE(setup_penalty(OSQP_PENALTY_L1L2, OSQP_NULL) == 0);

    SECTION("Negative alpha1 is rejected") {
      alpha1[2] = -1.0;
      mu_assert("Penalty: negative alpha1 accepted",
                osqp_update_penalty_params(solver.get(), alpha1, alpha2, OSQP_NULL) != 0);
    }

    SECTION("Negative alpha2 is rejected") {
      alpha2[0] = -1e-12;
      mu_assert("Penalty: negative alpha2 accepted",
                osqp_update_penalty_params(solver.get(), alpha1, alpha2, OSQP_NULL) != 0);
    }

    SECTION("NaN is rejected") {
      // NB: not OSQP_NAN, which is a bit pattern cast to float and compares
      // as an ordinary positive number
      alpha1[1] = std::numeric_limits<OSQPFloat>::quiet_NaN();
      mu_assert("Penalty: NaN alpha1 accepted",
                osqp_update_penalty_params(solver.get(), alpha1, alpha2, OSQP_NULL) != 0);
    }

    SECTION("An all-zero row is the zero penalty, and is accepted") {
      // phi = 0 leaves the row unconstrained, which is a legitimate thing to
      // ask for and needs no special case anywhere
      alpha1[3] = 0.0;
      alpha2[3] = 0.0;
      mu_assert("Penalty: zero L1L2 penalty rejected",
                osqp_update_penalty_params(solver.get(), alpha1, alpha2, OSQP_NULL) == 0);
    }

    SECTION("An infinite weight is rejected") {
      alpha2[1] = OSQP_INFTY;
      mu_assert("Penalty: infinite alpha2 accepted",
                osqp_update_penalty_params(solver.get(), alpha1, alpha2, OSQP_NULL) != 0);
    }

    SECTION("alpha1 = 0 is a quadratic penalty, and is accepted") {
      for (int i = 0; i < 4; i++) alpha1[i] = 0.0;
      mu_assert("Penalty: quadratic (alpha1 = 0) rejected",
                osqp_update_penalty_params(solver.get(), alpha1, alpha2, OSQP_NULL) == 0);
    }

    SECTION("alpha2 = 0 is a pure L1 penalty, and is accepted") {
      for (int i = 0; i < 4; i++) alpha2[i] = 0.0;
      mu_assert("Penalty: L1 (alpha2 = 0) rejected",
                osqp_update_penalty_params(solver.get(), alpha1, alpha2, OSQP_NULL) == 0);
    }
  }

  SECTION("Huber") {
    REQUIRE(setup_penalty(OSQP_PENALTY_HUBER, OSQP_NULL) == 0);

    SECTION("Zero alpha1 is rejected") {
      alpha1[0] = 0.0;
      mu_assert("Penalty: zero Huber weight accepted",
                osqp_update_penalty_params(solver.get(), alpha1, OSQP_NULL, delta) != 0);
    }

    SECTION("Zero delta is rejected") {
      delta[1] = 0.0;
      mu_assert("Penalty: zero Huber delta accepted",
                osqp_update_penalty_params(solver.get(), alpha1, OSQP_NULL, delta) != 0);
    }

    SECTION("Positive weight and delta are accepted") {
      mu_assert("Penalty: valid Huber parameters rejected",
                osqp_update_penalty_params(solver.get(), alpha1, OSQP_NULL, delta) == 0);
    }
  }
}

TEST_CASE_METHOD(penalty_test_fixture, "Penalty: rejected updates change nothing", "[penalty]")
{
  setup_solver();
  OSQPInt types[4] = {OSQP_PENALTY_L1L2, OSQP_PENALTY_HUBER,
                      OSQP_PENALTY_NONE, OSQP_PENALTY_L1L2};
  OSQPFloat good[4] = {2.0, 3.0, 4.0, 5.0};
  REQUIRE(setup_penalty(OSQP_PENALTY_NONE, types) == 0);
  REQUIRE(osqp_update_penalty_params(solver.get(), good, good, good) == 0);

  const auto a1 = to_vector(penalty()->alpha1, data->m);
  const auto a2 = to_vector(penalty()->alpha2, data->m);
  const auto delta = to_vector(penalty()->delta, data->m);

  SECTION("Parameter update with one invalid row") {
    OSQPFloat changed[4] = {6.0, 7.0, 8.0, 9.0};
    OSQPFloat bad[4] = {6.0, 7.0, 8.0, -9.0};
    REQUIRE(osqp_update_penalty_params(solver.get(), changed, bad, changed) != 0);
  }
  SECTION("Type update with one invalid row") {
    OSQPInt bad[4] = {OSQP_PENALTY_NONE, OSQP_PENALTY_L1L2,
                      OSQP_PENALTY_HUBER, 999};
    REQUIRE(osqp_update_penalty_types(solver.get(), OSQP_PENALTY_HUBER, bad) != 0);
  }

  REQUIRE(to_vector(penalty()->alpha1, data->m) == a1);
  REQUIRE(to_vector(penalty()->alpha2, data->m) == a2);
  REQUIRE(to_vector(penalty()->delta, data->m) == delta);
  REQUIRE(penalty()->uniform == 0);
  REQUIRE(penalty()->default_penalty_type == OSQP_PENALTY_NONE);
  OSQPInt stored[4];
  OSQPVectori_to_raw(stored, penalty()->type);
  for (OSQPInt i = 0; i < data->m; ++i) REQUIRE(stored[i] == types[i]);
}

TEST_CASE_METHOD(penalty_test_fixture, "Penalty: NULL arguments leave parameters alone", "[penalty]")
{
  settings->scaling = 10;
  setup_solver();

  OSQPFloat alpha1[4] = {1.0, 2.0, 3.0, 4.0};
  OSQPFloat alpha2[4] = {5.0, 6.0, 7.0, 8.0};
  OSQPFloat alpha2b[4] = {9.0, 9.0, 9.0, 9.0};

  REQUIRE(setup_penalty(OSQP_PENALTY_L1L2, OSQP_NULL) == 0);
  REQUIRE(osqp_update_penalty_params(solver.get(), alpha1, alpha2, OSQP_NULL) == 0);

  const auto before = to_vector(penalty()->alpha1, data->m);
  const auto E = to_vector(solver->work->scaling->E, data->m);
  const auto c = solver->work->scaling->c;

  // Update only alpha2
  REQUIRE(osqp_update_penalty_params(solver.get(), OSQP_NULL, alpha2b, OSQP_NULL) == 0);

  std::vector<OSQPFloat> a1 = to_vector(penalty()->alpha1, data->m);
  std::vector<OSQPFloat> a2 = to_vector(penalty()->alpha2, data->m);

  for (OSQPInt i = 0; i < data->m; i++) {
    mu_assert("Penalty: alpha1 changed by a NULL argument", a1[i] == Approx(before[i]).epsilon(TESTS_TOL));
    mu_assert("Penalty: alpha2 not updated",
              a2[i] == Approx(c * alpha2b[i] / (E[i] * E[i])).epsilon(TESTS_TOL));
  }
}

TEST_CASE_METHOD(penalty_test_fixture, "Penalty: uniform and per-row agree", "[penalty]")
{
  /* type == NULL and a homogeneous type vector describe the same penalty but
   * take different dispatch paths in the kernels, so they must agree. */
  OSQPFloat alpha1[4] = {1.0, 2.0, 3.0, 4.0};
  OSQPFloat alpha2[4] = {5.0, 6.0, 7.0, 8.0};

  OSQPInt types[4] = {OSQP_PENALTY_L1L2, OSQP_PENALTY_L1L2,
                      OSQP_PENALTY_L1L2, OSQP_PENALTY_L1L2};

  setup_solver();

  REQUIRE(setup_penalty(OSQP_PENALTY_L1L2, OSQP_NULL) == 0);
  REQUIRE(osqp_update_penalty_params(solver.get(), alpha1, alpha2, OSQP_NULL) == 0);

  mu_assert("Penalty: uniform flag not set", penalty()->uniform == 1);

  std::vector<OSQPFloat> uniform_a1 = to_vector(penalty()->alpha1, data->m);
  std::vector<OSQPFloat> uniform_a2 = to_vector(penalty()->alpha2, data->m);

  /* Restate the same types per-row. No type changes, so the parameters must
   * survive; this also pins that the old types are read before the per-row
   * vector is overwritten underneath them. */
  REQUIRE(osqp_update_penalty_types(solver.get(), OSQP_PENALTY_NONE, types) == 0);

  mu_assert("Penalty: uniform flag not cleared", penalty()->uniform == 0);

  std::vector<OSQPFloat> perrow_a1 = to_vector(penalty()->alpha1, data->m);
  std::vector<OSQPFloat> perrow_a2 = to_vector(penalty()->alpha2, data->m);

  for (OSQPInt i = 0; i < data->m; i++) {
    mu_assert("Penalty: alpha1 differs between dispatch paths, or was reset",
              uniform_a1[i] == perrow_a1[i]);
    mu_assert("Penalty: alpha2 differs between dispatch paths, or was reset",
              uniform_a2[i] == perrow_a2[i]);
  }

  // Going back to uniform must restore the fast path
  REQUIRE(osqp_update_penalty_types(solver.get(), OSQP_PENALTY_L1L2, OSQP_NULL) == 0);
  mu_assert("Penalty: uniform flag not restored", penalty()->uniform == 1);
}

TEST_CASE_METHOD(penalty_test_fixture, "Penalty: a type change keeps the weights", "[penalty]")
{
  OSQPInt   types[4]  = {OSQP_PENALTY_L1L2, OSQP_PENALTY_L1L2,
                         OSQP_PENALTY_L1L2, OSQP_PENALTY_L1L2};
  OSQPFloat alpha1[4] = {1.0, 2.0, 3.0, 4.0};
  OSQPFloat alpha2[4] = {0.5, 1.5, 2.5, 3.5};

  settings->scaling = 10;
  setup_solver();

  REQUIRE(osqp_setup_penalty(solver.get(), OSQP_PENALTY_NONE, types,
                             alpha1, alpha2, alpha1, 0, OSQP_NULL) == 0);

  // Change the type of rows 1 and 2 only. The stored weights are valid for the
  // new types, so the call succeeds and leaves every weight untouched.
  types[1] = OSQP_PENALTY_HUBER;
  types[2] = OSQP_PENALTY_NONE;
  REQUIRE(osqp_update_penalty_types(solver.get(), OSQP_PENALTY_NONE, types) == 0);

  const auto a1 = to_vector(penalty()->alpha1, data->m);
  const auto a2 = to_vector(penalty()->alpha2, data->m);
  const auto d  = to_vector(penalty()->delta,  data->m);
  const auto E  = to_vector(solver->work->scaling->E, data->m);
  const auto c  = solver->work->scaling->c;

  /* Rows 0 and 3 stay L1L2. Row 1 changes to Huber, and row 2 becomes hard.
   * Every row must retain the caller's weights under its new scaling rule. */
  for (OSQPInt i : {0, 3}) {
    REQUIRE(a1[i] == Approx(c * alpha1[i] / E[i]).epsilon(TESTS_TOL));
    REQUIRE(a2[i] == Approx(c * alpha2[i] / (E[i] * E[i])).epsilon(TESTS_TOL));
  }
  REQUIRE(a1[1] == Approx(c * alpha1[1] / (E[1] * E[1])).epsilon(TESTS_TOL));
  REQUIRE(d[1] == Approx(alpha1[1] * E[1]).epsilon(TESTS_TOL));
  REQUIRE(a1[2] == alpha1[2]);
  REQUIRE(a2[2] == alpha2[2]);
  REQUIRE(d[2] == alpha1[2]);
}

TEST_CASE_METHOD(penalty_test_fixture, "Penalty: parameter scaling", "[penalty]")
{
  /* The scaled penalty is c*phi(s/E[j]), so alpha1 and alpha2 pick up
   * different powers of E. */
  OSQPFloat alpha1[4] = {1.5, 2.5, 3.5, 4.5};
  OSQPFloat alpha2[4] = {0.5, 1.5, 2.5, 3.5};
  OSQPFloat delta[4]  = {2.0, 4.0, 6.0, 8.0};

  OSQPInt types[4] = {OSQP_PENALTY_NONE,  OSQP_PENALTY_L1L2,
                      OSQP_PENALTY_HUBER, OSQP_PENALTY_L1L2};

  SECTION("With scaling") {
    settings->scaling = 10;
    setup_solver();

    REQUIRE(setup_penalty(OSQP_PENALTY_NONE, types) == 0);
    REQUIRE(osqp_update_penalty_params(solver.get(), alpha1, alpha2, delta) == 0);

    std::vector<OSQPFloat> E = to_vector(solver->work->scaling->E, data->m);
    OSQPFloat              c = solver->work->scaling->c;

    std::vector<OSQPFloat> a1 = to_vector(penalty()->alpha1, data->m);
    std::vector<OSQPFloat> a2 = to_vector(penalty()->alpha2, data->m);
    std::vector<OSQPFloat> d  = to_vector(penalty()->delta,  data->m);

    // Only meaningful if the scaling actually does something
    mu_assert("Penalty: scaling is trivial, test cannot distinguish E from E^2",
              E[0] != E[3]);

    // Row 1: L1L2
    mu_assert("Penalty: L1L2 alpha1 must scale by c/E",
              c_absval(a1[1] - c * alpha1[1] / E[1]) < TESTS_TOL * c_absval(a1[1]));
    mu_assert("Penalty: L1L2 alpha2 must scale by c/E^2",
              c_absval(a2[1] - c * alpha2[1] / (E[1] * E[1])) < TESTS_TOL * c_absval(a2[1]));

    // Row 3: L1L2 with a different E, which pins the power of E
    mu_assert("Penalty: L1L2 alpha1 must scale by c/E",
              c_absval(a1[3] - c * alpha1[3] / E[3]) < TESTS_TOL * c_absval(a1[3]));
    mu_assert("Penalty: L1L2 alpha2 must scale by c/E^2",
              c_absval(a2[3] - c * alpha2[3] / (E[3] * E[3])) < TESTS_TOL * c_absval(a2[3]));

    // Row 2: Huber, whose weight and transition point scale differently
    mu_assert("Penalty: Huber alpha1 must scale by c/E^2",
              c_absval(a1[2] - c * alpha1[2] / (E[2] * E[2])) < TESTS_TOL * c_absval(a1[2]));
    mu_assert("Penalty: Huber delta must scale by E",
              c_absval(d[2] - delta[2] * E[2]) < TESTS_TOL * c_absval(d[2]));
  }

  SECTION("Without scaling") {
    settings->scaling = 0;
    setup_solver();

    REQUIRE(setup_penalty(OSQP_PENALTY_NONE, types) == 0);
    REQUIRE(osqp_update_penalty_params(solver.get(), alpha1, alpha2, delta) == 0);

    std::vector<OSQPFloat> a1 = to_vector(penalty()->alpha1, data->m);
    std::vector<OSQPFloat> a2 = to_vector(penalty()->alpha2, data->m);
    std::vector<OSQPFloat> d  = to_vector(penalty()->delta,  data->m);

    for (OSQPInt i = 0; i < data->m; i++) {
      mu_assert("Penalty: alpha1 rescaled without scaling", a1[i] == alpha1[i]);
      mu_assert("Penalty: alpha2 rescaled without scaling", a2[i] == alpha2[i]);
      mu_assert("Penalty: delta rescaled without scaling",  d[i]  == delta[i]);
    }
  }
}

TEST_CASE_METHOD(penalty_test_fixture, "Penalty: scaled parameters survive a matrix update", "[penalty]")
{
  OSQPInt types[4] = {OSQP_PENALTY_NONE, OSQP_PENALTY_L1L2,
                      OSQP_PENALTY_HUBER, OSQP_PENALTY_L1L2};
  OSQPFloat alpha1[4] = {1.5, 2.5, 3.5, OSQP_INFTY / 2};
  OSQPFloat alpha2[4] = {0.5, 1.5, 2.5, 3.5};
  OSQPFloat delta[4] = {2.0, 4.0, 6.0, 8.0};

  settings->scaling = 10;
  setup_solver();
  REQUIRE(setup_penalty(OSQP_PENALTY_NONE, types) == 0);
  REQUIRE(osqp_update_penalty_params(solver.get(), alpha1, alpha2, delta) == 0);
  const auto old_E = to_vector(solver->work->scaling->E, data->m);
  REQUIRE(to_vector(penalty()->alpha1, data->m)[3] > OSQP_INFTY);

  // Also exercise a partial update while a finite scaled weight exceeds INFTY.
  REQUIRE(osqp_update_penalty_params(solver.get(), OSQP_NULL, alpha2, OSQP_NULL) == 0);
  OSQPFloat A_new[4] = {5.0, 5000.0, 0.05, 50.0};
  REQUIRE(osqp_update_data_mat(solver.get(), OSQP_NULL, OSQP_NULL, 0,
                               A_new, OSQP_NULL, 4) == 0);

  const auto E = to_vector(solver->work->scaling->E, data->m);
  const auto c = solver->work->scaling->c;
  REQUIRE(E != old_E);
  const auto a1 = to_vector(penalty()->alpha1, data->m);
  const auto a2 = to_vector(penalty()->alpha2, data->m);
  const auto d = to_vector(penalty()->delta, data->m);

  REQUIRE(a1[0] == alpha1[0]);
  REQUIRE(a2[0] == alpha2[0]);
  REQUIRE(d[0] == delta[0]);
  for (OSQPInt i : {1, 3}) {
    REQUIRE(a1[i] == Approx(c * alpha1[i] / E[i]).epsilon(TESTS_TOL));
    REQUIRE(a2[i] == Approx(c * alpha2[i] / (E[i] * E[i])).epsilon(TESTS_TOL));
  }
  REQUIRE(a1[2] == Approx(c * alpha1[2] / (E[2] * E[2])).epsilon(TESTS_TOL));
  REQUIRE(d[2] == Approx(delta[2] * E[2]).epsilon(TESTS_TOL));
}

TEST_CASE_METHOD(penalty_test_fixture, "Penalty: the conjugate admits a boundary dual", "[penalty]")
{
  /* The y-update leaves y in dphi(s), so a row whose penalty is linear at its
     slack lands y exactly on the boundary of dom phi* and overshoots it by
     roundoff. An exact membership test would call that solution dual
     infeasible and report an infinite duality gap, stalling the gap
     termination check; the domain is therefore tested with a tolerance. */
  settings->scaling = 0;   /* Weights stay in the caller's units */
  setup_solver();

  std::vector<OSQPFloat> a1(data->m), a2(data->m), d(data->m);
  std::vector<OSQPFloat> y(data->m);
  OSQPFloat boundary;

  auto conj_at = [&](OSQPFloat val) {
    for (OSQPInt i = 0; i < data->m; i++) y[i] = (i % 2) ? -val : val;

    OSQPVectorf_ptr vec{OSQPVectorf_new(y.data(), data->m)};
    return penalty_conj_value(solver.get(), vec.get());
  };

  SECTION("Pure L1") {
    /* phi* is the indicator of |y| <= alpha1 */
    boundary = 1.0;
    std::fill(a1.begin(), a1.end(), boundary);
    std::fill(a2.begin(), a2.end(), 0.0);
    std::fill(d.begin(), d.end(), 1.0);

    REQUIRE(osqp_setup_penalty(solver.get(), OSQP_PENALTY_L1L2, OSQP_NULL,
                               a1.data(), a2.data(), d.data(), 0, OSQP_NULL) == 0);

    REQUIRE(conj_at(boundary) == Approx(0.0).margin(TESTS_TOL));
    REQUIRE(conj_at(std::nextafter(boundary, (OSQPFloat)2.0)) == Approx(0.0).margin(TESTS_TOL));

    mu_assert("The conjugate accepts a dual well outside its domain",
              conj_at(1.5 * boundary) >= OSQP_INFTY);
  }

  SECTION("Huber") {
    /* phi* is finite on |y| <= alpha1*delta, where it takes alpha1*delta^2/2 */
    const OSQPFloat alpha1 = 2.0, delta = 0.5;

    boundary = alpha1 * delta;
    std::fill(a1.begin(), a1.end(), alpha1);
    std::fill(a2.begin(), a2.end(), 0.0);
    std::fill(d.begin(), d.end(), delta);

    REQUIRE(osqp_setup_penalty(solver.get(), OSQP_PENALTY_HUBER, OSQP_NULL,
                               a1.data(), a2.data(), d.data(), 0, OSQP_NULL) == 0);

    const OSQPFloat val = data->m * 0.5 * alpha1 * delta * delta;

    REQUIRE(conj_at(boundary) == Approx(val).epsilon(TESTS_TOL));
    /* Past the boundary the value is the boundary's, never more */
    REQUIRE(conj_at(std::nextafter(boundary, (OSQPFloat)2.0)) == Approx(val).epsilon(TESTS_TOL));

    mu_assert("The conjugate accepts a dual well outside its domain",
              conj_at(1.5 * boundary) >= OSQP_INFTY);
  }
}
