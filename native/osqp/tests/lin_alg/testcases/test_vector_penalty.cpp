#include "test_lin_alg.h"

#include <limits>

TEST_CASE("Vector: Penalty scaling", "[vector],[operation],[penalty]")
{
  const OSQPInt n = 4;

  OSQPFloat a1_val[4] = {1.5, 2.5, 3.5, 4.5};
  OSQPFloat a2_val[4] = {0.5, 1.5, 0.0, 3.5};
  OSQPFloat d_val[4]  = {2.0, 4.0, 6.0, 8.0};
  OSQPFloat E_val[4]  = {0.5, 2.0, 8.0, 32.0};
  OSQPInt   type_val[4] = {OSQP_PENALTY_NONE,  OSQP_PENALTY_L1L2,
                           OSQP_PENALTY_HUBER, OSQP_PENALTY_L1L2};

  OSQPFloat c = 0.25;

  OSQPVectorf_ptr a1{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr a2{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr d{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr E{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr ref{OSQPVectorf_malloc(n)};
  OSQPVectori_ptr type{OSQPVectori_malloc(n)};

  OSQPVectorf_from_raw(E.get(), E_val);
  OSQPVectori_from_raw(type.get(), type_val);

  SECTION("Per-row types")
  {
    OSQPFloat ref_a1[4] = {1.5, 0.3125, 0.013671875, 0.03515625};
    OSQPFloat ref_a2[4] = {0.5, 0.09375, 0.0, 0.0008544921875};
    OSQPFloat ref_d[4] = {2.0, 4.0, 48.0, 8.0};

    OSQPVectorf_from_raw(a1.get(), a1_val);
    OSQPVectorf_from_raw(a2.get(), a2_val);
    OSQPVectorf_from_raw(d.get(),  d_val);

    OSQPVectorf_ew_scale_penalty(a1.get(), a2.get(), d.get(), type.get(),
                                 OSQP_PENALTY_NONE, c, E.get(), 0);

    OSQPVectorf_from_raw(ref.get(), ref_a1);
    mu_assert("Error scaling alpha1",
              OSQPVectorf_norm_inf_diff(ref.get(), a1.get()) < TESTS_TOL);

    OSQPVectorf_from_raw(ref.get(), ref_a2);
    mu_assert("Error scaling alpha2",
              OSQPVectorf_norm_inf_diff(ref.get(), a2.get()) < TESTS_TOL);

    OSQPVectorf_from_raw(ref.get(), ref_d);
    mu_assert("Error scaling delta",
              OSQPVectorf_norm_inf_diff(ref.get(), d.get()) < TESTS_TOL);
  }

  SECTION("A NULL type vector applies the default to every row")
  {
    OSQPVectorf_ptr b1{OSQPVectorf_malloc(n)};
    OSQPVectorf_ptr b2{OSQPVectorf_malloc(n)};
    OSQPVectorf_ptr bd{OSQPVectorf_malloc(n)};

    OSQPInt uniform[4] = {OSQP_PENALTY_HUBER, OSQP_PENALTY_HUBER,
                          OSQP_PENALTY_HUBER, OSQP_PENALTY_HUBER};
    OSQPVectori_ptr utype{OSQPVectori_malloc(n)};
    OSQPVectori_from_raw(utype.get(), uniform);

    OSQPVectorf_from_raw(a1.get(), a1_val);
    OSQPVectorf_from_raw(a2.get(), a2_val);
    OSQPVectorf_from_raw(d.get(),  d_val);
    OSQPVectorf_from_raw(b1.get(), a1_val);
    OSQPVectorf_from_raw(b2.get(), a2_val);
    OSQPVectorf_from_raw(bd.get(), d_val);

    OSQPVectorf_ew_scale_penalty(a1.get(), a2.get(), d.get(), OSQP_NULL,
                                 OSQP_PENALTY_HUBER, c, E.get(), 0);
    OSQPVectorf_ew_scale_penalty(b1.get(), b2.get(), bd.get(), utype.get(),
                                 OSQP_PENALTY_NONE, c, E.get(), 0);

    mu_assert("Default type differs from an explicit uniform vector",
              OSQPVectorf_norm_inf_diff(a1.get(), b1.get()) < TESTS_TOL);
    mu_assert("Default type differs from an explicit uniform vector",
              OSQPVectorf_norm_inf_diff(a2.get(), b2.get()) < TESTS_TOL);
    mu_assert("Default type differs from an explicit uniform vector",
              OSQPVectorf_norm_inf_diff(bd.get(), d.get()) < TESTS_TOL);
  }

  SECTION("Unscaling inverts scaling")
  {
    // Weights are required to be finite, so scaling is a plain multiply and
    // the round trip has to return the caller's values
    E_val[1] = 0.0625;
    OSQPVectorf_from_raw(E.get(), E_val);
    OSQPVectorf_from_raw(a1.get(), a1_val);
    OSQPVectorf_from_raw(a2.get(), a2_val);
    OSQPVectorf_from_raw(d.get(),  d_val);

    OSQPVectorf_ew_scale_penalty(a1.get(), a2.get(), d.get(), type.get(),
                                 OSQP_PENALTY_NONE, c, E.get(), 0);
    OSQPVectorf_ew_scale_penalty(a1.get(), a2.get(), d.get(), type.get(),
                                 OSQP_PENALTY_NONE, c, E.get(), 1);

    OSQPVectorf_from_raw(ref.get(), a1_val);
    mu_assert("Round trip changed alpha1",
              OSQPVectorf_norm_inf_diff(ref.get(), a1.get()) < TESTS_TOL);

    OSQPVectorf_from_raw(ref.get(), a2_val);
    mu_assert("Round trip changed alpha2",
              OSQPVectorf_norm_inf_diff(ref.get(), a2.get()) < TESTS_TOL);

    OSQPVectorf_from_raw(ref.get(), d_val);
    mu_assert("Round trip changed delta",
              OSQPVectorf_norm_inf_diff(ref.get(), d.get()) < TESTS_TOL);
  }

}

TEST_CASE("Vector: Quadratic penalties constrain recession directions",
          "[vector],[operation],[penalty]")
{
  OSQPFloat y_val[1]  = {1.0};
  OSQPFloat l_val[1]  = {-OSQP_INFTY};
  OSQPFloat u_val[1]  = {0.0};
  OSQPFloat a2_val[1] = {1.0};

  OSQPVectorf_ptr y{OSQPVectorf_new(y_val, 1)};
  OSQPVectorf_ptr l{OSQPVectorf_new(l_val, 1)};
  OSQPVectorf_ptr u{OSQPVectorf_new(u_val, 1)};
  OSQPVectorf_ptr a2{OSQPVectorf_new(a2_val, 1)};

  /* The direction violates the upper-bound recession cone. A quadratic soft
   * penalty preserves that cone, while a zero penalty makes the row free. */
  REQUIRE(OSQPVectorf_in_reccone(y.get(), l.get(), u.get(), a2.get(),
                                 OSQP_NULL, OSQP_PENALTY_L1L2,
                                 OSQP_INFTY, 0.0) == 0);

  OSQPVectorf_set_scalar(a2.get(), 0.0);
  REQUIRE(OSQPVectorf_in_reccone(y.get(), l.get(), u.get(), a2.get(),
                                 OSQP_NULL, OSQP_PENALTY_L1L2,
                                 OSQP_INFTY, 0.0) == 1);
}


TEST_CASE("Vector: Penalty prox", "[vector],[operation],[penalty]")
{
  const OSQPInt   n   = 6;
  const OSQPFloat rho = 2.0;

  // l == u pins the projection at zero, so the residual r is v itself and the
  // prox is exercised directly
  OSQPFloat v_val[6]    = {3.0, -3.0, 0.5, -0.5, 0.0, 7.0};
  OSQPFloat zero_val[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  OSQPVectorf_ptr z{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr v{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr l{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr u{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr a1{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr a2{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr d{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr ref{OSQPVectorf_malloc(n)};

  OSQPVectorf_from_raw(v.get(), v_val);
  OSQPVectorf_from_raw(l.get(), zero_val);
  OSQPVectorf_from_raw(u.get(), zero_val);

  SECTION("L1L2 is the quadratic prox at alpha1 == 0")
  {
    // rho/(rho + alpha2) * r
    OSQPFloat expected[6] = {2.0, -2.0, 1.0/3.0, -1.0/3.0, 0.0, 14.0/3.0};

    OSQPVectorf_set_scalar(a1.get(), 0.0);
    OSQPVectorf_set_scalar(a2.get(), 1.0);
    OSQPVectorf_set_scalar(d.get(),  1.0);

    OSQPVectorf_ew_prox_penalty(z.get(), v.get(), l.get(), u.get(),
                                OSQP_NULL, rho, a1.get(), a2.get(), d.get(),
                                OSQP_NULL, OSQP_PENALTY_L1L2);

    OSQPVectorf_from_raw(ref.get(), expected);
    mu_assert("Quadratic prox is wrong",
              OSQPVectorf_norm_inf_diff(ref.get(), z.get()) < TESTS_TOL);
  }

  SECTION("L1L2 is soft thresholding at alpha2 == 0")
  {
    // S_{alpha1/rho}(r)
    OSQPFloat expected[6] = {2.5, -2.5, 0.0, 0.0, 0.0, 6.5};

    OSQPVectorf_set_scalar(a1.get(), 1.0);
    OSQPVectorf_set_scalar(a2.get(), 0.0);
    OSQPVectorf_set_scalar(d.get(),  1.0);

    OSQPVectorf_ew_prox_penalty(z.get(), v.get(), l.get(), u.get(),
                                OSQP_NULL, rho, a1.get(), a2.get(), d.get(),
                                OSQP_NULL, OSQP_PENALTY_L1L2);

    OSQPVectorf_from_raw(ref.get(), expected);
    mu_assert("Soft thresholding prox is wrong",
              OSQPVectorf_norm_inf_diff(ref.get(), z.get()) < TESTS_TOL);
  }

  SECTION("L1L2 is the elastic net with both parameters")
  {
    // rho/(rho + alpha2) * S_{alpha1/rho}(r)
    OSQPFloat expected[6] = {5.0/3.0, -5.0/3.0, 0.0, 0.0, 0.0, 13.0/3.0};

    OSQPVectorf_set_scalar(a1.get(), 1.0);
    OSQPVectorf_set_scalar(a2.get(), 1.0);
    OSQPVectorf_set_scalar(d.get(),  1.0);

    OSQPVectorf_ew_prox_penalty(z.get(), v.get(), l.get(), u.get(),
                                OSQP_NULL, rho, a1.get(), a2.get(), d.get(),
                                OSQP_NULL, OSQP_PENALTY_L1L2);

    OSQPVectorf_from_raw(ref.get(), expected);
    mu_assert("Elastic net prox is wrong",
              OSQPVectorf_norm_inf_diff(ref.get(), z.get()) < TESTS_TOL);
  }

  SECTION("Huber prox, unweighted and weighted")
  {
    OSQPFloat expected[6]  = {2.5, -2.5, 1.0/3.0, -1.0/3.0, 0.0, 6.5};
    OSQPFloat expected2[6] = {2.0, -2.0, 0.25, -0.25, 0.0, 6.0};

    OSQPVectorf_set_scalar(a1.get(), 1.0);
    OSQPVectorf_set_scalar(a2.get(), 0.0);
    OSQPVectorf_set_scalar(d.get(),  1.0);

    OSQPVectorf_ew_prox_penalty(z.get(), v.get(), l.get(), u.get(),
                                OSQP_NULL, rho, a1.get(), a2.get(), d.get(),
                                OSQP_NULL, OSQP_PENALTY_HUBER);

    OSQPVectorf_from_raw(ref.get(), expected);
    mu_assert("Huber prox is wrong",
              OSQPVectorf_norm_inf_diff(ref.get(), z.get()) < TESTS_TOL);

    OSQPVectorf_set_scalar(a1.get(), 2.0);

    OSQPVectorf_ew_prox_penalty(z.get(), v.get(), l.get(), u.get(),
                                OSQP_NULL, rho, a1.get(), a2.get(), d.get(),
                                OSQP_NULL, OSQP_PENALTY_HUBER);

    OSQPVectorf_from_raw(ref.get(), expected2);
    mu_assert("Weighted Huber prox is wrong",
              OSQPVectorf_norm_inf_diff(ref.get(), z.get()) < TESTS_TOL);
  }

  SECTION("Each row uses its own rho")
  {
    OSQPFloat rhos[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    OSQPFloat expected[6] = {2.0, -2.5, 0.375, -0.4, 0.0, 41.0/6.0};
    OSQPVectorf_ptr rho_vec{OSQPVectorf_new(rhos, n)};
    OSQPVectorf_set_scalar(a1.get(), 1.0);
    OSQPVectorf_set_scalar(a2.get(), 0.0);
    OSQPVectorf_set_scalar(d.get(), 1.0);

    OSQPVectorf_ew_prox_penalty(z.get(), v.get(), l.get(), u.get(),
                                rho_vec.get(), 0.0, a1.get(), a2.get(), d.get(),
                                OSQP_NULL, OSQP_PENALTY_HUBER);
    OSQPVectorf_from_raw(ref.get(), expected);
    REQUIRE(OSQPVectorf_norm_inf_diff(ref.get(), z.get()) < TESTS_TOL);
  }

  SECTION("Large finite Huber weights preserve both prox branches")
  {
#ifdef OSQP_USE_FLOAT
    const OSQPFloat weight = 1e35, delta = 1e-30, small = 1e-31;
#else
    const OSQPFloat weight = 1e305, delta = 1e-300, small = 1e-301;
#endif
    OSQPFloat input[6] = {1e12, -1e12, 1e10, -1e10, 0.0, 2e12};
    OSQPFloat expected[6] = {9e11, -9e11, small, -small, 0.0, 1.9e12};
    OSQPFloat got[6];
    OSQPVectorf_from_raw(v.get(), input);
    OSQPVectorf_set_scalar(a1.get(), weight);
    OSQPVectorf_set_scalar(a2.get(), 0.0);
    OSQPVectorf_set_scalar(d.get(), delta);
    OSQPVectorf_ew_prox_penalty(z.get(), v.get(), l.get(), u.get(),
                                OSQP_NULL, 1e-6, a1.get(), a2.get(), d.get(),
                                OSQP_NULL, OSQP_PENALTY_HUBER);
    OSQPVectorf_to_raw(got, z.get());
    for (OSQPInt i = 0; i < n; ++i) {
      CAPTURE(i);
      // Normalize to check tiny finite results as well as the large ones.
      if (expected[i] == 0.0) REQUIRE(got[i] == 0.0);
      else REQUIRE(got[i] / expected[i] == Approx(1.0).epsilon(TESTS_TOL));
    }
    // The opposite scale: forming r/(rho+a1) first would overflow here.
    const OSQPFloat large_r = std::numeric_limits<OSQPFloat>::max() / 4;
    OSQPVectorf_set_scalar(v.get(), large_r);
    OSQPVectorf_set_scalar(a1.get(), 1e-6);
    OSQPVectorf_set_scalar(d.get(), large_r);   /* finite, but never reached */
    OSQPVectorf_ew_prox_penalty(z.get(), v.get(), l.get(), u.get(),
                                OSQP_NULL, 1e-6, a1.get(), a2.get(), d.get(),
                                OSQP_NULL, OSQP_PENALTY_HUBER);
    OSQPVectorf_to_raw(got, z.get());
    for (OSQPInt i = 0; i < n; ++i)
      REQUIRE(got[i] / large_r == Approx(0.5).epsilon(TESTS_TOL));
  }

  SECTION("Hard rows reproduce the plain projection")
  {
    OSQPFloat lo[6]  = {-1.0, -2.0, 0.0, -0.25, -3.0, 1.0};
    OSQPFloat hi[6]  = { 1.0,  2.0, 0.25, 0.25,  3.0, 5.0};
    OSQPFloat got[6], want[6];

    OSQPVectorf_from_raw(l.get(), lo);
    OSQPVectorf_from_raw(u.get(), hi);
    OSQPVectorf_set_scalar(a1.get(), 1.0);
    OSQPVectorf_set_scalar(a2.get(), 1.0);
    OSQPVectorf_set_scalar(d.get(),  1.0);

    OSQPVectorf_ew_bound_vec(ref.get(), v.get(), l.get(), u.get());
    OSQPVectorf_to_raw(want, ref.get());

    OSQPVectorf_ew_prox_penalty(z.get(), v.get(), l.get(), u.get(),
                                OSQP_NULL, rho, a1.get(), a2.get(), d.get(),
                                OSQP_NULL, OSQP_PENALTY_NONE);
    OSQPVectorf_to_raw(got, z.get());

    for (OSQPInt i = 0; i < n; i++)
      mu_assert("Hard rows are not bit-identical to the projection",
                got[i] == want[i]);

  }

  SECTION("A per-row type vector matches the uniform calls it mixes")
  {
    OSQPInt type_val[6] = {OSQP_PENALTY_NONE,  OSQP_PENALTY_L1L2,
                           OSQP_PENALTY_L1L2,  OSQP_PENALTY_HUBER,
                           OSQP_PENALTY_HUBER, OSQP_PENALTY_NONE};

    OSQPFloat lo[6] = {-1.0, -2.0, 0.0, -0.25, -3.0, 1.0};
    OSQPFloat hi[6] = { 1.0,  2.0, 0.25, 0.25,  3.0, 5.0};

    OSQPVectori_ptr type{OSQPVectori_malloc(n)};
    OSQPVectorf_ptr mixed{OSQPVectorf_malloc(n)};
    OSQPFloat       uniform[3][6];
    OSQPFloat       got[6];
    OSQPInt         i;

    OSQPVectori_from_raw(type.get(), type_val);
    OSQPVectorf_from_raw(l.get(), lo);
    OSQPVectorf_from_raw(u.get(), hi);
    OSQPVectorf_set_scalar(a1.get(), 1.5);
    OSQPVectorf_set_scalar(a2.get(), 0.75);
    OSQPVectorf_set_scalar(d.get(),  1.25);

    OSQPInt types[3] = {OSQP_PENALTY_NONE, OSQP_PENALTY_L1L2, OSQP_PENALTY_HUBER};

    for (i = 0; i < 3; i++) {
      OSQPVectorf_ew_prox_penalty(z.get(), v.get(), l.get(), u.get(),
                                  OSQP_NULL, rho, a1.get(), a2.get(), d.get(),
                                  OSQP_NULL, types[i]);
      OSQPVectorf_to_raw(uniform[i], z.get());
    }

    OSQPVectorf_ew_prox_penalty(mixed.get(), v.get(), l.get(), u.get(),
                                OSQP_NULL, rho, a1.get(), a2.get(), d.get(),
                                type.get(), OSQP_PENALTY_NONE);
    OSQPVectorf_to_raw(got, mixed.get());

    for (i = 0; i < n; i++) {
      OSQPInt k = (type_val[i] == OSQP_PENALTY_NONE) ? 0
                : (type_val[i] == OSQP_PENALTY_L1L2) ? 1 : 2;

      mu_assert("The per-row path disagrees with the uniform path",
                got[i] == uniform[k][i]);
    }
  }

  SECTION("z aliasing v is allowed")
  {
    OSQPVectorf_ptr inplace{OSQPVectorf_malloc(n)};

    OSQPVectorf_set_scalar(a1.get(), 1.0);
    OSQPVectorf_set_scalar(a2.get(), 1.0);
    OSQPVectorf_set_scalar(d.get(),  1.0);

    OSQPVectorf_copy(inplace.get(), v.get());

    OSQPVectorf_ew_prox_penalty(ref.get(), v.get(), l.get(), u.get(),
                                OSQP_NULL, rho, a1.get(), a2.get(), d.get(),
                                OSQP_NULL, OSQP_PENALTY_L1L2);
    OSQPVectorf_ew_prox_penalty(inplace.get(), inplace.get(), l.get(), u.get(),
                                OSQP_NULL, rho, a1.get(), a2.get(), d.get(),
                                OSQP_NULL, OSQP_PENALTY_L1L2);

    mu_assert("In-place prox differs from the out-of-place one",
              OSQPVectorf_norm_inf_diff(ref.get(), inplace.get()) < TESTS_TOL);
  }
}


TEST_CASE("Vector: Penalty value and conjugate", "[vector],[operation],[penalty]")
{
  const OSQPInt n = 6;

  // l == u again pins the projection, so R(z) is z itself
  OSQPFloat s_val[6]    = {1.0, -2.0, 0.5, 0.0, 0.0, 3.0};
  OSQPFloat zero_val[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  OSQPVectorf_ptr s{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr l{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr u{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr a1{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr a2{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr d{OSQPVectorf_malloc(n)};
  OSQPVectorf_ptr scratch{OSQPVectorf_malloc(1)};

  OSQPVectorf_from_raw(s.get(), s_val);
  OSQPVectorf_from_raw(l.get(), zero_val);
  OSQPVectorf_from_raw(u.get(), zero_val);

  SECTION("Objective contribution")
  {
    OSQPVectorf_set_scalar(a1.get(), 1.0);
    OSQPVectorf_set_scalar(a2.get(), 1.0);
    OSQPVectorf_set_scalar(d.get(),  1.0);

    // sum |s| + s^2/2
    mu_assert("L1L2 penalty value is wrong",
              c_absval(OSQPVectorf_penalty_value(s.get(), l.get(), u.get(),
                                                 a1.get(), a2.get(), d.get(),
                                                 OSQP_NULL, OSQP_PENALTY_L1L2,
                                                 scratch.get()) - 13.625) < TESTS_TOL);

    // sum h_1(s)
    mu_assert("Huber penalty value is wrong",
              c_absval(OSQPVectorf_penalty_value(s.get(), l.get(), u.get(),
                                                 a1.get(), a2.get(), d.get(),
                                                 OSQP_NULL, OSQP_PENALTY_HUBER,
                                                 scratch.get()) - 4.625) < TESTS_TOL);

    mu_assert("Hard rows contribute to the objective",
              OSQPVectorf_penalty_value(s.get(), l.get(), u.get(),
                                        a1.get(), a2.get(), d.get(),
                                        OSQP_NULL, OSQP_PENALTY_NONE,
                                        scratch.get()) == 0.0);
  }

  SECTION("Conjugate")
  {
    OSQPVectorf_set_scalar(a1.get(), 1.0);
    OSQPVectorf_set_scalar(a2.get(), 1.0);
    OSQPVectorf_set_scalar(d.get(),  4.0);

    // sum ((|y| - alpha1)_+)^2 / (2 alpha2)
    mu_assert("Elastic net conjugate is wrong",
              c_absval(OSQPVectorf_penalty_conj_value(s.get(), a1.get(), a2.get(),
                                                      d.get(), OSQP_NULL,
                                                      OSQP_PENALTY_L1L2,
                                                      scratch.get()) - 2.5) < TESTS_TOL);

    // sum y^2 / (2 alpha1) while |y| <= alpha1 * delta
    mu_assert("Huber conjugate is wrong",
              c_absval(OSQPVectorf_penalty_conj_value(s.get(), a1.get(), a2.get(),
                                                      d.get(), OSQP_NULL,
                                                      OSQP_PENALTY_HUBER,
                                                      scratch.get()) - 7.125) < TESTS_TOL);

    mu_assert("Hard rows contribute to the conjugate",
              OSQPVectorf_penalty_conj_value(s.get(), a1.get(), a2.get(),
                                             d.get(), OSQP_NULL,
                                             OSQP_PENALTY_NONE,
                                             scratch.get()) == 0.0);

    OSQPVectorf_set_scalar(d.get(), 1.0);
    mu_assert("Huber conjugate is finite outside its domain",
              OSQPVectorf_penalty_conj_value(s.get(), a1.get(), a2.get(),
                                             d.get(), OSQP_NULL,
                                             OSQP_PENALTY_HUBER,
                                             scratch.get()) >= OSQP_INFTY);

    OSQPVectorf_set_scalar(a1.get(), 0.0);
    OSQPVectorf_set_scalar(a2.get(), 2.0);
    mu_assert("Quadratic conjugate is wrong",
              c_absval(OSQPVectorf_penalty_conj_value(s.get(), a1.get(), a2.get(),
                                                      d.get(), OSQP_NULL,
                                                      OSQP_PENALTY_L1L2,
                                                      scratch.get()) - 3.5625) < TESTS_TOL);
  }

  SECTION("The alpha2 == 0 conjugate is an indicator, not a division")
  {
    // The general elastic net formula divides by alpha2, so pure L1 is a limit
    // rather than a substitution and needs its own branch.
    OSQPVectorf_set_scalar(a1.get(), 4.0);
    OSQPVectorf_set_scalar(a2.get(), 0.0);
    OSQPVectorf_set_scalar(d.get(),  1.0);

    mu_assert("Pure L1 conjugate is not zero inside its domain",
              OSQPVectorf_penalty_conj_value(s.get(), a1.get(), a2.get(),
                                             d.get(), OSQP_NULL,
                                             OSQP_PENALTY_L1L2,
                                             scratch.get()) == 0.0);

    OSQPVectorf_set_scalar(a1.get(), 2.5);
    mu_assert("Pure L1 conjugate is finite outside its domain",
              OSQPVectorf_penalty_conj_value(s.get(), a1.get(), a2.get(),
                                             d.get(), OSQP_NULL,
                                             OSQP_PENALTY_L1L2,
                                             scratch.get()) >= OSQP_INFTY);
  }

  SECTION("Conjugates attain y*s - phi(s) at known maximizers")
  {
    // Includes the L1 threshold, signed elastic-net maximizers, and Huber's
    // quadratic interior and both ends of its conjugate domain.
    struct Case { OSQPInt type; OSQPFloat a1, a2, d, y, maximizer, value; };
    const Case cases[] = {
      {OSQP_PENALTY_L1L2, 0.75, 1.5, 1.25, 0.5, 0.0, 0.0},
      {OSQP_PENALTY_L1L2, 0.75, 1.5, 1.25, 0.75, 0.0, 0.0},
      {OSQP_PENALTY_L1L2, 0.75, 1.5, 1.25, -2.25, -1.0, 0.75},
      {OSQP_PENALTY_L1L2, 0.75, 1.5, 1.25, 3.75, 2.0, 3.0},
      {OSQP_PENALTY_L1L2, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0},
      {OSQP_PENALTY_HUBER, 2.0, 0.0, 1.5, 1.0, 0.5, 0.25},
      {OSQP_PENALTY_HUBER, 2.0, 0.0, 1.5, 3.0, 1.5, 2.25},
      {OSQP_PENALTY_HUBER, 2.0, 0.0, 1.5, -3.0, -2.0, 2.25},
    };
    OSQPVectorf_ptr y{OSQPVectorf_calloc(n)};
    for (const auto& c : cases) {
      CAPTURE(c.type, c.y);
      OSQPFloat primal[6] = {c.maximizer, 0, 0, 0, 0, 0};
      OSQPFloat dual[6] = {c.y, 0, 0, 0, 0, 0};
      OSQPVectorf_from_raw(s.get(), primal);
      OSQPVectorf_from_raw(y.get(), dual);
      OSQPVectorf_set_scalar(a1.get(), c.a1);
      OSQPVectorf_set_scalar(a2.get(), c.a2);
      OSQPVectorf_set_scalar(d.get(), c.d);
      const auto conjugate = OSQPVectorf_penalty_conj_value(y.get(), a1.get(),
          a2.get(), d.get(), OSQP_NULL, c.type, scratch.get());
      const auto value = OSQPVectorf_penalty_value(s.get(), l.get(), u.get(),
          a1.get(), a2.get(), d.get(), OSQP_NULL, c.type, scratch.get());
      REQUIRE(conjugate == Approx(c.value).margin(TESTS_TOL));
      REQUIRE(value + conjugate == Approx(c.y * c.maximizer).margin(TESTS_TOL));
    }
  }

  SECTION("Conjugates avoid overflowing an intermediate square")
  {
#ifdef OSQP_USE_FLOAT
    const OSQPFloat large_y = 1e20, weight = 1e25;
#else
    const OSQPFloat large_y = 1e160, weight = 1e305;
#endif
    OSQPFloat dual[6] = {large_y, 0, 0, 0, 0, 0};
    OSQPVectorf_from_raw(s.get(), dual);
    OSQPVectorf_set_scalar(d.get(), 1.0);
    for (OSQPInt type : {OSQP_PENALTY_L1L2, OSQP_PENALTY_HUBER}) {
      CAPTURE(type);
      OSQPVectorf_set_scalar(a1.get(), type == OSQP_PENALTY_L1L2 ? 0.0 : weight);
      OSQPVectorf_set_scalar(a2.get(), weight);
      REQUIRE(OSQPVectorf_penalty_conj_value(s.get(), a1.get(), a2.get(),
          d.get(), OSQP_NULL, type, scratch.get()) == Approx(5e14).epsilon(TESTS_TOL));

    }
  }

  SECTION("Mixed values and conjugates include nonzero Huber rows")
  {
    OSQPInt types[6] = {OSQP_PENALTY_NONE, OSQP_PENALTY_L1L2,
                        OSQP_PENALTY_L1L2, OSQP_PENALTY_HUBER,
                        OSQP_PENALTY_HUBER, OSQP_PENALTY_NONE};
    OSQPFloat primal[6] = {99.0, -2.0, 0.5, 0.5, -2.0, -99.0};
    OSQPFloat dual[6] = {99.0, -2.0, 0.5, 0.5, -1.0, -99.0};
    OSQPVectori_ptr type{OSQPVectori_new(types, n)};
    OSQPVectorf_set_scalar(a1.get(), 1.0);
    OSQPVectorf_set_scalar(a2.get(), 1.0);
    OSQPVectorf_set_scalar(d.get(), 1.0);
    OSQPVectorf_from_raw(s.get(), primal);
    // L1L2: 4 + 0.625; Huber: 0.125 + 1.5. Hard rows add nothing.
    REQUIRE(OSQPVectorf_penalty_value(s.get(), l.get(), u.get(), a1.get(),
        a2.get(), d.get(), type.get(), OSQP_PENALTY_NONE, scratch.get())
        == Approx(6.25).margin(TESTS_TOL));
    OSQPVectorf_from_raw(s.get(), dual);
    // L1L2: 0.5 + 0; Huber: 0.125 + 0.5 (at the domain boundary).
    REQUIRE(OSQPVectorf_penalty_conj_value(s.get(), a1.get(), a2.get(), d.get(),
        type.get(), OSQP_PENALTY_NONE, scratch.get())
        == Approx(1.125).margin(TESTS_TOL));
    dual[4] = -1.25;
    OSQPVectorf_from_raw(s.get(), dual);
    REQUIRE(OSQPVectorf_penalty_conj_value(s.get(), a1.get(), a2.get(), d.get(),
        type.get(), OSQP_PENALTY_NONE, scratch.get()) == OSQP_INFTY);
  }
}

TEST_CASE("Vector: Penalty recession rate", "[vector],[operation],[penalty]")
{
  const OSQPInt n = 4;

  OSQPFloat w_val[4]  = {2.0, -3.0, 0.5, 0.1};
  OSQPFloat l_val[4]  = {-OSQP_INFTY, 0.0, 0.0, -OSQP_INFTY};
  OSQPFloat u_val[4]  = {0.0, OSQP_INFTY, OSQP_INFTY, 0.0};
  OSQPFloat a1_val[4] = {2.0, 3.0, 0.0, 2.0};
  OSQPFloat a2_val[4] = {0.0, 0.0, 0.0, 0.0};
  OSQPFloat d_val[4]  = {1.0, 0.5, 1.0, 1.0};
  OSQPInt type_val[4] = {OSQP_PENALTY_L1L2, OSQP_PENALTY_HUBER,
                         OSQP_PENALTY_L1L2, OSQP_PENALTY_L1L2};

  OSQPVectorf_ptr w{OSQPVectorf_new(w_val, n)};
  OSQPVectorf_ptr l{OSQPVectorf_new(l_val, n)};
  OSQPVectorf_ptr u{OSQPVectorf_new(u_val, n)};
  OSQPVectorf_ptr a1{OSQPVectorf_new(a1_val, n)};
  OSQPVectorf_ptr a2{OSQPVectorf_new(a2_val, n)};
  OSQPVectorf_ptr d{OSQPVectorf_new(d_val, n)};
  OSQPVectori_ptr type{OSQPVectori_new(type_val, n)};
  OSQPVectorf_ptr scratch{OSQPVectorf_malloc(1)};

  /* L1: 2*2, Huber: (3*0.5)*3, zero penalty: 0, small L1: 2*0.1.
     The last contribution remains exact even though it is below tol. */
  REQUIRE(OSQPVectorf_penalty_reccone_rate(w.get(), l.get(), u.get(),
      a1.get(), a2.get(), d.get(), type.get(), OSQP_PENALTY_NONE,
      OSQP_INFTY, 0.2, scratch.get()) == Approx(8.7).margin(TESTS_TOL));

  /* Making the final row quadratic turns it into a blocker. The certificate
     tolerance applies to membership of its recession cone. */
  a2_val[3] = 1.0;
  OSQPVectorf_from_raw(a2.get(), a2_val);
  REQUIRE(OSQPVectorf_penalty_reccone_rate(w.get(), l.get(), u.get(),
      a1.get(), a2.get(), d.get(), type.get(), OSQP_PENALTY_NONE,
      OSQP_INFTY, 0.2, scratch.get()) == Approx(8.5).margin(TESTS_TOL));
  REQUIRE(OSQPVectorf_penalty_reccone_rate(w.get(), l.get(), u.get(),
      a1.get(), a2.get(), d.get(), type.get(), OSQP_PENALTY_NONE,
      OSQP_INFTY, 0.05, scratch.get()) >= OSQP_INFTY);

  a2_val[3] = 0.0;
  type_val[3] = OSQP_PENALTY_NONE;
  OSQPVectorf_from_raw(a2.get(), a2_val);
  OSQPVectori_from_raw(type.get(), type_val);
  REQUIRE(OSQPVectorf_penalty_reccone_rate(w.get(), l.get(), u.get(),
      a1.get(), a2.get(), d.get(), type.get(), OSQP_PENALTY_NONE,
      OSQP_INFTY, 0.05, scratch.get()) >= OSQP_INFTY);
}
