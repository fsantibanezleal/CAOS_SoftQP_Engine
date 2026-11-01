/*
 * Test file to compile the generated workspaces from the test suite.
 */

#include <stdio.h>
#include "osqp.h"
#include "glob_opts.h"   /* c_absval */

#include "rho_is_vec_0_embedded_2_workspace.h"
#include "rho_is_vec_1_embedded_2_workspace.h"

#include "scaling_0_embedded_2_workspace.h"
#include "scaling_1_embedded_2_workspace.h"

#include "data_lp_embedded_2_workspace.h"
#include "data_nonconvex_2_embedded_2_workspace.h"
#include "data_unconstrained_embedded_2_workspace.h"

#include "data_penalty_uniform_embedded_2_workspace.h"
#include "data_penalty_mixed_embedded_2_workspace.h"
#include "data_penalty_group_embedded_2_workspace.h"

int main() {
  OSQPInt exitflag;

  printf( "Embedded test program for embedded mode 2 settings.\n");

  /*
   * rho_is_vec = 0
   */
  exitflag = osqp_solve( &rho_is_vec_0_embedded_2_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored on rho_is_vec_0: %s\n", osqp_error_message(exitflag));
    return (int)exitflag;
  } else {
    printf( "  Solved rho_is_vec_0 with no error.\n" );
  }


  /*
   * rho_is_vec = 1
   */
  exitflag = osqp_solve( &rho_is_vec_1_embedded_2_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored on rho_is_vec_1: %s\n", osqp_error_message(exitflag));
    return (int)exitflag;
  } else {
    printf( "  Solved rho_is_vec_1 with no error.\n" );
  }


  /*
   * scaling = 0
   */
  exitflag = osqp_solve( &scaling_0_embedded_2_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored on scaling_0: %s\n", osqp_error_message(exitflag));
    return (int)exitflag;
  } else {
    printf( "  Solved scaling_0 with no error.\n" );
  }


  /*
   * scaling = 1
   */
  exitflag = osqp_solve( &scaling_1_embedded_2_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored on scaling_1: %s\n", osqp_error_message(exitflag));
    return (int)exitflag;
  } else {
    printf( "  Solved scaling_1 with no error.\n" );
  }

  printf( "Embedded test program for embedded mode 1 data variations.\n");

  /*
   * Linear programming problem
   */
  exitflag = osqp_solve( &data_lp_embedded_2_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored on lp problem: %s\n", osqp_error_message(exitflag));
    return (int)exitflag;
  } else {
    printf( "  Solved lp problem with no error.\n" );
  }

  /*
   * Unconstrained problem
   */
  exitflag = osqp_solve( &data_unconstrained_embedded_2_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored on unconstrained problem: %s\n", osqp_error_message(exitflag));
    return (int)exitflag;
  } else {
    printf( "  Solved unconstrained problem with no error.\n" );
  }

  /*
   * Properly generated after convexification
   */
  exitflag = osqp_solve( &data_nonconvex_2_embedded_2_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored on non-nonconvex problem: %s\n", osqp_error_message(exitflag));
    return (int)exitflag;
  } else {
    printf( "  Solved non-nonconvex problem with no error.\n" );
  }

  
  /*
   * soft constraints, uniform penalty
   */
  exitflag = osqp_solve( &data_penalty_uniform_embedded_2_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored on data_penalty_uniform: %s\n", osqp_error_message(exitflag));
    return (int)exitflag;
  } else {
    /* The generated solver has to reproduce the library's answer: if codegen
       dropped the penalty it would silently solve the hard problem instead */
    OSQPFloat obj = data_penalty_uniform_embedded_2_solver.info->obj_val;
    OSQPFloat pen = data_penalty_uniform_embedded_2_solver.info->penalty_val;

    if( c_absval(obj - (OSQPFloat)0.332352941176) > 1e-6 ||
        c_absval(pen - (OSQPFloat)0.437820069204) > 1e-6 ) {
      printf( "  data_penalty_uniform disagrees with the library: obj %g, penalty %g\n",
              (double)obj, (double)pen );
      return 1;
    }

    printf( "  Solved data_penalty_uniform, penalty term %g\n", (double)pen );
  }


  /*
   * soft constraints, mixed penalty
   */
  exitflag = osqp_solve( &data_penalty_mixed_embedded_2_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored on data_penalty_mixed: %s\n", osqp_error_message(exitflag));
    return (int)exitflag;
  } else {
    /* The generated solver has to reproduce the library's answer: if codegen
       dropped the penalty it would silently solve the hard problem instead */
    OSQPFloat obj = data_penalty_mixed_embedded_2_solver.info->obj_val;
    OSQPFloat pen = data_penalty_mixed_embedded_2_solver.info->penalty_val;

    if( c_absval(obj - (OSQPFloat)0.326883116883) > 1e-6 ||
        c_absval(pen - (OSQPFloat)0.441971664699) > 1e-6 ) {
      printf( "  data_penalty_mixed disagrees with the library: obj %g, penalty %g\n",
              (double)obj, (double)pen );
      return 1;
    }

    printf( "  Solved data_penalty_mixed, penalty term %g\n", (double)pen );
  }


  /*
   * soft constraints, non-separable penalty groups
   */
  exitflag = osqp_solve( &data_penalty_group_embedded_2_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored on data_penalty_group: %s\n", osqp_error_message(exitflag));
    return (int)exitflag;
  } else {
    /* The group CSR and the sort scratch have to survive codegen: without them
       the generated solver would quietly treat the grouped rows as hard */
    OSQPFloat obj = data_penalty_group_embedded_2_solver.info->obj_val;
    OSQPFloat pen = data_penalty_group_embedded_2_solver.info->penalty_val;

    if( c_absval(obj - (OSQPFloat)2.214000000000) > 1e-6 ||
        c_absval(pen - (OSQPFloat)0.334000000000) > 1e-6 ) {
      printf( "  data_penalty_group disagrees with the library: obj %g, penalty %g\n",
              (double)obj, (double)pen );
      return 1;
    }

    printf( "  Solved data_penalty_group, penalty term %g\n", (double)pen );
  }

  return 0;
}
