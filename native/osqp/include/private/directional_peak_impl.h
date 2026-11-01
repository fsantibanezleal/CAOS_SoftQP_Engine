/* SoftQP directional coupled peak kernels, 2026-11-01.
 * SPDX-License-Identifier: Apache-2.0
 * Included once by penalty.c; CPU builtin, non-embedded implementation. */
#ifndef SOFTQP_DIRECTIONAL_PEAK_IMPL_H
#define SOFTQP_DIRECTIONAL_PEAK_IMPL_H
#include <stdlib.h>
#include <float.h>
#include <limits.h>

static int peak_compare(const void* a, const void* b) {
  OSQPFloat x = ((const OSQPFloat*)a)[0], y = ((const OSQPFloat*)b)[0];
  return x < y ? 1 : (x > y ? -1 : 0);
}

static OSQPFloat peak_ulp(OSQPFloat value) {
#ifdef OSQP_USE_FLOAT
  return value>0 ? value-nextafterf(value,0):nextafterf(0,1);
#else
  return value>0 ? value-nextafter(value,0):nextafter(0,1);
#endif
}

/* Each triple is (c*abs(r), rho*abs(r)/c, rho/c^2). No allocations in ADMM. */
static OSQPInt peak_threshold(OSQPFloat* terms, OSQPInt n,
                             OSQPFloat a, OSQPFloat b, OSQPFloat* threshold) {
  OSQPInt k; OSQPFloat sum_a=0, sum_b=b, root, lower, upper;
  if (!isfinite(a) || !isfinite(b) || a<0 || b<0 || n<=0) return 1;
  qsort(terms, (size_t)n, 3*sizeof(OSQPFloat), peak_compare);
  for (k=0;k<n;k++) {
    if (!isfinite(terms[3*k]) || !isfinite(terms[3*k+1]) ||
        !isfinite(terms[3*k+2]) || terms[3*k]<0 || terms[3*k+2]<=0) return 1;
  }
  if(a==0 && b==0) { *threshold=terms[0]; return 0; }
  for (k=0;k<n;k++) {
    sum_a += terms[3*k+1]; sum_b += terms[3*k+2];
    if (!isfinite(sum_a) || !isfinite(sum_b) || sum_b<=0) return 1;
    root=(sum_a-a)/sum_b;
    lower=k+1<n ? terms[3*(k+1)] : 0;
    upper=terms[3*k];
    if (!isfinite(root) || (sum_a>a && root==0)) return 1;
    if (root>=0 && root>=lower-16*peak_ulp(lower) && root<=upper+16*peak_ulp(upper)) {
      *threshold=c_min(c_max(root,lower),upper); return 0;
    }
  }
  if (sum_a<=a) { *threshold=0; return 0; }
  return 1;
}

static OSQPInt peak_term(OSQPFloat r, OSQPFloat cl, OSQPFloat cu,
                        OSQPFloat rho, OSQPFloat* term) {
  OSQPFloat c=r>=0 ? cu:cl, square=c*c;
  if (!isfinite(r) || !isfinite(c) || c<=0 || !isfinite(rho) || rho<=0 ||
      !isfinite(square) || square<=0) return 1;
  term[0]=c*c_absval(r); term[1]=(rho*c_absval(r))/c; term[2]=rho/square;
  if (!isfinite(term[0]) || !isfinite(term[1]) || !isfinite(term[2]) || term[2]<=0 ||
      (r!=0 && (term[0]==0 || term[1]==0))) return 1;
  return 0;
}

#ifndef OSQP_EMBEDDED_MODE
OSQPInt osqp_directional_peak_prox(OSQPInt n, const OSQPFloat* r,
 const OSQPFloat* cl, const OSQPFloat* cu, const OSQPFloat* rho,
 OSQPFloat a, OSQPFloat b, OSQPFloat* out, OSQPFloat* threshold) {
  OSQPInt i,status=0; OSQPFloat* terms;
  if (n<=0 || n>INT_MAX/3 || !r || !cl || !cu || !rho || !out || !threshold) return OSQP_DATA_VALIDATION_ERROR;
  terms=c_malloc(3*(size_t)n*sizeof(OSQPFloat));
  if (!terms) return OSQP_MEM_ALLOC_ERROR;
  for (i=0;i<n;i++) {
    if (!isfinite(cl[i]) || cl[i]<=0 || !isfinite(cu[i]) || cu[i]<=0 ||
        peak_term(r[i],cl[i],cu[i],rho[i],terms+3*i)) { status=1; break; }
  }
  if (!status) status=peak_threshold(terms,n,a,b,threshold);
  if (!status) for(i=0;i<n;i++) {
    OSQPFloat c=r[i]>=0 ? cu[i]:cl[i], magnitude=c_absval(r[i]), shrink=magnitude;
    if (c*magnitude>*threshold) shrink=*threshold/c;
    if (!isfinite(shrink) || (*threshold>0 && magnitude>0 && shrink==0)) { status=1; break; }
    out[i]=r[i]>=0 ? shrink:-shrink;
  }
  c_free(terms);
  return status ? OSQP_DATA_VALIDATION_ERROR:0;
}

OSQPInt osqp_setup_directional_peaks(OSQPSolver* solver, OSQPInt ngroups,
 const OSQPInt* group, const OSQPFloat* a, const OSQPFloat* b,
 const OSQPFloat* cl, const OSQPFloat* cu) {
  OSQPInt i,g,m,status=0; OSQPInt* types=OSQP_NULL; OSQPInt* first=OSQP_NULL;
  OSQPFloat* ones=OSQP_NULL; OSQPFloat* zeros=OSQP_NULL; OSQPPenaltyData* pen;
  if (!solver || !solver->work || !solver->work->data) return OSQP_WORKSPACE_NOT_INIT_ERROR;
  m=solver->work->data->m;
  if (solver->work->data->penalty || ngroups<=0 || ngroups>m || m>INT_MAX/3 ||
      !group || !a || !b || !cl || !cu) return OSQP_DATA_VALIDATION_ERROR;
  if (solver->settings->scaling || solver->settings->adaptive_rho ||
      !isfinite(solver->settings->alpha) || solver->settings->alpha<=0 || solver->settings->alpha>=2 ||
      solver->settings->polishing || solver->settings->linsys_solver!=OSQP_DIRECT_SOLVER) return OSQP_SETTINGS_VALIDATION_ERROR;
  types=c_malloc((size_t)m*sizeof(OSQPInt)); first=c_malloc((size_t)ngroups*sizeof(OSQPInt));
  ones=c_malloc((size_t)m*sizeof(OSQPFloat)); zeros=c_calloc((size_t)m,sizeof(OSQPFloat));
  if (!types || !first || !ones || !zeros) { status=OSQP_MEM_ALLOC_ERROR; goto cleanup; }
  for(g=0;g<ngroups;g++) first[g]=-1;
  for(i=0;i<m;i++) {
    OSQPFloat trial[3]; g=group[i];
    if (g < -1 || g>=ngroups || !isfinite(a[i]) || !isfinite(b[i]) || a[i]<0 || b[i]<0 ||
        !isfinite(cl[i]) || cl[i]<=0 || !isfinite(cu[i]) || cu[i]<=0 ||
        peak_term(1,cl[i],cu[i],solver->settings->rho,trial) ||
        peak_term(-1,cl[i],cu[i],solver->settings->rho,trial)) { status=OSQP_DATA_VALIDATION_ERROR; goto cleanup; }
    if (g<0 && (a[i]!=0 || b[i]!=0)) { status=OSQP_DATA_VALIDATION_ERROR; goto cleanup; }
    if (g>=0) {
      if (first[g]<0) first[g]=i;
      else if (a[first[g]]!=a[i] || b[first[g]]!=b[i]) { status=OSQP_DATA_VALIDATION_ERROR; goto cleanup; }
    }
    types[i]=g>=0 ? OSQP_PENALTY_NORMINF:OSQP_PENALTY_NONE; ones[i]=g>=0 ? 1:0;
  }
  /* Reuse audited partition/allocation/rho setup; temporary ordinary norm
     weights are replaced only after all directional arrays validate. */
  status=osqp_setup_penalty(solver,OSQP_PENALTY_NONE,types,ones,zeros,zeros,ngroups,group);
  if (status) goto cleanup;
  pen=solver->work->data->penalty;
  pen->peak_lower=OSQPVectorf_malloc(m); pen->peak_upper=OSQPVectorf_malloc(m);
  OSQPVectorf_free(solver->work->penalty_sort_tmp);
  solver->work->penalty_sort_tmp=OSQPVectorf_malloc(3*pen->ngrouped);
  if (!pen->peak_lower || !pen->peak_upper || !solver->work->penalty_sort_tmp) {
    penalty_free(solver->work); status=OSQP_MEM_ALLOC_ERROR; goto cleanup;
  }
  OSQPVectorf_from_raw(pen->peak_lower,cl); OSQPVectorf_from_raw(pen->peak_upper,cu);
  OSQPVectorf_from_raw(pen->alpha1,a); OSQPVectorf_from_raw(pen->alpha2,b);
  pen->peak_rho=solver->settings->rho; pen->peak_sigma=solver->settings->sigma;
  pen->peak_alpha=solver->settings->alpha;
  pen->peak_rho_is_vec=solver->settings->rho_is_vec;
cleanup:
  c_free(types); c_free(first); c_free(ones); c_free(zeros); return status;
}

#endif /* non-embedded setup/allocation */

OSQPInt penalty_peak_settings_valid(const OSQPSolver* solver,const OSQPSettings* settings) {
  OSQPPenaltyData* p=solver->work->data->penalty;
  return !p || !p->peak_lower || (!settings->scaling && !settings->adaptive_rho &&
    isfinite(settings->alpha) && settings->alpha==p->peak_alpha &&
    !settings->polishing && settings->linsys_solver==OSQP_DIRECT_SOLVER &&
    settings->rho==p->peak_rho && settings->sigma==p->peak_sigma && settings->rho_is_vec==p->peak_rho_is_vec);
}

void penalty_peak_project(OSQPSolver* solver, OSQPVectorf* z, const OSQPVectorf* v) {
  OSQPWorkspace* w=solver->work; OSQPPenaltyData* p=w->data->penalty;
  OSQPInt g,j,i,n; OSQPFloat t; OSQPFloat* terms=w->penalty_sort_tmp->values;
  for(g=0;g<p->ngroups;g++) {
    OSQPInt start=p->group_ptr->values[g], end=p->group_ptr->values[g+1], first=p->group_rows->values[start];
    if (p->alpha1->values[first]==0 && p->alpha2->values[first]==0) {
      for(j=start;j<end;j++) { i=p->group_rows->values[j]; if(!isfinite(v->values[i])) goto failure; z->values[i]=v->values[i]; }
      continue;
    }
    n=end-start;
    for(j=start;j<end;j++) {
      OSQPFloat r,rho; i=p->group_rows->values[j];
      r=v->values[i]-c_min(c_max(v->values[i],w->data->l->values[i]),w->data->u->values[i]);
      rho=solver->settings->rho_is_vec ? w->rho_vec->values[i]:solver->settings->rho;
      if (peak_term(r,p->peak_lower->values[i],p->peak_upper->values[i],rho,terms+3*(j-start))) goto failure;
    }
    if (peak_threshold(terms,n,p->alpha1->values[first],p->alpha2->values[first],&t)) goto failure;
    for(j=start;j<end;j++) {
      OSQPFloat vi,projection,r,c,magnitude,shrink; i=p->group_rows->values[j];
      vi=v->values[i]; projection=c_min(c_max(vi,w->data->l->values[i]),w->data->u->values[i]); r=vi-projection;
      c=r>=0 ? p->peak_upper->values[i]:p->peak_lower->values[i]; magnitude=c_absval(r); shrink=magnitude;
      if (c*magnitude>t) shrink=t/c;
      if (!isfinite(shrink) || (t>0 && magnitude>0 && shrink==0)) goto failure;
      z->values[i]=projection+(r>=0 ? shrink:-shrink);
      if (!isfinite(z->values[i])) goto failure;
    }
  }
  for(i=0;i<v->length;i++) if (p->type->values[i]==OSQP_PENALTY_NONE)
    z->values[i]=c_min(c_max(v->values[i],w->data->l->values[i]),w->data->u->values[i]);
  return;
failure:
  p->peak_numeric_error=1;
}

void penalty_peak_invalid_info(OSQPInfo* info) {
  update_status(info,OSQP_NUMERICAL_ERROR);
  info->obj_val=info->penalty_val=info->dual_obj_val=info->duality_gap=OSQP_NAN;
  info->prim_res=info->dual_res=OSQP_NAN;
}

void penalty_peak_zero_dual(OSQPSolver* solver) {
  OSQPPenaltyData* p=solver->work->data->penalty; OSQPInt i;
  if(!p || !p->peak_lower) return;
  for(i=0;i<solver->work->data->m;i++)
    if(p->type->values[i]!=OSQP_PENALTY_NONE && p->alpha1->values[i]==0 && p->alpha2->values[i]==0)
      solver->work->y->values[i]=0;
}

OSQPFloat penalty_peak_obj(const OSQPSolver* solver, const OSQPVectorf* z) {
  OSQPWorkspace* w=solver->work; OSQPPenaltyData* p=w->data->penalty;
  OSQPInt g,j,i; OSQPFloat total=0;
  for(g=0;g<p->ngroups;g++) {
    OSQPFloat peak=0; OSQPInt first=p->group_rows->values[p->group_ptr->values[g]];
    for(j=p->group_ptr->values[g];j<p->group_ptr->values[g+1];j++) {
      OSQPFloat r,c; i=p->group_rows->values[j];
      r=z->values[i]-c_min(c_max(z->values[i],w->data->l->values[i]),w->data->u->values[i]);
      c=r>=0 ? p->peak_upper->values[i]:p->peak_lower->values[i]; peak=c_max(peak,c*c_absval(r));
    }
    total+=p->alpha1->values[first]*peak+(.5*p->alpha2->values[first]*peak)*peak;
  }
  if (!isfinite(total)) { p->peak_numeric_error=1; return OSQP_INFTY; }
  return total;
}

OSQPFloat penalty_peak_conj(const OSQPSolver* solver, const OSQPVectorf* y) {
  OSQPPenaltyData* p=solver->work->data->penalty; OSQPInt g,j,i; OSQPFloat total=0;
  for(g=0;g<p->ngroups;g++) {
    OSQPFloat polar=0,excess,b; OSQPInt first=p->group_rows->values[p->group_ptr->values[g]];
    for(j=p->group_ptr->values[g];j<p->group_ptr->values[g+1];j++) {
      OSQPFloat yi; i=p->group_rows->values[j]; yi=y->values[i];
      polar+=c_absval(yi)/(yi>=0 ? p->peak_upper->values[i]:p->peak_lower->values[i]);
    }
    excess=c_max(polar-p->alpha1->values[first],0); b=p->alpha2->values[first];
    if (b==0) { if (excess>OSQP_PENALTY_CONJ_TOL*c_max(polar,p->alpha1->values[first])) return OSQP_INFTY; }
    else total+=(.5*excess)*(excess/b);
  }
  if (!isfinite(total)) return OSQP_INFTY;
  return total;
}

OSQPFloat penalty_peak_reccone(const OSQPSolver* solver,const OSQPVectorf* d,OSQPFloat tol) {
  OSQPWorkspace* w=solver->work; OSQPPenaltyData* p=w->data->penalty; OSQPInt g,j,i; OSQPFloat total=0;
  (void)tol; /* Conservative: never call nonzero quadratic/hard recession zero by a row-unit tolerance. */
  for(i=0;i<d->length;i++) if (p->type->values[i]==OSQP_PENALTY_NONE) {
    if ((d->values[i]>0 && w->data->u->values[i]<OSQP_INFTY*OSQP_MIN_SCALING) ||
        (d->values[i]<0 && w->data->l->values[i]>-OSQP_INFTY*OSQP_MIN_SCALING)) return OSQP_INFTY;
  }
  for(g=0;g<p->ngroups;g++) {
    OSQPFloat peak=0; OSQPInt first=p->group_rows->values[p->group_ptr->values[g]];
    for(j=p->group_ptr->values[g];j<p->group_ptr->values[g+1];j++) {
      OSQPFloat di,c; i=p->group_rows->values[j]; di=d->values[i];
      if ((di>=0 && w->data->u->values[i]>=OSQP_INFTY*OSQP_MIN_SCALING) ||
          (di<0 && w->data->l->values[i]<=-OSQP_INFTY*OSQP_MIN_SCALING)) continue;
      if (p->alpha2->values[first]>0 && di!=0) return OSQP_INFTY;
      c=di>=0 ? p->peak_upper->values[i]:p->peak_lower->values[i]; peak=c_max(peak,c*c_absval(di));
    }
    total+=p->alpha1->values[first]*peak;
  }
  return isfinite(total) ? total:OSQP_INFTY;
}
#endif
