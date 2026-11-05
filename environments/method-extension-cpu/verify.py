"""Official CPU binary controls only; no method, dataset or trained model admission."""
from __future__ import annotations
import argparse
import importlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import sys
import time

PINS = {"clarabel": "0.11.1", "proxsuite": "0.7.2", "piqp": "0.6.4", "daqp": "0.10.3", "osqp": "1.1.3", "torch": "2.15.0+cpu", "torch-geometric": "2.8.1", "numpy": "2.4.6", "scipy": "1.18.1", "softqp": "0.2.0", "elastiqp": "0.2.0"}

def serial(value):
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serial(v) for v in value]
    if value is None or type(value) in (bool, int, str): return value
    if isinstance(value, float): return value if math.isfinite(value) else {"nonfinite": str(value)}
    if hasattr(value, "tolist"): return serial(value.tolist())
    if hasattr(value, "name") and hasattr(value, "value"): return {"name": str(value.name), "value": int(value.value)}
    return str(value)

def defaults(obj):
    return {n: serial(getattr(obj, n)) for n in dir(obj) if not n.startswith("_") and not callable(getattr(obj, n))}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--check", action="store_true", required=True); parser.parse_args()
    sys.dont_write_bytecode = True
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]: os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    start = time.monotonic()
    for name, version in PINS.items(): assert md.version(name) == version, (name, md.version(name), version)
    import numpy as np
    from scipy import sparse
    import psutil
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    assert torch.version.cuda is None and torch.get_num_threads() == torch.get_num_interop_threads() == 1
    import clarabel, proxsuite, piqp, daqp, osqp, softqp, elastiqp
    from torch_geometric.nn import LEConv
    from threadpoolctl import threadpool_limits, threadpool_info
    limiter = threadpool_limits(limits=1)
    imports = time.monotonic() - start
    assert imports < 180, "cold import admission deadline"
    modules = {}
    for name in ["numpy", "scipy", "clarabel", "proxsuite", "piqp", "daqp", "osqp", "softqp", "elastiqp", "torch", "torch_geometric"]:
        module = importlib.import_module(name); path = Path(module.__file__).resolve(); assert path.is_relative_to(Path(sys.prefix).resolve()), (name, str(path))
        modules[name] = path.relative_to(Path(sys.prefix).resolve()).as_posix()
    P = sparse.eye(2, format="csc", dtype=np.float64); q = np.array([-2., -4.]); A = sparse.csc_matrix([[1., 1.], [1., 0.], [0., 1.]])
    assert all(x.dtype == np.float64 and x.has_canonical_format for x in [P, A])
    rows = []; raw_defaults = {}
    def check(name, x, y, flag, expected, dual, elapsed):
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype == np.float64 and x.shape == (2,) and y.shape == (3,), name
        assert np.max(np.abs(x - expected)) < 1e-7 and np.max(np.abs(y - dual)) < 1e-7, (name, x, y)
        residual = np.max(np.abs(P @ x + q + A.T @ y)); assert residual < 1e-7, (name, residual)
        assert elapsed < 10, (name, "numerical fixture deadline")
        rows.append(dict(library=name,fixture="analytic-upper" if expected[0] == 1 else "analytic-lower",x=x.tolist(),signed_y=y.tolist(),status=str(flag),objective=float(.5*x@x+q@x),stationarity=float(residual),seconds=elapsed))
    cases = [(.5, 1., [1., 1.], [1., 0., 2.]), (1.5, 2., [1.5, .5], [3.5, -3., 0.])]
    common = dict(eps_abs=1e-10,eps_rel=1e-10,max_iter=200000,rho=.1,rho_is_vec=False,sigma=1e-6,alpha=1.,scaling=0,adaptive_rho=False,polishing=True,check_termination=1,verbose=False)
    for mod in [osqp, softqp]:
        d = mod.ext_builtin.OSQPSettings(); mod.ext_builtin.osqp_set_default_settings(d); raw_defaults[mod.__name__] = defaults(d)
        assert mod.ext_builtin.OSQP_USE_FLOAT == 0
        for lo, up, expected, dual in cases:
            clock=time.monotonic(); solver=mod.OSQP(algebra="builtin"); solver.setup(P,q,A,np.array([2.,lo,-np.inf]),np.array([2.,np.inf,up]),**common)
            solver.warm_start(x=np.array([.2,.4]),y=np.array([-.1,.2,.3])); result=solver.solve(raise_error=False)
            assert result.info.status_val == 1; check(mod.__name__,result.x,result.y,result.info.status,expected,dual,time.monotonic()-clock)
    raw_defaults["clarabel"] = defaults(clarabel.DefaultSettings())
    for lo, up, expected, dual in cases:
        clock=time.monotonic(); settings=clarabel.DefaultSettings(); settings.verbose=False; settings.max_threads=1; settings.max_iter=200000
        for field in ["tol_gap_abs","tol_gap_rel","tol_feas","tol_infeas_abs","tol_infeas_rel"]: setattr(settings,field,1e-10)
        C=sparse.csc_matrix([[1.,1.],[-1.,0.],[0.,1.]])
        solver=clarabel.DefaultSolver(P,q,C,np.array([2.,-lo,up]),[clarabel.ZeroConeT(1),clarabel.NonnegativeConeT(2)],settings); result=solver.solve()
        assert result.status == clarabel.SolverStatus.Solved
        check("clarabel",result.x,np.array([result.z[0],-result.z[1],result.z[2]]),result.status,expected,dual,time.monotonic()-clock)
    eq=sparse.csc_matrix([[1.,1.]]); b=np.array([2.]); C=sparse.eye(2,format="csc",dtype=np.float64)
    raw_defaults["proxsuite"] = defaults(proxsuite.proxqp.sparse.QP(2,1,2).settings)
    for lo, up, expected, dual in cases:
        clock=time.monotonic(); solver=proxsuite.proxqp.sparse.QP(2,1,2); settings=solver.settings
        for field,value in dict(eps_abs=1e-10,eps_rel=1e-10,max_iter=200000,compute_timings=True,check_duality_gap=True,eps_duality_gap_abs=1e-10,eps_duality_gap_rel=1e-10).items(): setattr(settings,field,value)
        settings.initial_guess=proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        solver.init(P,q,eq,b,C,np.array([lo,-np.inf]),np.array([np.inf,up])); solver.solve(); result=solver.results
        assert result.info.status == proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED
        check("proxsuite",result.x,np.r_[result.y,result.z],result.info.status,expected,dual,time.monotonic()-clock)
    raw_defaults["piqp"] = defaults(piqp.SparseSolver().settings)
    for lo, up, expected, dual in cases:
        clock=time.monotonic(); solver=piqp.SparseSolver()
        for field,value in dict(eps_abs=1e-10,eps_rel=1e-10,max_iter=200000,compute_timings=True,verbose=False).items(): setattr(solver.settings,field,value)
        solver.setup(P,q,eq,b,C,np.array([lo,-np.inf]),np.array([np.inf,up])); status=solver.solve(); result=solver.result
        assert status == piqp.Status.PIQP_SOLVED
        check("piqp",result.x,np.r_[result.y,result.z_u-result.z_l],status,expected,dual,time.monotonic()-clock)
    raw_defaults["daqp"] = serial(daqp.Model().settings)
    for lo, up, expected, dual in cases:
        clock=time.monotonic(); x,f,flag,info=daqp.solve(np.eye(2),q,A.toarray(),np.array([2.,np.inf,up]),np.array([2.,lo,-np.inf]),np.array([5,0,0],dtype=np.int32),primal_tol=1e-10,dual_tol=1e-10,zero_tol=1e-12,iter_limit=200000,eps_prox=0.,eq_reduction=-1)
        assert flag in (1,2); check("daqp",x,info["lam"],flag,expected,dual,time.monotonic()-clock)
    settings=elastiqp.Settings("das"); raw_defaults["elastiqp"] = defaults(settings)
    settings.eps_abs=settings.eps_rel=1e-10; settings.max_iter=200000; settings.eps_prox=0.; settings.warm_start=False
    clock=time.monotonic(); solver=elastiqp.Solver("das"); solver.settings=settings; solver.setup(np.asfortranarray([[1.]]),np.array([-2.]),np.asfortranarray([[1.]]),np.array([1.]),np.array([.5])); result=solver.solve()
    assert result.status == elastiqp.Status.Solved and np.max(np.abs(result.x-[1.5]))<1e-7 and np.max(np.abs(result.z-[.5]))<1e-7
    assert abs(float(.5*result.x@result.x-2*result.x[0]+.5*max(result.x[0]-1,0))+1.625)<1e-7
    rows.append(dict(library="elastiqp",fixture="analytic-L1",x=result.x.tolist(),z=result.z.tolist(),status=str(result.status),objective=-1.625,seconds=time.monotonic()-clock))
    rhs=torch.tensor([2.,8.],dtype=torch.float64,requires_grad=True); answer=torch.linalg.solve(torch.diag(torch.tensor([2.,4.],dtype=torch.float64)),rhs); answer.sum().backward()
    assert torch.allclose(answer,torch.tensor([1.,2.],dtype=torch.float64),atol=1e-12,rtol=0) and torch.allclose(rhs.grad,torch.tensor([.5,.25],dtype=torch.float64),atol=1e-12,rtol=0)
    conv=LEConv(1,1).double()
    with torch.no_grad():
        conv.lin1.weight.fill_(2.); conv.lin1.bias.fill_(.5); conv.lin2.weight.fill_(3.); conv.lin3.weight.fill_(4.); conv.lin3.bias.fill_(1.)
    x=torch.tensor([[1.],[2.],[4.]],dtype=torch.float64,requires_grad=True); edges=torch.tensor([[0,1],[1,2]],dtype=torch.int64); weights=torch.tensor([2.,3.],dtype=torch.float64)
    out=conv(x,edges,weights); out.sum().backward()
    assert torch.allclose(out,torch.tensor([[5.],[2.],[-5.5]],dtype=torch.float64),atol=1e-12,rtol=0) and torch.allclose(x.grad,torch.tensor([[8.],[4.],[-5.]],dtype=torch.float64),atol=1e-12,rtol=0)
    graph=torch.nn.Module(); graph.input_layer=LEConv(4,128); graph.inner_layer=LEConv(128,128); graph.output_layer=LEConv(128,1); graph.norm1=torch.nn.LayerNorm(128); graph.norm2=torch.nn.LayerNorm(128); graph.double()
    assert sum(v.numel() for v in graph.parameters()) == 52098
    assert all("lin2.bias" not in name for name,_ in graph.named_parameters())
    # Largest declared coordinate-feature head H128-224, one transient untrained
    # linear kernel; no optimizer, loss labels, checkpoint or learning campaign.
    width=128*129//2+128*224+128+4*224; linear=torch.nn.Linear(width,128,dtype=torch.float64)
    with torch.no_grad(): linear.weight.fill_(.001); linear.bias.zero_()
    feature=torch.ones((1,width),dtype=torch.float64,requires_grad=True); value=linear(feature); value.sum().backward()
    assert torch.allclose(value,torch.full_like(value,width*.001),atol=1e-9,rtol=0) and torch.allclose(feature.grad,torch.full_like(feature,.128),atol=1e-12,rtol=0)
    thread_info=threadpool_info(); assert all(v["num_threads"]==1 for v in thread_info)
    mem=psutil.Process().memory_info(); peak=getattr(mem,"peak_wset",mem.rss); assert peak<8*2**30
    output=dict(scope="binary-admission-only; no scientific artifacts consumed or produced",pins=PINS,modules=modules,controls=rows,torch_linalg=dict(answer=answer.tolist(),rhs_gradient=rhs.grad.tolist(),cpu=True),pyg=dict(output=out.tolist(),input_gradient=x.grad.tolist(),architecture_parameters=52098,parameter_shapes={n:list(v.shape) for n,v in graph.named_parameters()}),resource=dict(representative_head=[128,224],feature_width=width,linear_parameters=sum(v.numel() for v in linear.parameters()),peak_process_rss_bytes=peak,os_threads=psutil.Process().num_threads(),threadpools=thread_info,cold_import_seconds=imports,total_seconds=time.monotonic()-start),defaults=raw_defaults,torch_source_git=torch.version.git_version)
    print(json.dumps(serial(output),sort_keys=True,indent=2,allow_nan=False))

if __name__ == "__main__": main()
