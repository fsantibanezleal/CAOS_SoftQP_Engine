import numpy as np
from scipy import sparse
import pytest
pytest.importorskip('daqp')
from softqp.problem import Problem, Penalty
from softqp.solvers import solve

@pytest.mark.parametrize('target',[-6.,0.,6.])
def test_side_weights_in_original_units_match_lifting(target):
    p=Problem(sparse.eye(1,format='csc'),np.array([-target]),sparse.csc_matrix([[3.]]),
              np.array([-1.]),np.array([2.]),
              Penalty(np.array([.3]),np.array([1.2]),np.array([2.]),np.array([4.]),np.array([True])))
    a,b=[solve(p,method) for method in ['daqp','lifted']]
    assert a['status_code'] in (1,2)
    assert a['x']==pytest.approx(b['x'],abs=2e-6)
    assert a['metrics']['objective']==pytest.approx(b['metrics']['objective'],abs=1e-7)
    assert a['metrics']['stationarity_residual']<1e-6

def test_hard_equality_is_not_softened():
    p=Problem(sparse.eye(2,format='csc'),np.array([-3.,2.]),sparse.csc_matrix([[1.,0.],[0.,1.],[1.,1.]]),
              np.array([-1.,-1.,0.]),np.array([1.,1.,0.]),
              Penalty(np.full(3,.3),np.full(3,1.2),np.full(3,2.),np.full(3,4.),np.array([True,True,False])))
    a,b=[solve(p,method) for method in ['daqp','lifted']]
    assert a['x']==pytest.approx(b['x'],abs=2e-6)
    assert a['metrics']['hard_violation']<1e-7

def test_semidefinite_reference_does_not_insert_regularization():
    p=Problem(sparse.csc_matrix([[0.]]),np.array([-1.]),sparse.eye(1,format='csc'),
              np.array([-1.]),np.array([1.]),Penalty.uniform(1))
    with pytest.raises(ValueError,match='positive definite'):solve(p,'daqp')
