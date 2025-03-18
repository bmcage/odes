import pytest
import numpy as np

from scikits_odes_sundials.ida import IDA
from scikits_odes_sundials.common_defs import DTYPE


def resfn(t, y, yp, res, user_data):
    res[0] = yp[0] + 0.04*y[0] - 1e4*y[1]*y[2]
    res[1] = yp[1] - 0.04*y[0] + 1e4*y[1]*y[2] + 3e7*y[1]**2
    res[2] = y[0] + y[1] + y[2] - 1
    
    
def jacfn(t, y, yp, res, cj, JJ, user_data):
    JJ[0,0] = 0.04 + cj
    JJ[0,1] = -1e4*y[2]
    JJ[0,2] = -1e4*y[1]
    JJ[1,0] = -0.04
    JJ[1,1] = 1e4*y[2] + 6e7*y[1] + cj
    JJ[1,2] = 1e4*y[1]
    JJ[2,0] = 1
    JJ[2,1] = 1
    JJ[2,2] = 1
    
    
def prec_setupfn(t, y, yp, res, cj, user_data):
    P = user_data['precond']
    jacfn(t, y, yp, res, cj, P, user_data)


def prec_solvefn(t, y, yp, res, rvec, zvec, cj, delta, user_data):
    P = user_data['precond']
    zvec[:] = np.linalg.solve(P, rvec)
    
    
@pytest.mark.parametrize('linsolver', ('spgmr', 'spbcgs', 'sptfqmr'))
def test_iterative_solvers(linsolver):
    tspan = np.logspace(-6, 6, 50)
    y0 = np.array([1, 0, 0])
    yp0 = np.zeros_like(y0)

    user_data = {'precond': np.zeros((y0.size, y0.size))}

    solver = IDA(resfn, algebraic_vars_idx=[2], compute_initcond='yp0',
                 atol=1e-8, linsolver=linsolver, precond_type='left',
                 prec_setupfn=prec_setupfn, prec_solvefn=prec_solvefn,
                 user_data=user_data)

    # np.linalg.solve does not support extended precision
    if DTYPE == np.longdouble:
        pass
    else:
        soln = solver.solve(tspan, y0, yp0)
        assert soln.flag == 0
   