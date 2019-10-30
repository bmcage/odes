# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 08:54:10 2016

@author: benny

based on the test in scipy.integrate:
# Authors: Nils Wagner, Ed Schofield, Pauli Virtanen, John Travers

Tests for odeint
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
                   allclose)

from numpy.testing import (
    assert_, TestCase, run_module_suite, assert_array_almost_equal,
    assert_raises, assert_allclose, assert_array_equal, assert_equal)

from scikits.odes.odeint import odeint
from scikits.odes.sundials.common_defs import DTYPE, has_lapack

# Lapack only compatible with precision double and index type 32.
TEST_LAPACK = (DTYPE == np.double and has_lapack)

#------------------------------------------------------------------------------
# Test ODE integrators
#------------------------------------------------------------------------------

class TestOdeint(TestCase):
    # Check integrate.odeint
    def _do_problem(self, problem):
        t = arange(0.0, problem.stop_t, 0.05)
        for method in ['bdf', 'admo', 'rk5', 'rk8']:
            sol = odeint(problem.f, t, problem.z0, method=method)
            assert_(problem.verify(sol.values.y, sol.values.t),
                    msg="Method {} on Problem {}. Printout {}".format(method, 
                            problem.__class__.__name__,
                            False
                            #problem.verify(sol.values.y, sol.values.t, True)
                            )
                    )

    def test_odeint(self):
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            self._do_problem(problem)

#------------------------------------------------------------------------------
# Test problems
#------------------------------------------------------------------------------

class ODE:
    """
    ODE problem
    """
    stiff = False
    cmplx = False
    stop_t = 1
    z0 = []

    lband = None
    uband = None

    atol = 1e-4 #1e-6
    rtol = 1e-6 #1e-5


class SimpleOscillator(ODE):
    r"""
    Free vibration of a simple oscillator::
        m \ddot{u} + k u = 0, u(0) = u_0 \dot{u}(0) \dot{u}_0
    Solution::
        u(t) = u_0*cos(sqrt(k/m)*t)+\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)
    """
    stop_t = 1 + 0.09
    z0 = array([1.0, 0.1], float)

    k = 4.0
    m = 1.0

    def f(self, t, z, rhs):
        tmp = np.zeros((2,2), float)
        tmp[0, 1] = 1.0
        tmp[1, 0] = -self.k / self.m
        rhs[:] = dot(tmp, z)
        return 0

    def verify(self, zs, t, printout=False):
        omega = sqrt(self.k / self.m)
        u = self.z0[0]*cos(omega*t) + self.z0[1]*sin(omega*t)/omega
        if printout:
            print ('exact=', u, ', computed=', zs[:, 0])
        return allclose(u, zs[:, 0], atol=self.atol, rtol=self.rtol)

class ComplexExp(ODE):
    r"""The equation :lm:`\dot u = i u`"""
    stop_t = 1.23*pi
    z0 = exp([1j, 2j, 3j, 4j, 5j])
    cmplx = True

    def f(self, t, z, rhs):
        rhs[:] = 1j*z
        return 0

    def jac(self, t, z, fz, J):
        J[:,:] = 1j*eye(5)
        return 0

    def verify(self, zs, t):
        u = self.z0 * exp(1j*t)
        return allclose(u, zs, atol=self.atol, rtol=self.rtol)


class Pi(ODE):
    r"""Integrate 1/(t + 1j) from t=-10 to t=10"""
    stop_t = 20
    z0 = [0]
    cmplx = True

    def f(self, t, z, rhs):
        rhs[:] = array([1./(t - 10 + 1j)])
        return 0

    def verify(self, zs, t, printout=False):
        u = -2j * np.arctan(10)
        return allclose(u, zs[-1, :], atol=self.atol, rtol=self.rtol)


class CoupledDecay(ODE):
    r"""
    3 coupled decays suited for banded treatment
    (banded mode makes it necessary when N>>3)
    """

    stiff = True
    stop_t = 0.5
    z0 = [5.0, 7.0, 13.0]
    lband = 1
    uband = 0

    lmbd = [0.17, 0.23, 0.29]  # fictious decay constants

    def f(self, t, z, rhs):
        lmbd = self.lmbd
        rhs[:] = np.array([-lmbd[0]*z[0],
                         -lmbd[1]*z[1] + lmbd[0]*z[0],
                         -lmbd[2]*z[2] + lmbd[1]*z[1]])
        return 0

    def jac(self, t, z, fz, J):
        # The full Jacobian is
        #
        #    [-lmbd[0]      0         0   ]
        #    [ lmbd[0]  -lmbd[1]      0   ]
        #    [    0      lmbd[1]  -lmbd[2]]
        #
        # The lower and upper bandwidths are lband=1 and uband=0, resp.
        # The representation of this array in packed format is
        #
        #    [-lmbd[0]  -lmbd[1]  -lmbd[2]]
        #    [ lmbd[0]   lmbd[1]      0   ]

        lmbd = self.lmbd
        j = np.zeros((self.lband + self.uband + 1, 3), order='F')

        def set_j(ri, ci, val):
            j[self.uband + ri - ci, ci] = val
        set_j(0, 0, -lmbd[0])
        set_j(1, 0, lmbd[0])
        set_j(1, 1, -lmbd[1])
        set_j(2, 1, lmbd[1])
        set_j(2, 2, -lmbd[2])
        J[:,:] = j
        return 0

    def verify(self, zs, t, printout=False):
        # Formulae derived by hand
        lmbd = np.array(self.lmbd)
        d10 = lmbd[1] - lmbd[0]
        d21 = lmbd[2] - lmbd[1]
        d20 = lmbd[2] - lmbd[0]
        e0 = np.exp(-lmbd[0] * t)
        e1 = np.exp(-lmbd[1] * t)
        e2 = np.exp(-lmbd[2] * t)
        u = np.vstack((
            self.z0[0] * e0,
            self.z0[1] * e1 + self.z0[0] * lmbd[0] / d10 * (e0 - e1),
            self.z0[2] * e2 + self.z0[1] * lmbd[1] / d21 * (e1 - e2) +
            lmbd[1] * lmbd[0] * self.z0[0] / d10 *
            (1 / d20 * (e0 - e2) - 1 / d21 * (e1 - e2)))).transpose()
        return allclose(u, zs, atol=self.atol, rtol=self.rtol)


PROBLEMS = [SimpleOscillator, ComplexExp, Pi, CoupledDecay]

#------------------------------------------------------------------------------


def f(t, x):
    dxdt = [x[1], -x[0]]
    return dxdt


def jac(t, x):
    j = array([[0.0, 1.0],
               [-1.0, 0.0]])
    return j


def f1(t, x, omega):
    dxdt = [omega*x[1], -omega*x[0]]
    return dxdt


def jac1(t, x, omega):
    j = array([[0.0, omega],
               [-omega, 0.0]])
    return j


def f2(t, x, omega1, omega2):
    dxdt = [omega1*x[1], -omega2*x[0]]
    return dxdt


def jac2(t, x, omega1, omega2):
    j = array([[0.0, omega1],
               [-omega2, 0.0]])
    return j


def fv(t, x, omega):
    dxdt = [omega[0]*x[1], -omega[1]*x[0]]
    return dxdt


def jacv(t, x, omega):
    j = array([[0.0, omega[0]],
               [-omega[1], 0.0]])
    return j

def test_odeint_trivial_time():
    # Test that odeint gives ValueError if only a single time point is given
    y0 = 1
    t = [0]
    for method in ['bdf', 'admo', 'rk5', 'rk8']:
        #sol = odeint(lambda t, y: -y, t, y0, method=method)
        assert_raises(ValueError, odeint, lambda t, y: -y, t, y0)

c = array([[-205, 0.01, 0.00, 0.0],
           [0.1, -2.50, 0.02, 0.0],
           [0.00, 0.01, -2.0, 0.01],
           [0.00, 0.00, 0.1, -1.0]])

cband1 = array([
           [0.  ,  0.01, 0.02,  0.01],
           [-205, -2.50, -2.0, -1.0 ],
           [0.1,   0.01,  0.1,  0.0 ],
           [0.  ,    0., 0.00,  0.0 ]])
cband2 = array([[0, 0.01, 0.02, 0.01],
           [-250, -2.50, -2., -1.],
           [0.1, 0.01, 0.1, 0.]])

def test_odeint_banded_jacobian():
    # Test the use of the `Dfun`, `ml` and `mu` options of odeint default bdf
    # solver

    def func(t, y, rhs):
        rhs[:] = c.dot(y)
        return 0

    def jac(t, y, fy, J):
        J[:,:] = c
        return 0

    def bjac1_rows(t, y, fy, J):
        J[:,:] = cband1
        return 0

    def bjac2_rows(t, y, fy, J):
        # define BAND_ELEM(A,i,j) ((A->cols)[j][(i)-(j)+(A->s_mu)])
        J[:,:] = cband2
        return 0

    y0 = np.ones(4)
    t = np.array([0, 5, 10, 100])

    # Use the full Jacobian.
    sol1 = odeint(func, t, y0,
                         atol=1e-13, rtol=1e-11, max_steps=10000,
                         jacfn=jac)

    # Use the full Jacobian.
    sol2 = odeint(func, t, y0,
                         atol=1e-13, rtol=1e-11, max_steps=10000,
                         linsolver='dense')
    # Use the banded Jacobian.
    sol3 = odeint(func, t, y0,
                         atol=1e-13, rtol=1e-11, max_steps=10000,
                         jacfn=bjac1_rows, lband=2, uband=1, linsolver='band')

    # Use the banded Jacobian.
    sol4 = odeint(func, t, y0,
                         atol=1e-13, rtol=1e-11, max_steps=10000,
                         jacfn=bjac2_rows, lband=1, uband=1, linsolver='band')

    # Use the banded Jacobian.
    sol5 = odeint(func, t, y0,
                         atol=1e-13, rtol=1e-11, max_steps=10000,
                         lband=2, uband=1, linsolver='band')

    # Use the banded Jacobian.
    sol6 = odeint(func, t, y0,
                         atol=1e-13, rtol=1e-11, max_steps=10000,
                         lband=1, uband=1, linsolver='band')
    # Use the diag Jacobian.
    sol7 = odeint(func, t, y0,
                         atol=1e-13, rtol=1e-11, max_steps=10000,
                         linsolver='diag')

    #use lapack versions:
    if TEST_LAPACK:
        # Use the full Jacobian.
        sol2a = odeint(func, t, y0,
                             atol=1e-13, rtol=1e-11, max_steps=10000,
                             linsolver='lapackdense')
        # Use the banded Jacobian.
        sol3a = odeint(func, t, y0,
                             atol=1e-13, rtol=1e-11, max_steps=10000,
                             jacfn=bjac1_rows, lband=2, uband=1, linsolver='lapackband')

        # Use the banded Jacobian.
        sol4a = odeint(func, t, y0,
                             atol=1e-13, rtol=1e-11, max_steps=10000,
                             jacfn=bjac2_rows, lband=1, uband=1, linsolver='lapackband')

        # Use the banded Jacobian.
        sol5a = odeint(func, t, y0,
                             atol=1e-13, rtol=1e-11, max_steps=10000,
                             lband=2, uband=1, linsolver='lapackband')

        # Use the banded Jacobian.
        sol6a = odeint(func, t, y0,
                             atol=1e-13, rtol=1e-11, max_steps=10000,
                             lband=1, uband=1, linsolver='lapackband')

    #finish with some other solvers
    sol10 = odeint(func, t, y0,
                         atol=1e-13, rtol=1e-11, max_steps=10000,
                         linsolver='spgmr')
    sol11 = odeint(func, t, y0,
                         atol=1e-13, rtol=1e-11, max_steps=10000,
                         linsolver='spbcgs')
    sol12 = odeint(func, t, y0,
                         atol=1e-13, rtol=1e-11, max_steps=10000,
                         linsolver='sptfqmr')

    assert_allclose(sol1.values.y, sol2.values.y, atol=1e-12, err_msg="sol1 != sol2")
    assert_allclose(sol1.values.y, sol3.values.y, atol=1e-12, err_msg="sol1 != sol3")
    assert_allclose(sol1.values.y, sol4.values.y, atol=1e-12, err_msg="sol1 != sol4")
    assert_allclose(sol1.values.y, sol5.values.y, atol=1e-12, err_msg="sol1 != sol5")
    assert_allclose(sol1.values.y, sol6.values.y, atol=1e-12, err_msg="sol1 != sol6")
    assert_allclose(sol1.values.y, sol7.values.y, atol=1e-12, err_msg="sol1 != sol7")
    assert_allclose(sol1.values.y, sol10.values.y, atol=1e-12, err_msg="sol1 != sol10")
    assert_allclose(sol1.values.y, sol11.values.y, atol=1e-12, err_msg="sol1 != sol11")
    assert_allclose(sol1.values.y, sol12.values.y, atol=1e-12, err_msg="sol1 != sol12")

    if TEST_LAPACK:
        assert_allclose(sol1.values.y, sol2a.values.y, atol=1e-12, err_msg="sol1 != sol2a")
        assert_allclose(sol1.values.y, sol3a.values.y, atol=1e-12, err_msg="sol1 != sol3a")
        assert_allclose(sol1.values.y, sol4a.values.y, atol=1e-12, err_msg="sol1 != sol4a")
        assert_allclose(sol1.values.y, sol5a.values.y, atol=1e-12, err_msg="sol1 != sol5a")
        assert_allclose(sol1.values.y, sol6a.values.y, atol=1e-12, err_msg="sol1 != sol6a")


def test_odeint_errors():
    def sys1d(t, x, rhs):
        rhs[:] = -100*x
        return 0

    def bad1(t, x, rhs):
        rhs[:] = 1.0/0
        return 0

    def bad2(t, x, rhs):
        rhs[:] = "foo"
        return 0

    def bad_jac1(t, x, fx, J):
        J[:,:] = 1.0/0
        return 0

    def bad_jac2(t, x, fx, J):
        J[:,:] = [["foo"]]
        return 0

    def sys2d(t, x, rhs):
        rhs[:] = [-100*x[0], -0.1*x[1]]
        return 0

    def sys2d_bad_jac(t, x, fx, J):
        J[:,:] = [[1.0/0, 0], [0, -0.1]]
        return 0

    assert_raises(ZeroDivisionError, odeint, bad1, [0, 1], [1.0])

    assert_raises(ValueError, odeint, bad2, [0, 1], [1.0])

    assert_raises(ZeroDivisionError, odeint, sys1d, [0, 1], [1.0], jacfn=bad_jac1)
    assert_raises(ValueError, odeint, sys1d, [0, 1], [1.0], jacfn=bad_jac2)

    assert_raises(ZeroDivisionError, odeint, sys2d, [0, 1], [1.0, 1.0],
                  jacfn=sys2d_bad_jac)

def test_odeint_bad_shapes():
    # Tests of some errors that can occur with odeint.

    def badrhs(t, x, rhs):
        rhs[:] = [1, -1]
        return 0

    def sys1(t, x, rhs):
        rhs[:] = -100*x
        return 0

    def badjac(t, x, fx, J):
        J[:,:] = [[0, 0, 0]]
        return 0

    # y0 must be at most 1-d.
    bad_y0 = [[0, 0], [0, 0]]
    assert_raises(ValueError, odeint, sys1, [0, 1], bad_y0)

    # t must be at most 1-d.
    bad_t = [[0, 1], [2, 3]]
    assert_raises(ValueError, odeint, sys1, bad_t, [10.0])

    # y0 is 10, should be an array [10]
    assert_raises(ValueError, odeint, badrhs, [0, 1], 10)
    # y0 is 10, but badrhs(x, t) returns [1, -1].
    assert_raises(ValueError, odeint, badrhs, [0, 1], [10])

    # shape of array returned by badjac(x, t) is not correct.
    assert_raises(ValueError, odeint, sys1, [0, 1], [10, 10], jacfn=badjac)


if __name__ == "__main__":
    run_module_suite()
