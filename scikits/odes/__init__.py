# -*- coding: utf-8 -*-
"""
scikits.odes is a scikit offering a cython wrapper around some extra ode/dae
solvers, so they can mature outside of scipy.

It offers wrappers around the following solvers from `SUNDIALS`_
 * CVODE
 * IDA

It additionally offers wrappers around
 * `ddaspk <http://www.netlib.org/ode/ddaspk.f>` (included)
 * `lsodi <http://www.netlib.org/ode/lsodi.f>` (included)

.. _SUNDIALS: https://computation.llnl.gov/casc/sundials/main.html
"""

from .dae import *
from .ode import *

__all__ = ['test'] + [s for s in dir() if not s.startswith('_')]

try:
    from numpy.testing import Tester
    test = Tester().test
except:
    #testing could not be loaded, old numpy version
    pass
