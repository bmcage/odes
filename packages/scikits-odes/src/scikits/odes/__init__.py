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

# Compat with older versions
from scikits_odes import *
from scikits_odes import __version__
