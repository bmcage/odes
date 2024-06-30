# -*- coding: utf-8 -*-
"""
scikits-odes is a scikit offering a cython wrapper around some extra ode/dae
solvers, so they can mature outside of scipy.

It offers wrappers around the following solvers from `SUNDIALS`_
 * CVODE
 * IDA

It additionally offers wrappers around
 * `ddaspk <http://www.netlib.org/ode/ddaspk.f>` (included)
 * `lsodi <http://www.netlib.org/ode/lsodi.f>` (included)

.. _SUNDIALS: https://computation.llnl.gov/casc/sundials/main.html
"""

from .dae import dae
from .ode import ode
from . import _version
__version__ = _version.get_versions()['version']
