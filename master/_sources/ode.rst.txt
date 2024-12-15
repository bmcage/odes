ODE Solvers
-----------
:py:mod:`scikits_odes` contains two main routines for solving ODEs: the simpler
:py:func:`scikits_odes.odeint.odeint`, and the more configurable
:py:class:`scikits_odes.ode.ode`. Both these routines allow selection of the
solver and solution method used. Additionally, it is also possible to directly
use the low level interface to individual solvers.

.. autofunction:: scikits_odes.odeint.odeint
   :no-index:

.. autoclass:: scikits_odes.ode.ode
   :members:
   :no-index:
