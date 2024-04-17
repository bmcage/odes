ODE Solvers
-----------
:py:mod:`scikits.odes` contains two main routines for solving ODEs: the simpler
:py:func:`scikits.odes.odeint.odeint`, and the more configurable
:py:class:`scikits.odes.ode.ode`. Both these routines allow selection of the
solver and solution method used. Additionally, it is also possible to directly
use the low level interface to individual solvers.

.. autofunction:: scikits.odes.odeint.odeint

.. autoclass:: scikits.odes.ode.ode
   :members:
