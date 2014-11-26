.. _choosing_solvers:

Choosing a Solver
#################
``odes`` interfaces with a number of different solvers:

`CVODE <https://computation.llnl.gov/projects/sundials/cvode>`_
    ODE solver with BDF linear multistep method for stiff problems and Adams-Moulton linear multistep method for nonstiff problems. Supports modern features such as: root (event) finding, error control, and (Krylov-)preconditioning. See `scikits.odes.sundials.cvode` for more details and solver specific arguments. Part of SUNDIALS, it is a replacement for the earlier ``vode``/``dvode``.

`IDA <https://computation.llnl.gov/projects/sundials/ida>`_
    DAE solver with BDF linear multistep method for stiff problems and Adams-Moulton linear multistep method for nonstiff problems. Supports modern features such as: root (event) finding, error control, and (Krylov-)preconditioning. See `scikits.odes.sundials.ida` for more details and solver specific arguments. Part of SUNDIALS.

`dopri5 <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_
    Part of ``scipy.integrate``, explicit Runge-Kutta method of order (4)5 with stepsize control.

`dop853 <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_
    Part of ``scipy.integrate``, explicit Runge-Kutta method of order 8(5,3) with stepsize control.

``odes`` also includes for comparison reasons the historical solvers:

`lsodi <http://www.netlib.org/odepack/opkd-sum>`_
    Part of `odepack <http://www.netlib.org/odepack/opkd-sum>`_, IDA should be
    used instead of this. See `scikits.odes.lsodiint` for more details.

`ddaspk <http://www.netlib.org/ode/>`_
    Part of `daspk <http://www.netlib.org/ode/>`_, IDA should be used instead of this. See `scikits.odes.ddaspkint` for more details.

Support for other SUNDIALS solvers (e.g. ARKODE) is currently not implemented,
nor is support for non-serial methods (e.g. MPI, OpenMP). Contributions adding
support new SUNDIALS solvers or features is welcome.

Performance of the Solvers
==========================

A comparison of different methods is given in following image. In this BDF, RK23, RK45 and Radau are `python implementations <https://github.com/scipy/scipy/pull/6326>`_; cvode is the CVODE interface included in ``odes``; lsoda, odeint and vode are the `scipy integrators (2016) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_, dopri5 and dop853 are the Runge-Kutta methods in `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_. For this problem, cvode performs fastest at a preset tolerance.

.. image:: ../ipython_examples/PerformanceTests.png

You can generate above graph via the `Performance notebook <https://github.com/bmcage/odes/blob/master/ipython_examples/Performance%20tests.ipynb>`_.
