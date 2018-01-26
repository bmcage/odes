.. _user_guide:

Structure of ``odes`` and User's Guide
######################################

There are a number of different ways of using ``odes`` to solve a system of
ODEs/DAEs:

 * :py:class:`scikits.odes.ode.ode` and :py:class:`scikits.odes.dae.dae` classes, which provides an object oriented interface and significant amount of control of the solver.
 * :py:func:`scikits.odes.odeint.odeint`, a single function alternative to the object
   oriented interface.
 * Accessing the lower-level solver-specific wrappers, such as the modules in :py:mod:`scikits.odes.sundials`.

In general, a user supplies a function with the signature::

    right_hand_side(t: float, y: Array[float], ydot: Array[float]) -> int

for the ODE solvers, and::

    right_hand_side(t: float, y: Array[float], ydot: Array[float], residue: Array[float]) -> int

for the DAE solvers, as well as positions to integrate between and initial
values.

.. _simple_function_guide:

Simple Function Interface (``odeint``)
--------------------------------------
The simplest user program using the ``odeint`` interface, assuming you have
implemented the ODE ``right_hand_side`` mentioned above, is::

    import numpy as np
    from scikits.odes.odeint import odeint

    tout = np.linspace(0, 1)
    initial_values = np.array([0])

    def right_hand_side(t, y, ydot):
        """
        User's right hand side function
        """
        pass

    output = odeint(right_hand_side, tout, initial_values)
    print(output.values.y)

By default, CVODE's BDF method is used, however a different method can be
specified via the ``method`` keyword. Methods specific to ``odeint``, which use
the recommended setting for the individual solvers, are:

``bdf``
    CVODE's BDF method (default)

``admo``
    CVODE's Adams-Moulton method

``rk5``
    `dopri5 <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_ Runge-Kutta method of order (4)5

``rk8``
    `dop853 <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_ Runge-Kutta method of order 8(5,3)

``beuler``
    Implicit/Backward Euler method (for educational purposes only!)

``trapz``
    Trapezoidal Rule method (for educational purposes only!)

A specific solver (e.g. CVODE) can also be passed in via ``method``, in the
same way specified by the Object Oriented Interface. Solver specific options
can be passed in via additional keyword arguments.

.. _object_orientated_guide:

Object Oriented Interface (``ode`` and ``dae``)
-----------------------------------------------
The object oriented interfaces for ``ode`` and ``dae`` are almost identical,
with solver customisations via either keyword arguments or via a
``set_options`` method, repeated usage of the same solver via the ``solve``
method, and individual stepping via the ``step`` method.

.. note::
    ``odes`` 2.2.2 and later have a new output format, which provides
    access to more solver information. In a future release, the default will be
    the new output format. To use the new output format, pass as a keyword
    argument ``old_api=False``.

.. _ode_guide:

``ode`` Object Oriented Interface
.................................
The simplest user program using the ``ode`` interface, assuming you have
implemented the ODE ``right_hand_side`` mentioned above, is::

    import numpy as np
    from scikits.odes.ode import ode

    SOLVER = 'cvode'
    tout = np.linspace(0, 1)
    initial_values = np.array([0])
    extra_options = {'old_api': False}

    def right_hand_side(t, y, ydot):
        """
        User's right hand side function
        """
        pass

    ode_solver = ode(SOLVER, right_hand_side, **extra_options)
    output = ode_solver.solve(tout, initial_values)
    print(output.values.y)

Extra options are solver specific, but there is usually support for passing in
user data (passed as additional arguments to the provided ``right_hand_side``),
and for setting the tolerance of the solver. See :ref:`choosing_solvers` for
more information about individual solvers.

.. _ode_examples:

Examples
^^^^^^^^
There are a number of ``ode`` examples showing different features, including
solver specific features. Here are some of them:

 * `https://github.com/bmcage/odes/blob/master/ipython_examples/Simple%20Oscillator.ipynb`_

.. _dae_guide:

``dae`` Object Oriented Interface
.................................
The simplest user program using the ``dae`` interface, assuming you have
implemented the DAE ``right_hand_side`` mentioned above, is::

    import numpy as np
    from scikits.odes.dae import dae

    SOLVER = 'ida'
    tout = np.linspace(0, 1)
    y_initial = np.array([0])
    ydot_initial = np.array([0])
    extra_options = {'old_api': False}

    def right_hand_side(t, y, ydot, residue):
        """
        User's right hand side function
        """
        pass

    dae_solver = dae(SOLVER, right_hand_side, **extra_options)
    output = dae_solver.solve(tout, y_initial, ydot_initial)
    print(output.values.y)

Extra options are solver specific, but there is usually support for passing in
user data (passed as additional arguments to the provided ``right_hand_side``),
and for setting the tolerance of the solver. See :ref:`choosing_solvers` for
more information about individual solvers.

Examples
^^^^^^^^
There are a number of ``dae`` examples showing different features, including
solver specific features. Here are some of them:

 * `https://github.com/bmcage/odes/blob/master/ipython_examples/Double%20Pendulum%20as%20DAE%20with%20roots.ipynb`_
 * `https://github.com/bmcage/odes/blob/master/ipython_examples/Planar%20Pendulum%20as%20DAE.ipynb`_

.. _lower_level_guide:

Lower-level interfaces
----------------------
Using the lower-level interfaces is solver-specific, see the `API docs for more
information <https://bmcage.github.io/odes>`_ and :ref:`choosing_solvers` for
comparisons between solvers.
