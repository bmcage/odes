.. Odes documentation master file, created by
   sphinx-quickstart on Tue Feb  2 13:13:32 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the ODES scikit documentation!
=========================================

The ODES scikit provides access to Ordinary Differential Equation (ODE) solvers and Differential Algebraic Equation (DAE) solvers not included in `scipy`_. A convenience function :py:func:`scikits.odes.odeint.odeint` is available for fast and fire and forget integration. Object oriented class solvers :py:class:`scikits.odes.ode.ode` and :py:class:`scikits.odes.dae.dae` are available for fine control. Finally, the low levels solvers are also directly exposed for specialised needs.

`Detailed API documentation can be found here`_

Contents:

.. toctree::
    :maxdepth: 2

    installation
    guide
    solvers
    contributing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _scipy: https://scipy.org/
.. _`Detailed API documentation can be found here`: https://bmcage.github.io/odes
