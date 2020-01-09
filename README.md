[![Documentation Status](https://readthedocs.org/projects/scikits-odes/badge/?version=latest)](https://scikits-odes.readthedocs.org/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/bmcage/odes.svg?branch=master)](https://travis-ci.org/bmcage/odes)
[![Version](https://img.shields.io/pypi/v/scikits.odes.svg)](https://pypi.python.org/pypi/scikits.odes/)
[![License](https://img.shields.io/pypi/l/scikits.odes.svg)](https://pypi.python.org/pypi/scikits.odes/)
[![Supported versions](https://img.shields.io/pypi/pyversions/scikits.odes.svg)](https://pypi.python.org/pypi/scikits.odes/)
[![Supported implementations](https://img.shields.io/pypi/implementation/scikits.odes.svg)](https://pypi.python.org/pypi/scikits.odes/)
[![PyPI](https://img.shields.io/pypi/status/scikits.odes.svg)](https://pypi.python.org/pypi/scikits.odes/)



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3602093.svg)](https://doi.org/10.5281/zenodo.3602093)

[![Paper DOI](http://joss.theoj.org/papers/10.21105/joss.00165/status.svg)](https://doi.org/10.21105/joss.00165)


ODES is a scikit for Python 2.7 and 3.3-3.5 offering extra ode/dae solvers, as an extension to what is available in scipy.
The documentation is available at [Read The Docs](https://scikits-odes.readthedocs.io/en/latest/), and API docs can be found at https://bmcage.github.io/odes.

# Available solvers:
ODES provides interfaces to the following solvers:
* BDF linear multistep method for stiff problems (CVODE and IDA from SUNDIALS)
* Adams-Moulton linear multistep method for nonstiff problems (CVODE and IDA from SUNDIALS)
* Explicit Runge-Kutta method of order (4)5 with stepsize control ( *dopri5* from `scipy.integrate`)
* Explicit Runge-Kutta method of order 8(5,3) with stepsize control ( *dop853* from `scipy.integrate`)
* Historical solvers: *lsodi* and *ddaspk* are available for comparison reasons. Use IDA instead! Note that *lsodi* fails on architecture *aarch64*.


# Usage
A simple example solving the Van der Pol oscillator is as follows:

```python
import matplotlib.pyplot as plt
import numpy as np
from scikits.odes import ode

t0, y0 = 1, np.array([0.5, 0.5])  # initial condition
def van_der_pol(t, y, ydot):
    """ we create rhs equations for the problem"""
    ydot[0] = y[1]
    ydot[1] = 1000*(1.0-y[0]**2)*y[1]-y[0]

solution = ode('cvode', van_der_pol, old_api=False).solve(np.linspace(t0,500,200), y0)
plt.plot(solution.values.t, solution.values.y[:,0], label='Van der Pol oscillator')
plt.show()
```

For simplicity there is also a convenience function `odeint` wrapping the ode solver class. See the [User Guide](https://scikits-odes.readthedocs.io/en/latest/guide.html) for a simple example for `odeint`, as well as simple examples for object orientated interfaces and further examples using ODES solvers.


# Projects that use odes
You can learn by example from following code that uses ODES:
* Centrifuge simulation, a wrapper around the ida solver: see [centrifuge-1d](https://github.com/bmcage/centrifuge-1d/blob/master/centrifuge1d/modules/shared/solver.py)

You have a project using odes? Do a pull request to add your project.

# Citing ODES
If you use ODES as part of your research, can you please cite the
[ODES JOSS paper](https://doi.org/10.21105/joss.00165). Additionally, if you use
one of the SUNDIALS solvers, we strongly encourage you to cite the
[SUNDIALS papers](https://computation.llnl.gov/projects/sundials/publications).
