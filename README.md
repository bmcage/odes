[![Documentation Status](https://readthedocs.org/projects/scikits-odes/badge/?version=latest)](http://scikits-odes.readthedocs.org/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/bmcage/scikits.odes.svg?branch=master)](https://travis-ci.org/bmcage/scikits.odes)
[![Version](https://img.shields.io/pypi/v/scikits.odes.svg)](https://pypi.python.org/pypi/scikits.odes/)
[![License](https://img.shields.io/pypi/l/scikits.odes.svg)](https://pypi.python.org/pypi/scikits.odes/)
[![Supported versions](https://img.shields.io/pypi/pyversions/scikits.odes.svg)](https://pypi.python.org/pypi/scikits.odes/)
[![Supported implementations](https://img.shields.io/pypi/implementation/scikits.odes.svg)](https://pypi.python.org/pypi/scikits.odes/)
[![PyPI](https://img.shields.io/pypi/status/scikits.odes.svg)](https://pypi.python.org/pypi/scikits.odes/)

This is a scikit offering some extra ode/dae solvers, so they can mature outside of scipy. The main solvers to use
are the *Sundials* solvers.

# General info

* You need scipy, 
* Tested with python 2.7 and 3.3-3.5
* Available solvers:
    * BDF linear multistep method  for stiff problems. This is done or via *cvode*, which is an improvement on the ode (vode/dvode) solver in scipy.integrate, or for DAE systems via *ida*. Both are part of the sundials package. Use it to have modern features
    * Adams-Moulton linear multistep method for nonstiff problems. This is done also via *cvode* or *ida* (option lmm_type='ADAMS')
    * Explicit Runge-Kutta method of order (4)5 with stepsize control. This is done via *dopri5* from scipy.integrate.
    * Explicit Runge-Kutta method of order 8(5,3) with stepsize control. This is done via *dop853* from scipy.integrate.
    * Historical solvers: *lsodi* and *ddaspk* are available for comparison reasons. Use *ida* instead!

# Documentation
## Example use
Since 2.2.0, a new API is available, which will become the default. Typical usage is:

    import pylab
    import numpy as np
    from scikits.odes import ode
    
    t0, y0 = 1, np.array([0.5, 0.5])  # initial condition
    def van_der_pol(t, y, ydot):
        """ we create rhs equations for the problem"""
        ydot[0] = y[1]
        ydot[1] = 1000*(1.0-y[0]**2)*y[1]-y[0]
    
    solution = ode('cvode', van_der_pol, old_api=False).solve(np.linspace(t0,500,200), y0)
    pylab.plot(solution.values.t, solution.values.y[:,0], label='Van der Pol oscillator')
    pylab.show()

For simplicity there is also a convenience function *odeint* wrapping the ode solver class. See example use in [simple.py](https://github.com/bmcage/odes/blob/master/docs/src/examples/odeint/simple.py).

## Notebook examples
Basic use:
* [Simple oscillator](https://github.com/bmcage/odes/blob/master/docs/ipython/Simple%20Oscillator.ipynb) solved with cvode
* [DAE example: planar pendulum](https://github.com/bmcage/odes/blob/master/docs/ipython/Planar%20Pendulum%20as%20DAE.ipynb) solved with ida

Advanced use:
* [Double pendulum](https://github.com/bmcage/odes/blob/master/docs/ipython/Double%20Pendulum%20as%20DAE%20with%20roots.ipynb) Example of using classes to pass residual and jacobian functions to IDA, and of how to implement roots functionality.
* [Cython to speed up integration](https://github.com/bmcage/odes/blob/master/docs/ipython/Cython%20cvode%20speedup.ipynb)
Example of using a cython rhs to speed up the ODE integration. As sundials mostly uses internal C code, the benefits of using cython for the rhs are normally small.

## Performance
A comparison of different methods is given in following image. In this BDF, RK23, RK45 and Radau are [python implementations](https://github.com/scipy/scipy/pull/6326); cvode is the method cvode of this scikit odes; lsoda, odeint and vode are the scipy integrators (2016), dopxxx are the RK methods in scipy. For this problem, cvode performs fastest at a preset tolerance.

![Performance graph](https://github.com/bmcage/odes/blob/master/docs/ipython/Performance%20tests.png)

You can generate above graph via the [Performance notebook](https://github.com/bmcage/odes/blob/master/docs/ipython/Performance%20tests.ipynb).

## Python examples
For examples, see the [docs/src/examples](https://github.com/bmcage/odes/blob/master/docs/src/examples) directory and [scikits/odes/tests](https://github.com/bmcage/odes/blob/master/scikits/odes/tests) directory. 

## Projects that use odes
You can learn by example from following code that uses odes
* Centrifuge simulation, a wrapper around the ida solver: see [centrifuge-1d](https://github.com/bmcage/centrifuge-1d/blob/master/centrifuge1d/modules/shared/solver.py)

You have a project using odes? Do a pull request to add your project.

# Installation

## Requirements before install

1. You need numpy and scipy installed, as the aim is to extend scipy.integrate
2. You need to have cython installed and executable
3. You need python development files available (python-dev package)
4. You need a fortran compiler to install from source.
5. If you use python < 3.4, you need the [enum34 package](https://pypi.python.org/pypi/enum34) (eg via command: pip install enum34)  
6. You need to have the sundials package version 2.7.0 installed, see (https://computation.llnl.gov/casc/sundials/download/download.html)

It is required that the Blas/Lapack interface in included in sundials, so check
the Fortran Settings section. A typical install if sundials download package is
extracted into directory sundials-2.7.0 is on a *nix system:
```
 mkdir build-sundials-2.7.0
 cd build-sundials-2.7.0/
 cmake -DLAPACK_ENABLE=ON ../sundials-2.7.0/
 make
 ```
as root: 
```
 make install
 ```

This should install sundials in _/usr/local/lib_
Make sure you use the fortran compiler as used for your lapack/blas install!

## Installation of ODES scikit from sources

You can copy the git repository locally in directory odes with:
```
 git clone git://github.com/bmcage/odes.git odes
```
In the top directory (the same as the file you are reading now), just do as root:
```
 python setup.py build
```
This builds the packages in the build directory.
Libraries are searched in `/usr/lib` and `/usr/local/lib` by default.
You can also set `$SUNDIALS_INST` in your environment to the directory you have installed SUNDIALS into.
It should contain `lib` and `include` directories.

For a working scikit compile, LAPACK, ATLAS and BLAS must be found. A typical output of the build is:

    lapack_info:
      FOUND:
        libraries = ['lapack']
        library_dirs = ['/usr/lib']
        language = f77
    
    blas_info:
      FOUND:
        libraries = ['blas']
        library_dirs = ['/usr/lib']
        language = f77
    
      FOUND:
        libraries = ['lapack', 'blas']
        library_dirs = ['/usr/lib']
        define_macros = [('NO_ATLAS_INFO', 1)]
        language = f77


You can try it without installation by using `PYTHONPATH` and if needed adding local sundials install to `LD_LIBRARY_PATH`. For example:
On my box, the build libs are in odes/build/lib.linux-x86_64-2.7/, hence I can
use them with:
```
 PYTHONPATH=/path-to-odes/odes/build/lib.linux-x86_64-2.7/  python -c'import scikits.odes.sundials'
```
To install, as root:
```
 python setup.py install
```

# Usage

This installs the scikit, to use it in your python scripts use eg:
```python
from scikits.odes import dae
```

Note: you need to add the local sundials install to `LD_LIBRARY_PATH` if not installed in the standard locations.

See the examples for more info.

# Developer info
## Tests

You need nose to run the tests. Eg, to install it, run
```
pip install nose
```
To run the tests do in the python shell:

```python
>>> import scikits.odes as od; od.test()
```

or shorter, in a terminal: 

```
PYTHONPATH=/path-to-build python -c 'import scikits.odes as od; od.test()'
```

## IPython
Please submit extra ipython notebook examples of usage of odes scikit. To install and use ipython, typical install instructions would be (use `pip` instead of `pip3` for python2):
```
pip3 install jupyter
jupyter-notebook
```
Which should open a browser window from the current directory to work on a python notebook. Do this in the directory  `odes/docs/ipython`. You might obtain errors due to missing dependencies. For example, if simplegeneric missing, install it via `pip3`. Typical problem is that several conflicting versions are installed. If so, remove the versions installed on your system, and use `pip3` to install the most recent version. For example, an old ipython install will create problems, removing it (`sudo apt-get remove --purge ipython3 ipython`) fixes those issues.


## Release info

Release: 

1. set in common.py version string and DEV=False, commit this.
2. tag like: `git tag -a v2.2.0 -m "version 2.2.0"`
3. push tag: `git push --tags`
4. update to pypi repo: `python setup.py sdist --formats=gztar,zip register upload`
5. update version string to a higher number, and DEV=True

For the documentation, you need following packages
```
sudo apt-get install python-sphinx python-numpydoc
```

After local install, create the new documentation via

1. go to the sphinx directory: `cd sphinxdoc`
2. create the documentation: `make html`
3. upload the new html doc.

## Troubleshooting 
### linking errors - Lapack not found
Most issues with using the scikit are due to incorrectly setting the lapack libraries, resulting in error, typically:
`AttributeError: module 'scikits.odes.sundials.cvode' has no attribute 'CVODE'`
or
`undefined reference to dcopy_`

This is an indication the scikit does not link correctly to the lapack directories. You can solve this as follows:
When installing sundials, look at output of cmake. If it has:                                             
```
  -- A library with BLAS API not found. Please specify library location.                                               
  -- LAPACK requires BLAS                                                                                              
  -- A library with LAPACK API not found. Please specify library location.
```
then the scikit will not work ! First make sure you install sundials with BLAS and LAPACK found!
Eg on ubuntu one needs `sudo apt-get install libblas-dev libatlas-base-dev libopenblas-dev liblapack-dev gfortran`
Once installed correctly, the sundials cmake output should be
```
  -- A library with BLAS API found.
  -- Looking for Fortran cheev
  -- Looking for Fortran cheev - found
  -- A library with LAPACK API found.
  -- Looking for LAPACK libraries... OK
  -- Checking if Lapack works... OK
```
You can check the CMakeCache.txt file to see which libraries are found. It should have output as eg:
```
  //Blas and Lapack libraries
  LAPACK_LIBRARIES:STRING=/usr/lib/liblapack.so;/usr/lib/libf77blas.so;/usr/lib/libatlas.so
  //Path to a library.
  LAPACK_lapack_LIBRARY:FILEPATH=/usr/lib/liblapack.so
```
With above output, you can set the LAPACK directories and libs correctly. To force the scikit to find these directories you can set them by force by editing the file `scikits/odes/sundials/setup.py`, and passing the directories and libs as used by sundials:
```
  INCL_DIRS_LAPACK = ['/usr/include', '/usr/include/atlas']
  LIB_DIRS_LAPACK  = ['/usr/lib']
  LIBS_LAPACK      = ['lapack', 'f77blas', 'atlas']
```

Note that on your install, these directories and libs might be different than the example above! With these variables set, installation of the scikit should be successful.

### linking errors
Verify you link to the correct sundials version. Easiest to ensure you only have one libsundials_xxx installed. If several are installed, pass the correct one via the `$SUNDIALS_INST` environment variable.

### test failures
Most common reason for errors with the tests is because they are run in the scikits.odes download directory. This is not supported. The tests must be run from a different location, passing the install location if not installed globally. See info above on how to set PYTHONPATH if needed.
