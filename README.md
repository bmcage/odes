This is a scikit offering some extra ode/dae solvers, so they can mature outside of scipy. The main solvers to use
are the *Sundials* solvers.

# General info

* You need scipy, 
* Tested with python 2.7 and 3.2
* cvode is an improvement on the ode (vode/dvode) solver in scipy.integrate. Use it to have modern features
* ida is a Differential Algebraic Equation solver. 

# Documentation
## Example use
Since 2.2.0, a new API is available, which will become the default. Typical usage is:

> import pylab
> import numpy as np
> from scikits.odes import ode
> 
> t0, y0 = 1, np.array([0.5, 0.5])  # initial condition
> def van_der_pol(t, y, ydot):
>     """ we create rhs equations for the problem"""
>     ydot[0] = y[1]
>     ydot[1] = 1000*(1.0-y[0]**2)*y[1]-y[0]
> 
> solution = ode('cvode', van_der_pol, old_api=False).solve(np.linspace(t0,500,200), y0)
> pylab.plot(solution.values.t, solution.values.y[:,0], label='Van der Pol oscillator')
> pylab.show()

## Notebook examples
Basic use:
* [Simple oscillator](https://github.com/bmcage/odes/blob/master/docs/ipython/Simple%20Oscillator.ipynb) solved with cvode
* [DAE example: planar pendulum](https://github.com/bmcage/odes/blob/master/docs/ipython/Planar%20Pendulum%20as%20DAE.ipynb) solved with ida

Advanced use:
* [Double pendulum](https://github.com/bmcage/odes/blob/master/docs/ipython/Double%20Pendulum%20as%20DAE%20with%20roots.ipynb) Example of using classes to pass residual and jacobian functions to IDA, and of how to implement roots functionality.

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
6. You need to have the sundials package version 2.6.2 installed, see (https://computation.llnl.gov/casc/sundials/download/download.html)

It is required that the Blas/Lapack interface in included in sundials, so check
the Fortran Settings section. A typical install if sundials download package is
extracted into directory sundials-2.6.2 is on a *nix system:
```
 mkdir build-sundials-2.6.2
 cd build-sundials-2.6.2/
 cmake -DLAPACK_ENABLE=ON ../sundials-2.6.2/
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
This builds the packages in the build directory. Libraries are searched in /usr/lib 
and /usr/local/lib, edit setup.py for other locations.

For a working scikit compile, LAPACK, ATLAS and BLAS must be found. A typical output of the build is:
> lapack_info:
>   FOUND:
>     libraries = ['lapack']
>     library_dirs = ['/usr/lib']
>     language = f77
> 
> blas_info:
>   FOUND:
>     libraries = ['blas']
>     library_dirs = ['/usr/lib']
>     language = f77
> 
>   FOUND:
>     libraries = ['lapack', 'blas']
>     library_dirs = ['/usr/lib']
>     define_macros = [('NO_ATLAS_INFO', 1)]
>     language = f77


You can try it without installation by using PYTHONPATH. For example:
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

See the examples for more info.

# Developer info
## Tests

You need nose to run the tests. Eg, to install it, run
```
easy_install nose
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
Please submit extra ipython notebook examples of usage of odes scikit. To install and use ipython, typical install instructions on Ubuntu 14.04 would be:
```
pip install "ipython[notebook]"
ipython notebook
```
Which should open a browser window from the current directory to work on a python notebook. Do this in the directory  `odes/docs/ipython`. You might obtain errors due to missing dependencies. For example, common is simplegeneric missing. Again, in Ubuntu 14.04 you would install it with
```
sudo apt-get install python-simplegeneric
```

## Release info

Release: 

1. set in common.py version string and DEV=False, commit this.
2. tag like: `git tag -a v2.2.0 -m "version 2.2.0"`
3. push tag: `git push --tags`
4. update to pypi repo: `python setup.py sdist --formats=gztar,zip register upload`
5. update version string to a higher number, and DEV=True
