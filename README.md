This is a scikit offering some extra ode/dae solvers, so they can mature outside of scipy. The main solvers to use
are the *Sundials* solvers.

# General info

* You need scipy, 
* Tested with python 2.7 and 3.2
* cvode is an improvement on the ode (vode/dvode) solver in scipy.integrate. Use it to have modern features
* ida is a Differential Algebraic Equation solver. 

# Documentation

For examples, see the docs/src/examples directory and scikits/odes/tests directory. 

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

## Installation from sources

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

## Release info

Release: 

1. set in common.py version string and DEV=False, commit this.
2. tag like: `git tag -a v1.0.0 -m "version 1.0.0"`
3. push tag: `git push --tags`
4. update to pypi repo: `python setup.py sdist --formats=gztar,zip register upload`
5. update version string to a higher number, and DEV=True
