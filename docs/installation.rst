Installation
============

Requirements before install
---------------------------
Before building ``odes``, you need to have installed:

    * numpy (automatically dealt with if using pip >=10)
    * Python header files (``python-dev``/``python3-dev`` on Debian/Ubuntu-based
      distributions, ``python-devel`` on Fedora)
    * C compiler
    * Fortran compiler (e.g. gfortran)
    * `Sundials 4.1.0 <https://computation.llnl.gov/casc/sundials/download/download.html>`_ 

In addition, if building from a git checkout, you'll also need Cython.

It is required that Sundials is built with the BLAS/LAPACK interface enabled, so check
the Fortran Settings section. A typical install if sundials download package is
extracted into directory sundials-3.1.1 is on a \*nix system::

    mkdir build-sundials-4.1.0
    cd build-sundials-4.1.0/
    cmake -DLAPACK_ENABLE=ON -DSUNDIALS_INDEX_SIZE=64 -DCMAKE_INSTALL_PREFIX=<install_path> ../sundials-4.1.0/
    make install

.. warning::

    Make sure you use the Fortran compiler as used for your BLAS/LAPACK install!

.. tip::

    We recommend using `OpenBLAS <http://www.openblas.net/>`_, which provides a
    optimised BLAS implementation which widely distributed, and which doesn't
    need to be recompiled for different CPUs.

Installation
------------
To install ``odes``, use::

    pip install scikits.odes

which will download the latest version from PyPI. This will handle the installation of the additional runtime dependencies of ``odes``. You should then run the tests to make sure everything is set up correctly.

If you have installed SUNDIALS in a non-standard path (e.g. ``/usr/`` or ``/usr/local/``), you can set ``$SUNDIALS_INST`` in your environment to the installation prefix of SUNDIALS (i.e. value of ``<install_path>`` mentioned above).


Testing your version of ``odes``
................................
To test the version in python, use in the python shell::

    >>> import pkg_resources
    >>> pkg_resources.get_distribution("scikits.odes").version

    
Running the Tests
.................
You need nose to run the tests. To install nose, run::

    pip install nose

To run the tests, in the python shell::

    >>> import scikits.odes as od; od.test()
    
Note that the sundials library must be in your ``LD_LIBRARY_PATH``. So, make sure the directory ``$SUNDIALS_INST/lib`` is included. You can do this for example as follows (assuming sundials was installed in ``/usr/local``::

    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

Installation of ODES from git checkout
---------------------------------------------
You can copy the git repository locally in directory odes with::

    git clone git://github.com/bmcage/odes.git odes

Inside the ``odes`` directory, run::

    pip install .

which will install the checked out version of ``odes``. The same environment
variables mentioned above can be used to control installation options.

.. note::
    If you try to run the tests whilst in the ``odes`` directory, Python will pick up the source directory, and not the built version. Move to a different directory when running the tests.

Troubleshooting
---------------


LAPACK Not Found
................
Most issues with using ``odes`` are due to incorrectly setting the LAPACK libraries, resulting in error, typically::

    AttributeError: module 'scikits.odes.sundials.cvode' has no attribute 'CVODE'

or::

    undefined reference to dcopy_

This is an indication ``odes`` does not link correctly to the LAPACK directories. You can solve this as follows:
When installing sundials, look at output of cmake. If it has::

  -- A library with BLAS API not found. Please specify library location.
  -- LAPACK requires BLAS
  -- A library with LAPACK API not found. Please specify library location.

then ``odes`` will not work. First make sure you install sundials with BLAS and LAPACK found. On Debian/Ubuntu one needs ``sudo apt-get install libopenblas-dev liblapack-dev``
Once installed correctly, the sundials cmake output should be::

  -- A library with BLAS API found.
  -- Looking for Fortran cheev
  -- Looking for Fortran cheev - found
  -- A library with LAPACK API found.
  -- Looking for LAPACK libraries... OK
  -- Checking if Lapack works... OK

You can check the CMakeCache.txt file to see which libraries are found. It should have output similar to::

  //Blas and Lapack libraries
  LAPACK_LIBRARIES:STRING=/usr/lib/liblapack.so;/usr/lib/libf77blas.so;/usr/lib/libatlas.so
  //Path to a library.
  LAPACK_lapack_LIBRARY:FILEPATH=/usr/lib/liblapack.so

With above output, you can set the LAPACK directories and libs correctly. To force ``odes`` to find these directories you can set them by force by editing the file ``scikits/odes/sundials/setup.py``, and passing the directories and libs as used by sundials::

  INCL_DIRS_LAPACK = ['/usr/include', '/usr/include/atlas']
  LIB_DIRS_LAPACK  = ['/usr/lib']
  LIBS_LAPACK      = ['lapack', 'f77blas', 'atlas']

Note that on your install, these directories and libs might be different than the example above! With these variables set, installation of ``odes`` should be successful.

Linking Errors
..............
Verify you link to the correct sundials version. Easiest to ensure you only have one ``libsundials_xxx`` installed. If several are installed, pass the correct one via the ``$SUNDIALS_INST`` environment variable.
