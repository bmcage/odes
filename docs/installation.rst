Installation
============

Installation of ``odes`` can either be done by installing from source, or via
one of the distributions listed at the end of the page.

Requirements before install
---------------------------

Before building ``odes``, you need to have installed:

    * Python header files (``python-dev``/``python3-dev`` on Debian/Ubuntu-based
      distributions, ``python-devel`` on Fedora)
    * C compiler
    * Fortran compiler (e.g. gfortran)
    * `SUNDIALS <https://sundials.readthedocs.io/>`_ 

Python-based build dependencies should be installed auto-installed as part of
the build, see the pyproject.toml file for more details.

SUNDIALS Install
................
Detail of how to install SUNDIALS are part of their
`documentation <https://sundials.readthedocs.io/en/latest/Install_link.html>`_,
however there are some key things to consider when installing SUNDIALS for use
with ``odes``:

    #. As SUNDIALS does change its API quite regularly, you will need to choose
        a version of ``odes`` which supports the version of SUNDIALS you have
        installed. See below for an approximate compatibility table.
    #. If you wish to use 64-bit floats, then you will need to link to
        BLAS/LAPACK when building SUNDIALS. Other floating point precisions
        should not link with BLAS/LAPACK.

.. warning::

    Make sure you use the Fortran compiler as used for your BLAS/LAPACK install!

.. tip::

    We recommend using `OpenBLAS <http://www.openblas.net/>`_, which provides a
    optimised BLAS implementation which widely distributed, and which doesn't
    need to be recompiled for different CPUs.

.. list-table:: Approximate supported versions
    :header-rows: 1

    * - SUNDIALS version
      - ``odes`` version
    * - 7.x
      - 3.1.x
    * - 6.x
      - 2.7.x and 3.0.x
    * - 5.x
      - 2.6.x
    * - 4.x
      - 2.5.x
    * - 3.x
      - 2.4.x
    * - 2.x and earlier
      - 2.3.x and earlier

.. warning::
    Note that the SUNDIALS 2.x series tended to have more API changes compared
    to that of later series (which generally had the changes occur only on major
    releases), so matching SUNDIALS and ``odes`` may be much harder.

Installation
------------
To install ``odes``, use::

    pip install scikits-odes

which will download the latest version from PyPI. This will handle the installation of the additional runtime dependencies of ``odes``. You should then run the tests to make sure everything is set up correctly.

If you have installed SUNDIALS in a non-standard path (e.g. ``/usr/`` or ``/usr/local/``), you can set ``$SUNDIALS_INST`` in your environment to the installation prefix of SUNDIALS (i.e. value of ``<install_path>`` mentioned above).


Testing your version of ``odes``
................................
To test the version in python, use in the python shell::

    >>> import pkg_resources
    >>> pkg_resources.get_distribution("scikits-odes").version

    
Running the Tests
.................
You need nose to run the tests. To install nose, run::

    pip install nose

To run the tests, in the python shell::

    >>> import scikits_odes as od; od.test()
    
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

    AttributeError: module 'scikits_odes_sundials.cvode' has no attribute 'CVODE'

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

Installing via a Distribution
-----------------------------

Some distributions have packaged ``odes``. The maintainers of those packages
have added the following sections. If they do not work, please contact those
maintainers, and send us a Pull Request with any fixes.

Nix
...

By using the Nix package manager, you can install scikits-odes in one
line. Of course you need to install `nix <https://nixos.org/nix/>`_
first::

  curl https://nixos.org/nix/install | sh

And now you can start a python shell with scikits-odes (and numpy) ready for use::

  nix-shell \  
  -p python37Packages.scikits-odes \  
  -p python37Packages.numpy \  
  --run "python3"

You can verify that lapack is available (although the nix install will have
run many tests to check this already), try the following python snippet in the interpreter::

    import numpy as np
    from scikits_odes.odeint import odeint
    
    tout = np.linspace(0, 1)
    initial_values = np.array([1])
    
    def right_hand_side(t, y, ydot):
      ydot[0] = y[0]
    
    output = odeint(right_hand_side, tout, initial_values,linsolver='lapackdense')
    print(output.values.y)

You'll probably want to write a ``shell.nix`` or similar for your
project but you should refer to the nix documentation for this.
