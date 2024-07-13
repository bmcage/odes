Sundials Solver Options
#######################

Selecting Precision
-------------------
Sundials can be built with different precisions (currently `'single'` which maps
to C `'float'`; `'double'` (the default) which maps to C `'double'`; and
`'extended'` which maps to C `'long double'`), and scikits-odes-sundials
supports using whichever precision Sundials is built with. To take advantage of
this, you must first have sundials built and installed with the desired
precision setting.

Once you have done this, build scikits-odes-sundials against this particular
version of sundials (see the main documentation for how to do this).
scikits-odes-sundials will automatically detect the precision, and store this in
a variable called
:py:const:`DTYPE`. :py:const:`DTYPE` should be accessed from
:py:mod:`scikits_odes_sundials` (other modules may have :py:const:`DTYPE`
defined, but :py:mod:`scikits_odes_sundials` should be preferred). Additionally
:py:const:`scikits_odes_sundials.precision` contains the precision setting found
by scikits.odes.

To use :py:const:`DTYPE`, treat it as a numpy dtype; use it whenever you need to
create an array::

    np.array([1,2,3], dtype=DTYPE)

or when using scalars::

    DTYPE(0.1) * DTYPE(0.1)


