#!/usr/bin/env python
from __future__ import print_function

import os
from numpy.distutils.system_info import get_info
from scikits.odes._build import cython
#from Cython.Distutils import build_ext

base_path = os.path.abspath(os.path.dirname(__file__))

# Edit following paths if programs are installed differently!
INCL_DIRS_LAPACK = []
LIB_DIRS_LAPACK  = []
LIBS_LAPACK      = []
# Troubleshooting: 
# when installing sundials, look at output of cmake. If it has:                                             
#  -- A library with BLAS API not found. Please specify library location.                                               
#  -- LAPACK requires BLAS                                                                                              
#  -- A library with LAPACK API not found. Please specify library location.
# then this scikit will not work ! First make sure you install sundials with BLAS and LAPACK found
# 
# eg on ubuntu one needs sudo apt-get install libblas-dev libatlas-base-dev libopenblas-dev liblapack-dev gfortran
# then cmake output is 
#  -- A library with BLAS API found.
#  -- Looking for Fortran cheev
#  -- Looking for Fortran cheev - found
#  -- A library with LAPACK API found.
#  -- Looking for LAPACK libraries... OK
#  -- Checking if Lapack works... OK
# You can check CMakeCache.txt to see which libraries are found. It should have output as eg:
#  //Blas and Lapack libraries
#  LAPACK_LIBRARIES:STRING=/usr/lib/liblapack.so;/usr/lib/libf77blas.so;/usr/lib/libatlas.so
#  //Path to a library.
#  LAPACK_lapack_LIBRARY:FILEPATH=/usr/lib/liblapack.so
#
# With above output, you can set the LAPACK directories and libs correctly:
#INCL_DIRS_LAPACK = ['/usr/include', '/usr/include/atlas']
#LIB_DIRS_LAPACK  = ['/usr/lib']
#LIBS_LAPACK      = ['lapack', 'f77blas', 'atlas']

# paths for SUNDIALS
INCL_DIRS_SUNDIALS = [base_path]
# make sure libsundials_* only occurs in one location on your system !
LIB_DIRS_SUNDIALS  = [base_path, '/usr/local/lib/', '/usr/lib']

LIBS_SUNDIALS = ['sundials_nvecserial']
LIBS_IDA      = ['sundials_ida']
LIBS_CVODE    = ['sundials_cvode']

# paths for FORTRAN
LIB_DIRS_FORTRAN = []
LIBS_FORTRAN     = []

# use pkgconfig to find sundials
PKGCONFIG_CVODE = 'sundials-cvode-serial'

try:
    import pkgconfig
    try:
        if pkgconfig.exists(PKGCONFIG_CVODE):
            pkgconf = pkgconfig.parse(PKGCONFIG_CVODE)
            for d in pkgconf['library_dirs']:
                LIB_DIRS_SUNDIALS.append(str(d))
            for d in pkgconf['include_dirs']:
                INCL_DIRS_SUNDIALS.append(str(d))
    except EnvironmentError:
        pass
except ImportError:
    print("pkgconfig module not found, using preset paths")

if "SUNDIALS_INST" in os.environ:
    LIB_DIRS_SUNDIALS.append(os.path.join(os.environ["SUNDIALS_INST"], "lib"))
    INCL_DIRS_SUNDIALS.append(os.path.join(os.environ["SUNDIALS_INST"], "include"))
    print("SUNDIALS installation path set to `{}` via $SUNDIALS_INST.".format(
        os.environ["SUNDIALS_INST"]))
else:
    print("No path for SUNDIALS installation set by $SUNDIALS_INST.")

use_lapack = False
try:
    if INCL_DIRS_LAPACK and LIB_DIRS_LAPACK and LIBS_LAPACK:
        print('Using user provided LAPACK paths...')
        use_lapack = True
    else:
        lapack_opt = get_info('lapack_opt', notfound_action=2)

        if lapack_opt:
            INCL_DIRS_LAPACK = lapack_opt.get('include_dirs',[])
            LIB_DIRS_LAPACK  = lapack_opt.get('library_dirs',[])
            LIBS_LAPACK      = lapack_opt.get('libraries',[])
            use_lapack = True
        else:
            raise ValueError
        print('Found LAPACK paths via lapack_opt ...')
except:
    print('LAPACK was not detected, disabling sundials solvers')


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    print("=============================================")
    print("parent package is %s" % parent_package)
    print("top path is %s" % top_path)
    print("=============================================")
    config = Configuration('sundials', parent_package, top_path)

    if use_lapack:
        # sundials library
        ## assume installed globally at the moment
        ##config.add_library('sundials_ida',
        
        # sundials cython wrappers
        cython(['common_defs.pyx'], working_path=base_path, 
                        include_dirs=[])
        
        config.add_extension("common_defs", 
                             sources=['common_defs.c'], 
                             include_dirs=INCL_DIRS_SUNDIALS)

        cython(['ida.pyx'], working_path=base_path)
        config.add_extension("ida",
                             sources=['ida.c'],
                             depends=['common_defs.c'], 
                             include_dirs=INCL_DIRS_SUNDIALS+INCL_DIRS_LAPACK,
                             library_dirs=LIB_DIRS_SUNDIALS+LIB_DIRS_LAPACK+LIB_DIRS_FORTRAN,
                             libraries=LIBS_IDA+LIBS_SUNDIALS+LIBS_LAPACK+LIBS_FORTRAN)

        cython(['cvode.pyx'], working_path=base_path)
        config.add_extension("cvode",
                             sources=['cvode.c'],
                             depends=['common_defs.c'],
                             include_dirs=INCL_DIRS_SUNDIALS+INCL_DIRS_LAPACK,
                             library_dirs=LIB_DIRS_SUNDIALS+LIB_DIRS_LAPACK+LIB_DIRS_FORTRAN,
                             libraries=LIBS_CVODE+LIBS_SUNDIALS+LIBS_LAPACK+LIBS_FORTRAN)
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
