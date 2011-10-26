from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import os

INCL_DIRS=['/usr/lib/python3.2/site-packages/numpy/core/include/',
 '/home/archetyp/projekty/python/gent-centrifuge/cython_wrapper']

SUNDIALS_SOURCES=['sundials_auxiliary/sundials_auxiliary.c', 'c_sundials.pxd']
IDA_SOURCES=['c_ida.pxd', 'common_defs.pyx','ida.pyx' ]

extension_modules=[
    #Extension("nvectorserial", ["c_nvector_serial.pxd"]),
    #Extension("sundials", ["sundials.pxd"]),
    Extension("common_defs", sources=['common_defs.pyx'], include_dirs=INCL_DIRS),
    Extension("ida", sources=['c_ida.pxd', 'ida.pyx'], depends=['common_defs.c'], include_dirs=INCL_DIRS,
                         library_dirs=['/usr/lib'],
                         libraries=['sundials_ida', 'sundials_nvecserial', 'lapack'])
] 

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = extension_modules
)
