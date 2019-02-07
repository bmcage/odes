#! /usr/bin/env python
"""
Odes is a scikit toolkit for scipy to add some extra ode solvers. 
At present it provides dae solvers you can use, extending the capabilities 
offered in scipy.integrade.ode.

LICENSE: the license of odes is the same as scipy, new BSD.
"""

import os
import sys

from setuptools import find_packages

if '' not in sys.path:
    sys.path.insert(0, '')

from common import *

additional_kwargs = {}

if "bdist_wheel" in sys.argv or "install" in sys.argv:
    from os.path import join
    from glob import glob
    from numpy.distutils.core import setup, Extension
    from setup_build import build_ext

    # add cython build logic
    additional_kwargs["cmdclass"] = {'build_ext': build_ext}

    # f2py requires build_src to be called
    base_path = join('scikits', 'odes')
    daepack_paths = glob(join(base_path, 'daepack', '*.f'))
    additional_kwargs['ext_modules'] = [
        Extension('scikits.odes.ddaspk',
            sources=[join(base_path, 'ddaspk.pyf')] + daepack_paths,
        ), Extension('scikits.odes.lsodi',
            sources=[join(base_path, 'lsodi.pyf')] + daepack_paths,
        ),
    ]

else:
    from setuptools import setup

setup(
    name = DISTNAME,
    version = VERSION,
    maintainer = MAINTAINER,
    maintainer_email = MAINTAINER_EMAIL,
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    url = URL,
    license = LICENSE,
    setup_requires = BUILD_REQUIRES,
    install_requires = INSTALL_REQUIRES,
    packages = find_packages(),
    namespace_packages = ['scikits'],
    zip_safe = False,
    package_data = {
        # If any package contains *.pxd files, include them:
        '': ['*.pxd'],
    },
    classifiers = CLASSIFIERS,
    **additional_kwargs
)
