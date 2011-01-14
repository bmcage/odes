#! /usr/bin/env python
"""
Odes is a scikit toolkit for scipy to add some extra ode solvers. 
At present it provides dae solvers you can use, extending the capabilities 
offered in scipy.integrade.ode.

LICENSE: the license of odes is the same as scipy, new BSD.
"""

import os
import sys

import setuptools

from common import *

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path,
        license = LICENSE,
        download_url = DOWNLOAD_URL,
        long_description = LONG_DESCRIPTION,
        namespace_packages=['scikits'])
    config.add_subpackage(DISTNAME)
    config.add_data_files('scikits/__init__.py')
    return config

def setup_package():

    from numpy.distutils.core import setup
    setup(name=DISTNAME, packages=['scikits'],
        version = VERSION,
        maintainer = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        description = DESCRIPTION,
        url = URL,
        license = LICENSE,
        configuration = configuration,
        install_requires = 'scipy',
        zip_safe = False
        )
    return

if __name__ == '__main__':
    setup_package()