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

if (sys.version_info[0] < 3) or (
    sys.version_info[0] == 3 and sys.version_info[1] < 4
):
    INSTALL_REQUIRES.append('enum34')

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path,
        license = LICENSE,
        download_url = DOWNLOAD_URL,
        long_description = LONG_DESCRIPTION,
        namespace_packages=['scikits'])
    # Avoid non-useful msg: "Ignoring attempt to set 'name' (from ... "
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True
    )
    config.add_subpackage(DISTNAME)
    config.add_data_files('scikits/__init__.py')
    config.add_data_files('scikits/odes/sundials/sundials_auxiliary/sundials_auxiliary.c')
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
        install_requires = INSTALL_REQUIRES,
        zip_safe = False,
        package_data = {
            # If any package contains *.pxd files, include them:
            '': ['*.pxd'],
        },
        )
    return

if __name__ == '__main__':
    setup_package()
