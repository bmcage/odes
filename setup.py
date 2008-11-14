#! /usr/bin/env python

descr   = """
"""

import os
import sys

import setuptools

DISTNAME            = 'scikits.odes'
DESCRIPTION         = 'A python module for ordinary differential equation and'+\
                      'differential algebraic equation solvers'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'mainteiner of odes is B. Malengier'
MAINTAINER_EMAIL    = 'benny.malengier@gmail.org'
URL                 = 'http://todo.bb'
LICENSE             = 'new BSD'

odes_version = '0.01'

DOWNLOAD_URL        = 'http://todo.bb/odes.' + odes_version + '.tar.bz2'


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
        version = odes_version,
        maintainer = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        description = DESCRIPTION,
        url = URL,
        license = LICENSE,
        configuration = configuration,
        install_requires = 'scipy'
        )
    return

if __name__ == '__main__':
    setup_package()