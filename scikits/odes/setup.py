#!/usr/bin/env python
from __future__ import print_function

from os.path import join

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    print("=============================================")
    print("parent package is %s" % parent_package)
    print("top path is %s" % top_path)
    print("=============================================")
    config = Configuration('odes',parent_package,top_path)

    config.add_library('daepack',
                       sources=[join('daepack','*.f')])
    
    # ddaspk
    config.add_extension('ddaspk',
                         sources=['ddaspk.pyf'],
                         libraries=['daepack'])
    # lsodi
    config.add_extension('lsodi',
                         sources=['lsodi.pyf'],
                        libraries=['daepack'])
    # sundials
    config.add_subpackage('sundials')

    # add the test directory
    config.add_subpackage('tests')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
