descr   = """
Odes is a scikit toolkit for scipy to add extra ode solvers.
Specifically it interfaces the Sundials solvers cvode, cvodes, ida and idas.
It this way it provides extra modern ode and dae solvers you can use, 
extending the capabilities offered in scipy.integrade.ode.

LICENSE: the license of odes is the same as scipy, new BSD.
"""

DISTNAME            = 'scikits.odes'
DESCRIPTION         = 'A python module for ordinary differential equation and'+\
                      ' differential algebraic equation solvers'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'maintainer of odes is B. Malengier'
MAINTAINER_EMAIL    = 'benny.malengier@gmail.org'
URL                 = 'https://github.com/bmcage/odes'
LICENSE             = 'new BSD'

DOWNLOAD_URL        = URL

INSTALL_REQUIRES = ['scipy']
BUILD_REQUIRES = ['numpy', 'cython<3.0.0a8'] # This is need for older pip

MAJOR = 2
MINOR = 7
MICRO = 0
DEV = False

CLASSIFIERS = [
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        "Programming Language :: Python :: 3 :: Only",
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
]

def build_verstring():
    return '%d.%d.%d' % (MAJOR, MINOR, MICRO)

def build_fverstring():
    if DEV:
        return build_verstring() + '.dev0'
    else:
        return build_verstring()

def write_version(fname):
    f = open(fname, "w")
    f.writelines("version = '%s'\n" % build_verstring())
    f.writelines("dev =%s\n" % DEV)
    f.writelines("full_version = '%s'\n" % build_fverstring())
    f.close()

VERSION = build_fverstring()
