descr   = """
Odes is a scikit toolkit for scipy to add some extra ode solvers. 
At present it provides dae solvers you can use, extending the capabilities 
offered in scipy.integrade.ode.

LICENSE: the license of odes is the same as scipy, new BSD.
"""

DISTNAME            = 'scikits.odes'
DESCRIPTION         = 'A python module for ordinary differential equation and'+\
                      'differential algebraic equation solvers'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'maintainer of odes is B. Malengier'
MAINTAINER_EMAIL    = 'benny.malengier@gmail.org'
URL                 = 'https://github.com/bmcage/odes'
LICENSE             = 'new BSD'

DOWNLOAD_URL        = URL

MAJOR = 2
MINOR = 1
MICRO = 0
DEV = False

CLASSIFIERS = [
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering']

def build_verstring():
    return '%d.%d.%d' % (MAJOR, MINOR, MICRO)

def build_fverstring():
    if DEV:
        return build_verstring() + 'dev'
    else:
        return build_verstring()

def write_version(fname):
    f = open(fname, "w")
    f.writelines("version = '%s'\n" % build_verstring())
    f.writelines("dev =%s\n" % DEV)
    f.writelines("full_version = '%s'\n" % build_fverstring())
    f.close()

VERSION = build_fverstring()
INSTALL_REQUIRE = 'scipy'
