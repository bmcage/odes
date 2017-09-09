import os
from distutils.log import info

from numpy.distutils.system_info import get_info
from scikits.odes._build import cython

BASE_PATH = os.path.abspath(os.path.dirname(__file__))

PKGCONFIG_CVODE = 'sundials-cvode-serial'
PKGCONFIG_IDA = 'sundials-ida-serial'

SUNDIALS_LIBRARIES = []
CVODE_LIBRARIES = []
IDA_LIBRARIES = []

SUNDIALS_LIBRARY_DIRS = []
CVODE_LIBRARY_DIRS = []
IDA_LIBRARY_DIRS = []

SUNDIALS_INCLUDE_DIRS = []
CVODE_INCLUDE_DIRS = []
IDA_INCLUDE_DIRS = []

SUNDIALS_LIBDIR = os.environ.get("SUNDIALS_LIBDIR")
SUNDIALS_INCLUDEDIR = os.environ.get("SUNDIALS_INCLUDEDIR")
SUNDIALS_INST_PREFIX = os.environ.get("SUNDIALS_INST")

if SUNDIALS_LIBDIR or SUNDIALS_INCLUDEDIR:
    SUNDIALS_INCLUDE_DIRS.extend(
        [SUNDIALS_INCLUDEDIR] if SUNDIALS_INCLUDEDIR is not None else []
    )
    SUNDIALS_LIBRARY_DIRS.extend(
        [SUNDIALS_LIBDIR] if SUNDIALS_LIBDIR is not None else []
    )

elif SUNDIALS_INST_PREFIX is not None:
    SUNDIALS_LIBRARY_DIRS.append(os.path.join(SUNDIALS_INST_PREFIX, "lib"))
    SUNDIALS_INCLUDE_DIRS.append(os.path.join(SUNDIALS_INST_PREFIX, "include"))
    info("SUNDIALS installation path set to `{}` via $SUNDIALS_INST.".format(
        SUNDIALS_INST_PREFIX))
else:
    info("Searching for SUNDIALS path...")

    # use pkgconfig to find sundials
    try:
        import pkgconfig
        try:
            cvode_pkgconf = pkgconfig.parse(PKGCONFIG_CVODE)
            for d in cvode_pkgconf.get('library_dirs', []):
                CVODE_LIBRARY_DIRS.append(str(d))
            for d in cvode_pkgconf.get('include_dirs', []):
                CVODE_INCLUDE_DIRS.append(str(d))
            for lib in cvode_pkgconf.get('include_dirs', []):
                CVODE_LIBRARIES.append(str(lib))

            ida_pkgconf = pkgconfig.parse(PKGCONFIG_IDA)
            for d in ida_pkgconf.get('library_dirs', []):
                IDA_LIBRARY_DIRS.append(str(d))
            for d in ida_pkgconf.get('include_dirs', []):
                IDA_INCLUDE_DIRS.append(str(d))
            for lib in ida_pkgconf.get('include_dirs', []):
                IDA_LIBRARIES.append(str(lib))
        except EnvironmentError:
            pass
    except ImportError:
        info("pkgconfig module not found, using preset paths")

if not SUNDIALS_LIBRARIES:
    SUNDIALS_LIBRARIES.append('sundials_nvecserial')

if not IDA_LIBRARIES:
    IDA_LIBRARIES.append('sundials_ida')

if not CVODE_LIBRARIES:
    CVODE_LIBRARIES.append('sundials_cvode')


use_lapack = False
try:
    lapack_opt = get_info('lapack_opt', notfound_action=2)

    if lapack_opt:
        SUNDIALS_INCLUDE_DIRS.extend(lapack_opt.get('include_dirs',[]))
        SUNDIALS_LIBRARY_DIRS.extend(lapack_opt.get('library_dirs',[]))
        SUNDIALS_LIBRARIES.extend(lapack_opt.get('libraries',[]))
        use_lapack = True
    else:
        raise ValueError
    info('Found LAPACK paths via lapack_opt ...')
except:
    info('LAPACK was not detected, disabling sundials solvers')


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    info("=============================================")
    info("parent package is %s" % parent_package)
    info("top path is %s" % top_path)
    info("=============================================")
    config = Configuration('sundials', parent_package, top_path)

    if use_lapack:
        CVODE_LIBRARIES.extend(SUNDIALS_LIBRARIES)
        IDA_LIBRARIES.extend(SUNDIALS_LIBRARIES)
        CVODE_INCLUDE_DIRS.extend(SUNDIALS_INCLUDE_DIRS)
        IDA_INCLUDE_DIRS.extend(SUNDIALS_INCLUDE_DIRS)
        CVODE_LIBRARY_DIRS.extend(SUNDIALS_LIBRARY_DIRS)
        IDA_LIBRARY_DIRS.extend(SUNDIALS_LIBRARY_DIRS)

        cython(['common_defs.pyx'], working_path=BASE_PATH)
        config.add_extension("common_defs",
                             sources=['common_defs.c'],
                             include_dirs=SUNDIALS_INCLUDE_DIRS)

        cython(['ida.pyx'], working_path=BASE_PATH)
        config.add_extension("ida",
                             sources=['ida.c'],
                             depends=['common_defs.c'],
                             include_dirs=IDA_INCLUDE_DIRS,
                             library_dirs=IDA_LIBRARY_DIRS,
                             libraries=IDA_LIBRARIES)

        cython(['cvode.pyx'], working_path=BASE_PATH)
        config.add_extension("cvode",
                             sources=['cvode.c'],
                             depends=['common_defs.c'],
                             include_dirs=CVODE_INCLUDE_DIRS,
                             library_dirs=CVODE_LIBRARY_DIRS,
                             libraries=CVODE_LIBRARIES)
    return config
