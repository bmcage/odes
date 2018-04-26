
import io
import os
from os.path import join
from distutils.log import info

from numpy.distutils.command.build_ext import build_ext as _build_ext

PKGCONFIG_CVODE = 'sundials-cvode-serial'
PKGCONFIG_IDA = 'sundials-ida-serial'


def write_pxi(filename, definitions):
    """
    Write a cython include file (.pxi), `filename`, with the definitions in the
    `definitions` mapping.
    """
    with io.open(filename, mode='w', encoding='utf-8') as pxi_file:
        for name, val in definitions.items():
            pxi_file.write(u"DEF {name} = {val}\n".format(name=name, val=val))
    return filename


def get_sundials_config_pxi(include_dirs, dist):
    """
    Create pxi file containing some of sundials build config

    Don't ask why this is a function, something crazy about
    distutils/numpy not setting _setup_distribution at the right time or
    something...
    """
    SUNDIALS_CONFIG_H = "sundials/sundials_config.h"
    BASE_PATH = join('scikits', 'odes', 'sundials')

    config_cmd = dist.get_command_obj("config")
    if config_cmd.check_macro_true(
        "SUNDIALS_DOUBLE_PRECISION", headers=[SUNDIALS_CONFIG_H],
        include_dirs=include_dirs
    ):
        SUNDIALS_FLOAT_TYPE = '"double"'
        info("Found sundials built with double precision.")
    elif config_cmd.check_macro_true(
        "SUNDIALS_SINGLE_PRECISION", headers=[SUNDIALS_CONFIG_H],
        include_dirs=include_dirs
    ):
        SUNDIALS_FLOAT_TYPE = '"single"'
        info("Found sundials built with single precision.")
    elif config_cmd.check_macro_true(
        "SUNDIALS_EXTENDED_PRECISION", headers=[SUNDIALS_CONFIG_H],
        include_dirs=include_dirs
    ):
        SUNDIALS_FLOAT_TYPE = '"extended"'
        info("Found sundials built with extended precision.")
    else:
        # fall back to double
        SUNDIALS_FLOAT_TYPE = '"double"'
        info("Failed to find sundials precision, falling back to double...")

    return write_pxi(join(BASE_PATH, "sundials_config.pxi"),
        dict(SUNDIALS_FLOAT_TYPE=SUNDIALS_FLOAT_TYPE),
    )


class build_ext(_build_ext):
    """
        Custom distutils command which encapsulates api_gen pre-building,
        Cython building, and C compilation.
        Also handles making the Extension modules, since we can't rely on
        NumPy being present in the main body of the setup script.
    """
    def _get_cython_ext(self):
        from numpy.distutils.system_info import get_info
        from setuptools import Extension

        base_path = join('scikits', 'odes', 'sundials')
        base_module = "scikits.odes.sundials"

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
            return []

        CVODE_LIBRARIES.extend(SUNDIALS_LIBRARIES)
        IDA_LIBRARIES.extend(SUNDIALS_LIBRARIES)
        CVODE_INCLUDE_DIRS.extend(SUNDIALS_INCLUDE_DIRS)
        IDA_INCLUDE_DIRS.extend(SUNDIALS_INCLUDE_DIRS)
        CVODE_LIBRARY_DIRS.extend(SUNDIALS_LIBRARY_DIRS)
        IDA_LIBRARY_DIRS.extend(SUNDIALS_LIBRARY_DIRS)

        sundials_pxi = get_sundials_config_pxi(SUNDIALS_INCLUDE_DIRS,
                self.distribution)

        return [
            Extension(
                base_module + '.' + "common_defs",
                sources = [join(base_path, 'common_defs.pyx')],
                include_dirs=SUNDIALS_INCLUDE_DIRS,
            ),
            Extension(
                base_module + '.' + "cvode",
                sources = [join(base_path, 'cvode.pyx')],
                include_dirs=CVODE_INCLUDE_DIRS,
                library_dirs=CVODE_LIBRARY_DIRS,
                libraries=CVODE_LIBRARIES,
            ),
            Extension(
                base_module + '.' + "ida",
                sources = [join(base_path, 'ida.pyx')],
                include_dirs=IDA_INCLUDE_DIRS,
                library_dirs=IDA_LIBRARY_DIRS,
                libraries=IDA_LIBRARIES,
            ),
        ]


    def run(self):
        """ Distutils calls this method to run the command """
        from Cython.Build import cythonize
        self.extensions.extend(cythonize(self._get_cython_ext()))
        _build_ext.run(self) # actually do the build

