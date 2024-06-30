import setuptools

import versioneer

from setup_build import build_ext

cmdclass = {}
cmdclass["build_ext"] = build_ext

setuptools.setup(
    version = versioneer.get_version(),
    packages = setuptools.find_packages('src'),
    package_dir = {'': 'src'},
    cmdclass=versioneer.get_cmdclass(cmdclass),
    # To trick build into running build_ext
    ext_modules = [
        setuptools.Extension('scikits_odes_sundials.x',['x.c'])
    ],
)
