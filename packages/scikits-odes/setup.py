import setuptools

import versioneer

setuptools.setup(
    version = versioneer.get_version(),
    packages = setuptools.find_namespace_packages('src'),
    package_dir = {'': 'src'},
    cmdclass=versioneer.get_cmdclass(),
)
