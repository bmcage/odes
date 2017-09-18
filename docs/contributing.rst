Reporting Bugs, Contributing and Releasing
==========================================
We welcome contributions, whether as bug reports, improvements to the code, or
more examples.

Please note that all contributions are subject to our `code of conduct <https://github.com/bmcage/odes/blob/master/CONTRIBUTING.md>`_.

Reporting Bugs
--------------
``odes`` bug tracker is on `GitHub <https://github.com/bmcage/odes>`_.

When reporting bugs, please include the versions of Python, ``odes`` and SUNDIALS,
as well as which OS this appears on.

Getting the code
----------------
The primary repository is at https://github.com/bmcage/odes, and it is the
repository that pull requests should be made against.

Work should be done in a private branch based on master, with pull requests made
against master.

Running the Tests
-----------------
``odes`` uses `tox <https://tox.readthedocs.io/>`_ to manage testing across
different versions.

To install tox, use::

    pip install tox

and to run the tests, inside the top level of the repository, run::

    tox

Adding Examples
---------------
Examples should be added in the ``examples`` folder.

Adding ipython/jupyter notebook examples
........................................
Please submit extra jupyter notebook examples of usage of ``odes``. Example
notebooks should go in ``ipython_examples``, and add a short description to
``ipython_examples/README.md``.

Creating a New Release
----------------------

1. Set in ``common.py`` version string and ``DEV=False``, commit this.
2. On GitHub, `draft a new release <https://github.com/bmcage/odes/releases>`_ by clicking the appropriate button. Give correct version number, and hit release. This will upload the release for a DOI to `Zenodo <https://zenodo.org>`_ as draft.
3. Go to uploads in `Zenodo <https://zenodo.org>`_, edit the uploaded new release, save and hit the publish button. This will generate a DOI.
4. Update to PyPI: ``python setup.py sdist --formats=gztar register upload``
5. Update version string to a higher number in ``common.py``, and ``DEV=True``, next copy the DOI badge of Zenodo in the ``README.md``, commit these two files.

For the documentation, you need following packages::

    sudo apt-get install python-sphinx python-numpydoc python-mock

After local install, create the new documentation via

1. Go to the sphinx directory: ``cd sphinxdoc``
2. Create the documentation: ``make html``
3. Upload the new html doc.
