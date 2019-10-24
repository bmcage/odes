Reporting Bugs, Contributing and Releasing
==========================================
We welcome contributions, whether as bug reports, improvements to the code, or
more examples.

Please note that all contributions are subject to our `code of conduct <https://github.com/bmcage/odes/blob/master/CONTRIBUTING.md>`_.

Reporting Bugs
--------------
``odes`` bug tracker is on GitHub_.

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

Building the Docs
-----------------

The documentation for ``odes`` is split into two parts, the main docs (of which
this is a part), and the API docs. Both the main docs and API docs use sphinx_
to build the docs, and running ``make html`` inside either of the associated
directories will cause sphinx to create a html version of the docs.

The main docs are located in the ``docs`` directory, and the requirements for
building it are in ``docs/requirements.txt``.

The API docs are located in the ``apidocs`` directory, and the requirements for
building it are in ``apidocs/requirements.txt``.


Creating a New Release
----------------------

There are five steps to creating a new ``odes`` release:

1. Make a non-development version.
2. Create a new release on GitHub_.
3. Publish the new release on Zenodo_.
4. Upload the new release to PyPI_.
5. Bump the version to the next development version.

The main docs should automatically build on readthedocs_, and the API docs should
be built by doctr_. You should check that the docs have updated once you have
make the release. If docs are not updated automatically, login to readthedocs_
go to scikits, builds, and build latest and master manually.

Making a non-development version
................................

To make a non-development version, inside ``common.py`` change ``DEV=True`` to ``DEV=False``, and if needed, modify ``MAJOR``, ``MINOR`` and ``MICRO`` to set the new release version.
Then commit only these changes and push them to the main repository (bmcage/odes).

Creating a new release on GitHub_
.................................

On GitHub, `draft a new release <https://github.com/bmcage/odes/releases>`_ by clicking the appropriate button. Use the version number from the non-development commit as the title, and hit release. This will upload the release for a DOI to Zenodo_ as draft.

Publishing the new release on Zenodo_
.....................................

Go to uploads in Zenodo_, edit the uploaded new release, adding addition information as needed such as ORCID_, save and hit the publish button. This will generate a DOI.

Uploading the new release to PyPI_
..................................

Make sure the current checkout is the non-development commit. To make sure no
additional changes are included in the release, run::

    git stash save --no-keep-index --all

This saves the current working directory, then cleans it. The changes can be
retrieved by running ``git stash pop`` (but you should not do this until the
end).

In the cleaned repository, run::

    python setup.py sdist --formats=gztar

which creates a ``dist`` directory containing a ``tar.gz`` file, the sdist for
the release. To upload the sdist to PyPI_, run::

    python3 -m pip install --user --upgrade twine
    python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

See https://packaging.python.org/tutorials/distributing-packages/#uploading-your-project-to-pypi for more information about uploading to PyPI_.

Bumping the version to the next development version
...................................................

Modify ``MAJOR``, ``MINOR`` and ``MICRO`` in ``common.py`` to a later version (increasing ``MICRO`` by 1 is sufficient). Also in ``common.py``, change back to ``DEV=True``. Finally, copy the DOI badge of of the latest release from Zenodo_ to the ``README.md``, and commit only these two files. You can now run ``git stash pop`` to retrieve what you were working on.

.. _Zenodo: https://zenodo.org
.. _Github: https://github.com/bmcage/odes
.. _PyPI: https://pypi.org
.. _readthedocs: https://readthedocs.org
.. _doctr: https://drdoctr.github.io/doctr/
.. _ORCID: https://orcid.org/
.. _sphinx: http://www.sphinx-doc.org/
