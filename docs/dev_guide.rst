Developers Guide
================

The instructions in this following section are based upon resources:

* `Astropy Dev Workflow <https://docs.astropy.org/en/latest/development/workflow/development_workflow.html>`_
* `Astropy Dev environment <https://docs.astropy.org/en/latest/development/workflow/get_devel_version.html#get-devel>`_
* `Astropy Pull Request Example <https://docs.astropy.org/en/latest/development/workflow/git_edit_workflow_examples.html#astropy-fix-example>`_
* `Sunpy Newcomers' Guide <https://docs.sunpy.org/en/latest/dev_guide/newcomers.html>`_

Fork and Clone Repository
-------------------------
Working of your own forked version of the repository is the preferred approach. To fork the
repository visit the repository page at https://github.com/samaloney/stixpy (make sure you are logged
into github) and click on the fork button at the to right of the page.

Clone your forked version of the

.. code:: bash

    git clone https://github.com/<username>/stixpy.git

It is also advisable to configure the upstream remote at this point

.. code:: bash

    git remote add upstream https://github.com/samaloney/stixpy


Isolated Environment
--------------------
It is highly recommended to work in an isolated python environment there are a number of tools
available to help mange and create isolated environment such as

* `Anaconda <https://anaconda.org>`__
* `Pyenv <https://github.com/pyenv/pyenv>`__
* Python 3.6+ inbuilt venv.

For this documentation we will proceed using Python's venv but the step would be similar in other
tools. For a more detailed overview see the virturl enviroment section of astropy development
workflow `here <https://docs.astropy.org/en/stable/development/workflow/virtual_pythons.html#virtual-envs>`_.



First verify the python version installed by running `'python -V'` or possibly `'python3 -V'` depending
on your system it should be greater then 3.6. Next create a new virtual environment in a directory
outside the git repo and activate.

.. code:: bash

    python3 -m venv /path/to/new/virtual/environment
    source /path/to/new/virtual/environment/bin/activate
    #note the prompt change

The next step is to install the required dependencies (ensure you are working from inside you virtual
environment which can be verified by comparing the path returned from `'python -m pip -V'` to the path
used in the above steps)

.. code:: bash

    python -m pip install -e .
    # to install all development dependencies documentation, testing etc use
    python -m pip install -e .[dev]


Working on code
---------------
It's import to always be working from the most recent version of the so before working on any code
start by getting the latest changes and then creating a branch for you new code.

.. code:: bash

    git checkout master
    git pull upstream master
    git checkout -b <branch-name>

Branch names should ideally be short and descriptive e.g. 'feature-xmlparseing', 'bugfix-ql-fits',
'docs-devguide' and preferably separated by dashes '-' rather than underscores '_'.

Once you are happy with your changes push the changes to github

.. code:: bash

    git add <list of modified or changed files>
    git commit
    git push origin <branch-name>

and open a pull request (PR).

Note a series of checks will be automatically run on code once a PR is created it is recommended
that you locally test the code as outlined below. Additionally it is  recommended that you install
and configure `pre-commit <https://pre-commit.com>`_ which runs various style and code quality
checks before commit.

.. code:: bash

    python -m pip install pre-commit
    pre-commit install


Testing
-------
Testing is built on the `PyTest <https://docs.pytest.org/en/stable/>`_ and there are a number of
ways to run the tests. During development it is often beneficial to run a subset of
test relevant to the current code this can be accomplished by running one of the commands below.

.. code:: bash

    pytest stixcore/path/to/test_file.py:test_one        # run a specific test function
    pytest stixcore/path/to/test_file.py                 # run a specific test file
    pytest stixcore/module                               # run all test for a modules
    pytst                                                # run all tests


Additionally `tox <https://tox.readthedocs.io/en/latest/>`_ is use to create and run tests in
reproducible environments. To see a list of tox environment use `'tox -l'` to run a specific
environment run `'tox -e <envname>'` or to run all simply run `'tox'`.

.. note::

    This is the same process that is run on the CI


Documentation
-------------
Documentation is built using `Sphinx <https://www.sphinx-doc.org/en/master/>`_ similarly to the
tests above this can be run manually or through tox. To run manually cd to the docs directory and
run `'make html'` to run via tox `'tox -e build_docs'`.
