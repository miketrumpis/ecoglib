.. |date| date::
.. |time| date:: %H:%M

Welcome to ecoglib's documentation!
===================================

This library is home to a collection of statistical estimation packages for spatial and temporal processes, as well as plotting methods for the same.

Quick install
-------------

Generic steps for cloning ecoglib and installing follow.
If you are using `conda`_ or `pyenv`_ then activate environments and/or change the install procedure accordingly.

.. attention::
   It is currently necessary to change to the Python3 migration branch using ``git checkout py3_remake``

.. code-block:: bash

    $ git clone git@bitbucket.org:tneuro/ecoglib.git
    $ cd ecoglib
    $ git checkout py3_remake
    $ pip install -r requirements.txt
    $ pip install .

More about ecoglib
------------------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   api
   usage_demos/index

Misc
----

Documents rebuilt |date| at |time|.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _conda: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
.. _pyenv: https://github.com/pyenv/pyenv