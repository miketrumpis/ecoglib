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
   April 2021: the ``py3_remake`` branch has been merged as the primary (``master``) branch. Please update already cloned repositories using the following steps

.. code-block:: bash

    $ git checkout py3_remake
    $ git fetch
    $ git branch -D master
    $ git checkout master

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