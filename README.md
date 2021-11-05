# ecoglib: visualization and statistics for high density microecog signals

![building workflow](https://github.com/miketrumpis/ecoglib/actions/workflows/build_wheels.yml/badge.svg?branch=master)
![tox workflow](https://github.com/miketrumpis/ecoglib/actions/workflows/run_tox.yml/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/miketrumpis/ecoglib/branch/master/graph/badge.svg?token=DJLCE6UAEN)](https://codecov.io/gh/miketrumpis/ecoglib)

This library contains high-level analysis tools for "topos" and "chronos" aspects of spatio-temporal signals (array timesries).
The packages are organized for

* ``ecoglib.estimation`` statistical estimation tools for spatial, timeseries, and general data
* ``ecoglib.vis`` multiple plotting modules for visualizing multi-channel timeseries, spatial maps, and other results
* ``ecoglib.signal_testing`` signal diagonstics for electrode based recordings

This library builds on top of [ecogdata](https://github.com/miketrumpis/ecogdata), which can preprocess electrode recordings from multiple file types

## Install

First step: set up ``ecogdata`` following instructions here: https://github.com/miketrumpis/ecogdata

Whether you have chosen to use conda or a plain virtual environment, use pip to install ecoglib.

**Choose whether to use PyQt5 or PySide2.**

* PyQt5: this is probably the best option (presently), but it is known not to work on Windows 8
* PySide2: also works, has a less restrictive license

This choice affects the install variation, which is specified in the brackets.
You can either clone & install in one step (using PyQt5 in these example, replace with "pyside2" if needed):

```bash
$ pip install "ecoglib[pyqt] @ git+https://github.com/miketrumpis/ecoglib.git"
```

Or, to track the repository, clone and install separately.

```bash
$ git clone https:github.com/miketrumpis/ecoglib.git
$ pip install ./ecoglib[pyqt]
```

## Install variation for testing

To run tests, install with the ``[pyqt,test]`` variation and run

```bash
$ python -m pytest --pyargs ecoglib
```

## Docs & demo notebooks

To build API documentation and usage demos, you must clone the repository.
Then add ``[pyqt,docs]`` to the install command to get Sphinx and other tools.
You can now run:

```bash
$ cd docs
$ make all
```

Alternatively, install ``jupyter`` and run the notebooks in ``docs/source/usage_demos`` interactively.
