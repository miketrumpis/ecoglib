# ecoglib: visualization and statistics for high density microecog signals

This library contains high-level analysis tools for "topos" and "chronos" aspects of spatio-temporal signals (array timesries).
The packages are organized for

* ``ecoglib.estimation`` statistical estimation tools for spatial, timeseries, and general data
* ``ecoglib.vis`` multiple plotting modules for visualizing multi-channel timeseries, spatial maps, and other results
* ``ecoglib.signal_testing`` signal diagonstics for electrode based recordings

This library builds on top of [ecogdata](https://github.com/miketrumpis/ecogdata), which can preprocess electrode recordings from multiple file types

## Install

First step: set up ``ecogdata`` following instructions here: https://github.com/miketrumpis/ecogdata

Next, clone and install ecogdata, ecoglib and dependencies:

```bash
$ git clone https://bitbucket.org/tneuro/ecoglib
```

Update requirements. Using pip:

```bash
$ pip install -r ecoglib/requirements.txt
```

Using conda: **change "tables" to "pytables" in requirements.txt** (and add conda forge channel to your settings to avoid "-c")

```bash
$ conda install -c conda-forge -n <your-env-name> --file ecoglib/requirements.txt
```

Last, install ecoglib in any way you choose. 
I use "editable" mode to avoid re-installing after git pulls: pip install -e 

```bash
$ pip install -e ./ecoglib
```

Run tests to check install:

```bash
$ python -m pytest ecoglib
```

## Docs & demo notebooks

To build API documentation and usage demos, first install requirements in requirements-docs.txt.
Then:

```bash
$ cd docs
$ make all
```

Alternatively, install ``jupyter`` and run the notebooks in ``docs/source/usage_demos`` interactively.