# Advanced Fenestration Control (AFC)

![Actions Status](https://github.com/LBNL-ETA/AFC/workflows/Syntax/badge.svg)
![Actions Status](https://github.com/LBNL-ETA/AFC/workflows/UnitTests/badge.svg)

#### Predictive Control Solution for Advanced Fenestration and Integrated Energy Systems
----------------------------------------------------------------------------------------

## General
This package was developed to host a suite of control algorithms developed by Lawrence Berkeley National Laboratory's [Windows Group](https://windows.lbl.gov/). Most controls are based on [Model Predictive Control](https://en.wikipedia.org/wiki/Model_predictive_control). This framework utilizes the Distributed Optimal Energy Resources ([DOPER](https://github.com/LBNL-ETA/DOPER)) to implement the algorithms.

The functionality of an *Integrated Controller* (i.e., Advanced Fenestration Control) is illustrated in the following figure. Setpoints for light levels and occupant glare sensitivity are provided as inputs. At each five minute timestep the controller receives updated information from exterior and interior sensors and controls the elctric lights and dynamc facade (e.g., motorized blinds) accordingly.

![illustrate_system.jpg](https://github.com/LBNL-ETA/AFC/blob/master/docs/illustrate_system.jpg)

*Please note that the AFC package and especially the examples are still under development. Please open an issue for specific questions.*

## Getting Started
The following link permits users to clone the source directory containing the [AFC](https://github.com/LBNL-ETA/AFC) package.

The package depends on external modules which can be installed from pypi with ```pip install -r requirements.txt```.

## Example
To test the installation and illustrate the functionality of AFC, the following command can be executed to run the [BasicTest.py](https://github.com/LBNL-ETA/AFC/blob/master/examples/dummy.py).

```python
python examples/BasicTest.py
```

The output should be:

```
==> Completed 5 out of 5 tests. <==
Congratulations, the Advanced Fenestration Control was successfully installed!
Thank you for your interest and checkout more examples in the example folder!
```

Additional examples with interactive Jupyter Notebooks can be found in the [examples](https://github.com/LBNL-ETA/AFC/blob/master/examples).

## License
Advanced Fenestration Control (AFC) Copyright (c) 2022, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit other to do so.

## Cite
To cite the AFC package, please use:

Gehbauer, C., Blum, D., Wang, T. and Lee, E.S., 2020. An assessment of the load modifying potential of model predictive controlled dynamic facades within the California context. Energy and Buildings, 210, p.109762.
