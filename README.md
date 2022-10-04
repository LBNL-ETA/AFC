# Advanced Fenestration Control (AFC)

![Actions Status](https://github.com/LBNL-ETA/AFC/workflows/Syntax/badge.svg)
![Actions Status](https://github.com/LBNL-ETA/AFC/workflows/UnitTests/badge.svg)

#### Predictive Control Solution for Advanced Fenestration and Integrated Energy Systems
----------------------------------------------------------------------------------------

## General
This package was developed to host a suite of control algorithms developed by Lawrence Berkeley National Laboratory's [Windows Group](https://windows.lbl.gov/). Most controls are based on [Model Predictive Control](https://en.wikipedia.org/wiki/Model_predictive_control). This framework utilizes the Distributed Optimal Energy Resources ([DOPER](https://github.com/LBNL-ETA/DOPER)) to implement the algorithms.

The functionality of an *Integrated Controller* (i.e., Advanced Fenestration Control) is illustrated in the following figure. Setpoints for light levels and occupant glare sensitivity are provided as inputs. At each five minute timestep the controller receives updated information from exterior and interior sensors and controls the elctric lights and dynamc facade (e.g., motorized blinds, electrochromic windows) accordingly.

![illustrate_system.jpg](https://github.com/LBNL-ETA/AFC/blob/master/docs/illustrate_system.jpg)

*Please note that the AFC package and especially the examples are still under development. Please open an issue for specific questions.*

## Getting Started
The following link permits users to clone the source directory containing the [AFC](https://github.com/LBNL-ETA/AFC) package.

The package depends on external modules which can be installed from pypi with ```pip install .```.

In addition the [Radiance](https://www.radiance-online.org/download-install/radiance-source-code/latest-release) tool needs to be installed. Binaries can also be found [here](https://github.com/LBNL-ETA/Radiance/releases).

## Example
To test the installation and illustrate the functionality of AFC, the following command can be executed to run the [example_1.py](https://github.com/LBNL-ETA/AFC/blob/master/examples/example_1.py).

```python
python examples/example_1.py
```

The output should be:

```
Running AFC Example1...
Configuration: three zone electrochromic window

Log-message:
Solver                  CBC 2.10.5
Duration [s]            3.28
Objective [$]           29.48           13.94 (Total Cost)
Cost [$]                16.15 (Energy)  13.21 (Demand)

Facade actuation during the day (when DNI > 0).
Facade 0 = bottom zone, Facade 1 = middle zone, Facade 2 = top zone
State 0.0 = fully tinted, State 1.0 and 2.0 = intermediate tint, state 3.0 = clear (double low-e)

                     Facade State 0  Facade State 1  Facade State 2
2022-07-01 05:00:00             3.0             3.0             3.0
2022-07-01 06:00:00             3.0             3.0             3.0
2022-07-01 07:00:00             3.0             2.0             3.0
2022-07-01 08:00:00             1.0             1.0             3.0
2022-07-01 09:00:00             0.0             0.0             3.0
2022-07-01 10:00:00             0.0             0.0             3.0
2022-07-01 11:00:00             0.0             0.0             3.0
2022-07-01 12:00:00             0.0             0.0             3.0
2022-07-01 13:00:00             0.0             0.0             3.0
2022-07-01 14:00:00             0.0             0.0             3.0
2022-07-01 15:00:00             0.0             0.0             3.0
2022-07-01 16:00:00             1.0             2.0             3.0
2022-07-01 17:00:00             1.0             3.0             3.0
2022-07-01 18:00:00             3.0             3.0             3.0
2022-07-01 19:00:00             3.0             3.0             3.0
```

Additional examples with interactive Jupyter Notebooks can be found in the [examples](https://github.com/LBNL-ETA/AFC/blob/master/examples).

## License
Advanced Fenestration Control (AFC) Copyright (c) 2022, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit other to do so.

## Cite
To cite the AFC package, please use:

```bibtex
@article{gehbauer2020assessment,
title={An assessment of the load modifying potential of model predictive controlled dynamic facades within the California context},
author={Gehbauer, Christoph and Blum, David and Wang, Taoning and Lee, Eleanor S},
journal={Energy and Buildings},
volume={210},
pages={109762},
year={2020},
publisher={Elsevier}
}
```
