---
title: 'AircraftDetective: A Python package for calculating the efficiency of commercial aircraft'
tags:
  - Python
  - aviation
  - efficiency
  - index decomposition analysis
authors:
  - name: Michael P. Weinold
    orcid: 0000-0003-4859-2650
    equal-contrib: false
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Russell McKenna
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    orcid: 0000-0001-6758-482X
    affiliation: "1, 2"
affiliations:
 - name: Laboratory for Energy Systems Analysis, PSI Centers for Nuclear Engineering \& Sciences and Energy \& Environmental Sciences, Villigen, Switzerland
   index: 1
 - name: Chair of Energy Systems Analysis, Institute of Energy and Process Engineering, Department of Mechanical and Process Engineering, ETH Zurich, Zurich, Switzerland
   index: 2
date: 07 October 2025
bibliography: paper.bib

---

# Summary

`aircraftdetective` is a Python package that allows users to compute the overall energy efficiency and different sub-efficiencies of commercial aircraft, based on publicly available data. It is designed to be used in the context of environmental impact assessment of air travel and energy systems analysis. It supports calculations in physical units, allowing for quick conversion between imperial and metric units and dimensionality checks of function inputs. It is lightweight (<40kB packaged) and relies only dependencies which are compatible with [the Pyodide distribution](https://pyodide.org/en/stable/), therefore allowing for easy integration into WebAssembly kernels for interactive use in the browser. The package is open-source and distributed under a permissive MIT license. Interactive documentation is available, which allows users to compute aircraft efficiency in the browser without the need to install the package locally.

# Statement of Need

The total carbon emissions of passenger air transport can be described through a framework based on the _Kaya identity_ [@delbecq2023sustainable, Sec.3]:

$$
CO_2 = \frac{CO_2}{E} \times \frac{E}{RPK} \times RPK
$$

where $E$ is the energy used for transport in the form of fuel and $RPK$ is the amount of revenue-passenger kilometers flown \(S\). Aircraft efficiency $E/RPK$ is in turn determined by the product of several sub-efficiencies, including the operational efficiency, the aerodynamic efficiency, and the propulsion efficiency [@lee2004aircraft].

_"The historical development of (...)_ [the technological and operational efficiency metrics] _provides a benchmark from which the impacts of environmental improvements on growth can be assessed  and a basis for outlining the technological and operational features that determine the substitution rate of capital for operating costs across the air transport system."_ [@lee2001historical, P.168-169].

Therefore, robust methods for computing aircraft efficiency are central to evaluating the past and future environmental and economic performance of aircraft.

Lee et al.\ [@lee2001historical][@lee2004aircraft] and Babikian et al.\ [@babikian2002historical] in 2001-2004 were the first to provide comprehensive data on the historical efficiency of the global aircraft fleet. The data and trends presented in their work have been highly influential and are frequently reproduced, most prominently in the 2009 IEA report _Transport, Energy and CO2_ [@international2009transport] and derived policy documents. Despite the importance of aircraft efficiency as a metrics, a transparent and open-source software implementation for computation has been lacking, making it difficult to reproduce and extend existing work with more recent aircraft data.

The `aircraftdetective` package fills this gap as the first comprehensive Python package for computing the efficiency of commercial aircraft from publicly available information. It uses the governing equations of aerodynamics and thermodynamics together with publicly available aircraft and engine parameters to estimate aircraft sub-efficiencies and overall efficiency. While a publicly available dataset of aircraft specifications is provided in [@weinold_zenodo_aircraftdetective_2025], users are free to use their own data.

# Auxiliary Functions

The `jetfuelburn` package includes helper functions for basic problems in atmospheric physics, such as computation of airspeed from mach number based on ambient pressure.

# Interactive Documentation

The package documentation allows users to compute fuel burn directly in the browser, without the need to install the package locally. This is achieved through the use of a [Pyodide](https://pyodide.org/en/stable/) Web Assembly Python kernel. The interactive documentation is available at [jetfuelburn.readthedocs.io](https://aircraftdetective.readthedocs.io).

# Acknowledgements

This work has been supported by the Swiss Innovation Agency Innosuisse in the context of the WISER flagship project (PFFS-21-72). In addition, Michael P. Weinold gratefully acknowledges the support of the Swiss Study Foundation.

# References