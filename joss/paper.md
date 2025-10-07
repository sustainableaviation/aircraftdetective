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

The overall environmental impact of air transport can be described through the widely used Kaya identity [@delbecq2023sustainable, Sec.3]:

$$
CO_2 = P \times \frac{E}{P} \times \frac{S}{E} \times \frac{C}{S} \times \frac{CO_2}{C}
$$

where \(P\) is the number of passengers, \(E\) is the energy consumption, \(S\) is the revenue seat-kilometers, and \(C\) is the fuel consumption. Aircraft efficiency in turn is determined by the product of several sub-efficiencies, including the load factor, the operational efficiency, the aerodynamic efficiency, and the propulsion efficiency [@lee2004aircraft].

(WHY IS IT IMPORTANT TO KNOW THIS?)

"The historical development of these two figures of merit provides a benchmark from which the impacts of environmental improvements on growth can be assessed  and a basis for outlining the technological and operational features that determine the substitution rate of capital for operating costs across the air transport system."_ [@lee2001historical, P.168-169].

Robust methods for computing two key parameters are therefore central to any reliable evaluation of the environmental impact of air travel: The fuel burn of the aircraft itself and the environmental burdens associated with fuel production.

Two publications have first provided comprehensive data on the efficiency of the global aircraft fleet [@babikian2002historical] and [@lee2004aircraft] is a standout, offering an innovative and user-friendly solution for fuel burn modeling. However, the methods described in these publications still rely on for comparative analysis of different models. This data was first used in the earlier publication Lee et al. (2001) in Figure 10. It is frequently reproduced, most prominently as Figure 7.2 in the 2009 IEA report Transport, Energy and CO2 (see file Figure 7-2 IEA (2009).pdf, reproduced under CC BY 4.0).

The `aircraftdetective` package fills this gap as the first comprehensive Python package for computing the efficiency of commercial aircraft from publicly available information. While the a dataset is provided in [@weinold_zenodo_aircraftdetective_2025], users are free to use more recent data. It will extend other models or integrated assessment models, such as AeroMAPS [@planes2023aeromaps]

\clearpage

# Auxiliary Functions

The `jetfuelburn` package includes helper functions for basic problems in atmospheric physics, such as computation of airspeed from mach number based on ambient pressure.

# Interactive Documentation

The package documentation allows users to compute fuel burn directly in the browser, without the need to install the package locally. This is achieved through the use of a [Pyodide](https://pyodide.org/en/stable/) Web Assembly Python kernel. The interactive documentation is available at [jetfuelburn.readthedocs.io](https://aircraftdetective.readthedocs.io).

# Acknowledgements

This work has been supported by the Swiss Innovation Agency Innosuisse in the context of the WISER flagship project (PFFS-21-72). In addition, Michael P. Weinold gratefully acknowledges the support of the Swiss Study Foundation.

# References