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
date: 17 April 2026
bibliography: paper.bib

---

# Summary

`aircraftdetective` is a Python package that allows users to compute the overall energy efficiency and constituent sub-efficiencies of commercial aircraft, based on publicly available data. It is designed to be used in the context of environmental impact assessment of air travel and energy systems analysis. It supports calculations in physical units, allowing for quick conversion between imperial and metric units and dimensionality checks of function inputs. It is lightweight (<40kB packaged), open-source and distributed under a permissive MIT license.

# Statement of Need

The total carbon emissions of passenger air transport can be described by an equation based on the _Kaya identity_ [@delbecq2023sustainable, Sec.3]:

$$
CO_2 = CO_2/E \times E/RPK \times RPK
$$

where $E$ is the energy used for transport in the form of fuel and $RPK$ is the amount of revenue-passenger kilometers flown. Aircraft efficiency $E/RPK$ is in turn determined by the product of several sub-efficiencies, including the operational efficiency, the aerodynamic efficiency, and the propulsion efficiency [@lee2004aircraft].

$$
\eta = \eta_{\text{ops}} \times \eta_{\text{aero}} \times \eta_{\text{prop}} \times \eta_{\text{struct}}
$$

![Simplified cut-away drawing of a twin-engine, single-isle aircraft with annotated sub-efficiencies contributing to overall efficiency $\eta$.[^acft-subeff-source]](_media/acft_subeff.pdf){width=40%}

[^acft-subeff-source]: Adapted from the public domain image [“Aircraft parts hr”](https://commons.wikimedia.org/wiki/File:Aircraft_parts_hr.jpg) by Wikimedia Commons user "Dtom".

\textit{"The historical development of (...)} [the technological and operational efficiency metrics] \textit{provides a benchmark from which the impacts of environmental improvements on growth can be assessed and a basis for outlining the technological and operational features that determine the substitution rate of capital for operating costs across the air transport system"} [@lee2001historical, pp. 168-169].

Therefore, robust methods for computing aircraft efficiency are central to evaluating the past and future environmental and economic performance of aircraft.

Lee et al. [-@lee2001historical; -@lee2004aircraft] and Babikian et al. [-@babikian2002historical] were the first to provide comprehensive data on the historical efficiency of the global aircraft fleet. The data and trends presented in their work have been highly influential and are frequently reproduced, most prominently in the 2009 report _Transport, Energy and CO$_2$_ [@international2009transport] and derived policy documents. Despite the importance of aircraft efficiency as a metric, a transparent and open-source software implementation for computation has been lacking, making it difficult to reproduce and extend existing work with more recent aircraft data.

The `aircraftdetective` package fills this gap as the first comprehensive Python package for computing the efficiency of commercial aircraft from publicly available information. It uses the governing equations of aerodynamics and thermodynamics together with publicly available aircraft and engine parameters to estimate aircraft sub-efficiencies and overall efficiency. While a publicly available dataset of aircraft specifications is provided by @weinold_zenodo_aircraftdetective_2025, users are free to use their own data.

# State of the Field

Several established software tools address adjacent problems in aviation research. This includes tools for conceptual aircraft design, such as SUAVE [@lukaczyk2015suave] and FAST-OAD [@david2021fastoad], tools for aircraft performance and trajectory prediction, such as OpenAP [@sun2020openap] and EUROCONTROL BADA [@nuic2010bada; @eurocontrol2026bada] or tools for the comparison of methods for aircraft fuel consumption estimation, such as `jetfuelburn` [@weinold2026jetfuelburn]. While these tools are valuable and widely used, their primary purpose is vehicle design study, flight trajectory optimization, or aggregate fuel-burn estimation rather than historical benchmarking of in-service commercial aircraft efficiency from public datasets.

\textbf{Table 1} Selection of representative open-source software packages for aviation research. Abbreviations: Py. = Python, Ftn. = Fortran, Jl. = Julia.

```{=latex}
\begingroup
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}p{0.24\linewidth}p{0.09\linewidth}p{0.59\linewidth}@{}}
\hline
Name & Lang. & Purpose \\
\hline
OpenAP (\href{https://doi.org/10.3390/aerospace7080104}{2020}) & Py. & Performance, fuel flow, emissions, trajectories \\
JetFuelBurn (\href{https://doi.org/10.21105/joss.09280}{2026}) & Py. & Fuel-burn and emissions estimation \\
traffic (\href{https://doi.org/10.21105/joss.01518}{2019}) & Py. & Flight-data analysis and traffic processing \\
FAST-OAD (\href{https://doi.org/10.1088/1757-899X/1024/1/012062}{2021}) & Py. & Overall aircraft design and sizing \\
SUAVE (\href{https://doi.org/10.2514/6.2015-3087}{2015}) & Py. & Conceptual aircraft design \\
TASOPT.jl (\href{https://doi.org/10.21105/joss.08521}{2025}) & Ftn./Jl. & Aircraft design and optimization \\
ADRpy (\href{https://github.com/sobester/ADRpy}{2025}) & Py. & Rapid sizing and performance analysis \\
OpenConcept (\href{https://doi.org/10.2514/6.2018-4979}{2018}) & Py. & Electric/hybrid conceptual design \\
AeroMAPS (\href{https://doi.org/10.59490/joas.2023.7147}{2023}) & Py. & Air-transport scenario assessment \\
AIM2015 (\href{https://doi.org/10.1016/j.tranpol.2019.04.013}{2019}) & Java & Global aviation systems modelling \\
BlueSky (\href{https://resolver.tudelft.nl/uuid:d1131a90-f0ea-4489-a217-ad29987689a1}{2016}) & Py. & Air-traffic simulation \\
OpenMDAO (\href{https://doi.org/10.1007/s00158-019-02211-z}{2019}) & Py. & MDO framework \\
\hline
\end{tabular}
\endgroup
```

`aircraftdetective` therefore fills a different scholarly niche. It provides a transparent and lightweight workflow for reconstructing interpretable aircraft efficiency metrics and sub-efficiency proxies from publicly available aircraft, engine, and US DOT data, including aerodynamic efficiency via lift-to-drag ratio, propulsion efficiency proxies based on cruise TSFC, structural efficiency proxies based on weight per seat, and decomposition of historical efficiency improvements. This is the workflow needed to reproduce and extend the retrospective analyses of Lee et al.\ and Babikian et al.\ [@lee2001historical; @babikian2002historical], but it is not directly offered by the packages above.

# Software Design

The central design choice of `aircraftdetective` is to use tabular data structures based on `pandas` [-@reback2020pandas] DataFrames, augmented with \href{https://pypi.org/project/Pint/}{`pint`} and \href{https://pypi.org/project/Pint-Pandas/}{`pint-pandas`} for physical units and dimensionality checks. This reflects the format in which aircraft and engine data is most frequently stored and published: key public databases such as the @icao_eedb_2026 Databank organize records in tabular form, with each row representing one engine variant and columns covering bypass ratio, pressure ratio, rated thrust, and fuel flow at each certification thrust setting:

\textbf{Table 2} Sample of engine parameters from the @icao_eedb_2026 Databank, with units added by `aircraftdetective` for clarity. Abbreviations: B/P = bypass, TSFC = thrust-specific fuel consumption.

```{=latex}
\begingroup
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{0.17\linewidth}p{0.12\linewidth}p{0.16\linewidth}p{0.14\linewidth}p{0.16\linewidth}p{0.16\linewidth}@{}}
\hline
Engine & B/P Ratio & Pressure Ratio & Rated Thrust & TSFC (takeoff) & TSFC (cruise) \\
\hline
No Unit & No Unit & No Unit & kN & g/(kN s) & g/(kN s) \\
\hline
AE3007A & 5.23 & 18.1 & 33.7 & 11.2 & 18.6 \\
V2533-A5 & 4.46 & 33.4 & 140.6 & 10.1 & 17.5 \\
\hline
\end{tabular}
\endgroup
```

![Takeoff vs. cruise TSFC for engines in the reference dataset, with linear (red) and quadratic (blue) polynomial fits produced by `engines.plot_takeoff_to_cruise_tsfc_ratio()`.](_media/engine_tsfc_figure.pdf){#fig:tsfc width=55%}

The package is organized as a small set of focused modules for calculations, data processing, and utilities. The `calculations` subpackage implements the core physics- and statistics-based methods, such as lift-to-drag estimation from the Breguet range equation, scaling of engine TSFC data, and decomposition of efficiency improvements. The `processing` subpackage handles ingestion and harmonization of public datasets such as US DOT Form 41 Schedule T-100/T2 data, while the `utility` subpackage provides reusable helpers for typed tabular data, polynomial fitting, and plotting.

# Auxiliary Functions

The `aircraftdetective` package includes helper functions for basic problems in atmospheric physics, such as computation of airspeed from mach number based on ambient pressure.

# Interactive Documentation

The package documentation allows users to compute aircraft efficiency directly in the browser, without the need to install the package locally. This is achieved through the use of a [Pyodide](https://pyodide.org/en/stable/) WebAssembly Python kernel. The interactive documentation is available at [aircraftdetective.readthedocs.io](https://aircraftdetective.readthedocs.io).

# AI Usage Disclosure

Large language models (LLMs) were used as programming assistants for repository maintenance, the improvement of unit and integration tests, and the formatting of the JOSS publication. All resulting changes were reviewed by the authors.

# Acknowledgements

This work has been supported by the Swiss Innovation Agency Innosuisse in the context of the WISER flagship project (PFFS-21-72). In addition, Michael P. Weinold gratefully acknowledges the support of the Swiss Study Foundation.

# References
