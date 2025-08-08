# %%
import pandas as pd
import numpy as np
from pathlib import Path

from aircraftdetective import ureg
from aircraftdetective.utility import plotting
from aircraftdetective.utility import tabular
from aircraftdetective.utility.statistics import r_squared
from typing import Any
import matplotlib.pyplot as plt


def determine_takeoff_to_cruise_tsfc_ratio(
    path_excel_engine_data_for_calibration: str
) -> dict[str, Any]:
    r"""
    Given a path to an Excel file with engine TSFC data for takeoff and cruise,
    computes two fitted polynomials (linear and quadratic).
    Also returns the $R^2$ values for both fits and the cleaned DataFrame with engine data.

    $$
    \begin{aligned}
    TSFC_{cruise} &= \beta_0 + \beta_1 \cdot TSFC_{takeoff} + \epsilon \\
    TSFC_{cruise} &= \beta_0 + \beta_1 \cdot TSFC_{takeoff} + \beta_2 \cdot TSFC_{takeoff}^2 + \epsilon
    \end{aligned}
    $$

    where

    | Symbol           | Description                                      |
    |------------------|--------------------------------------------------|
    | $TSFC_{cruise}$  | Thrust-Specific Fuel Consumption at cruise       |
    | $TSFC_{takeoff}$ | Thrust-Specific Fuel Consumption at takeoff      |
    | $\beta_0$        | Intercept of the fitted polynomial               |
    | $\beta_1$        | Slope of the fitted polynomial                   |
    | $\beta_2$        | Quadratic coefficient of the fitted polynomial   |
    | $\epsilon$       | Error term                                       |

    Notes
    -----

    The Excel file must have a sheet named `Data` with at least the columns `[Engine Identification, TSFC (cruise), TSFC (takeoff)]`.
    The first row of the sheet must contain the column names.
    The second row must contain the units of the columns in square brackets `[]`, [in a format supported by Pint](https://pint.readthedocs.io/en/stable/user/formatting.html#pint-format-types).
    Columns with no relevant units must be marked with `No Unit`.

    | Engine Identification | TSFC (cruise)  | TSFC (takeoff) |
    |-----------------------|----------------|----------------|
    | **No Unit**           | **[g/(kN*s)]** | **[g/(kN*s)]** |
    | D-30KU                | 19.82788889    | 13.87952222    |
    | D-100                 | 15.2958        | 8.101108889    |
    | CFE738                | 18.26998333    | 10.45213       |

    See Also
    --------
    - [`numpy.polynomial.polynomial.Polynomial.fit()`](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html)
    - [Section 4.6.2 in Sadraey, 2nd Edition (2023)](https://doi.org/10.1201/9781003279068)
    - [Thrust-Specific Fuel Consumption on Wikipedia](https://en.wikipedia.org/wiki/Thrust-specific_fuel_consumption)

    Parameters
    ----------
    path_excel_engine_data_for_calibration : str
        _description_

    Returns
    -------
    tuple[pd.DataFrame, np.polynomial.Polynomial, np.polynomial.Polynomial, float, float]
        DataFrame with engine data and fitted polynomials with their R² values

    Raises
    ------
    ValueError
        If the Excel file does not contain the required columns or units.
    """
    df_engines = pd.read_excel(
        io=path_excel_engine_data_for_calibration,
        sheet_name='Data',
        header=[0, 1],
        engine='openpyxl', 
    )
    df_engines = df_engines.pint.quantify(level=1)

    if ('TSFC (cruise)') not in df_engines.columns or ('TSFC (takeoff)') not in df_engines.columns:
        raise ValueError(f"Excel file must contain 'TSFC (cruise)' and 'TSFC (takeoff)' columns.")

    df_engines = df_engines[df_engines[['TSFC (cruise)','TSFC (takeoff)']].notna().all(axis=1)]
    df_engines_grouped = df_engines.groupby(['Engine Identification'], as_index=False).agg(
        {
            'TSFC (cruise)' : 'mean',
            'TSFC (takeoff)' : 'mean',
        }
    )

    df_engines_grouped['TSFC (cruise)'] = df_engines_grouped['TSFC (cruise)'].astype("pint[g/(kN*s)]")
    df_engines_grouped['TSFC (takeoff)'] = df_engines_grouped['TSFC (takeoff)'].astype("pint[g/(kN*s)]")

    x = df_engines_grouped['TSFC (takeoff)'].astype("float64")
    y = df_engines_grouped['TSFC (cruise)'].astype("float64")

    linear_fit = np.polynomial.Polynomial.fit(
        x=x,
        y=y,
        deg=1,
    )
    polynomial_fit = np.polynomial.Polynomial.fit(
        x=x,
        y=y,
        deg=2,
    )

    r_squared_linear = r_squared(y, linear_fit(x))
    r_squared_polynomial = r_squared(y, polynomial_fit(x))

    return {
        'df_engines': df_engines_grouped,
        'pol_linear_fit': linear_fit,
        'pol_quadratic_fit': polynomial_fit,
        'r_squared_linear_fit': r_squared_linear,
        'r_squared_quadratic_fit': r_squared_polynomial
    }


def plot_takeoff_to_cruise_tsfc_ratio(
    df_engines: pd.DataFrame,
    linear_fit: np.polynomial.Polynomial,
    polynomial_fit: np.polynomial.Polynomial,
):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    df_engines : pd.DataFrame
        _description_
    linear_fit : np.polynomial.Polynomial
        _description_
    polynomial_fit : np.polynomial.Polynomial
        _description_
    """
    
    # DATA PREPARATION ##########

    df_engines = df_engines.copy()
    df_engines = df_engines.pint.dequantify()

    max_tsfc_takeoff = df_engines['TSFC (takeoff)'].max().item()
    min_tsfc_takeoff = df_engines['TSFC (takeoff)'].min().item()

    list_tsfc_takeoff = np.linspace(
        start=min_tsfc_takeoff - 5,
        stop=max_tsfc_takeoff + 5,
        num=100
    )
    df_linear_fit = pd.DataFrame(
        {
            'TSFC (takeoff)': list_tsfc_takeoff,
            'TSFC (cruise)': linear_fit(list_tsfc_takeoff)
        }
    )
    df_polynomial_fit = pd.DataFrame(
        {
            'TSFC (takeoff)': list_tsfc_takeoff,
            'TSFC (cruise)': polynomial_fit(list_tsfc_takeoff)
        }
    )

    # SETUP ######################

    fig, ax = plotting.set_figure_and_axes()

    # AXIS LIMITS ################

    #ax.set_xlim(5, 20)
    #ax.set_ylim(12, 25)

    # TICKS AND LABELS ###########

    ax.set_xlabel('TSFC (takeoff) [g/kNs]')
    ax.set_ylabel('TSFC (cruise) [g/kNs]')

    # PLOTTING ###################

    scatterplot = ax.scatter(
        df_engines['TSFC (takeoff)'],
        df_engines['TSFC (cruise)'],
        marker='o',
        edgecolor='k',
        plotnonfinite=True # for NaN values
    )
    ax.plot(
        df_linear_fit['TSFC (takeoff)'],
        df_linear_fit['TSFC (cruise)'],
        color='red',
        linestyle='--',
        label='Polynomial Fit'
    )
    ax.plot(
        df_polynomial_fit['TSFC (takeoff)'],
        df_polynomial_fit['TSFC (cruise)'],
        color='blue',
        linestyle='--',
        label='Linear Fit'
    )

    # LEGEND ####################

    fig.show()


def scale_engine_data_from_icao_emissions_database(
    path_excel_engine_data_icao_in: Path,
    scaling_polynomial: np.polynomial.Polynomial
) -> pd.DataFrame:
    r"""
    Given a path or URL to the ICAO Aircraft Engine Emissions Databank Excel file,
    scales the TSFC (takeoff) values to cruise using a provided polynomial.

    $$
    TSFC_{cruise} = f(TSFC_{takeoff})
    $$

    where $f$ is the scaling polynomial.

    Notes
    -----
    The returns DataFrame has only a subset of columns relevant to engine efficiency.
    Columns labeled `🆕` are new columns computed by this function:

    | Output DataFrame Column Name | Unit/Data Type    |
    |------------------------|-------------------------|
    | Engine Identification  | `str`                   |
    | Final Test Date        | `float`                 |
    | 🆕 TSFC (cruise)       | `pint[g/(kN*s)]`        |
    | 🆕 TSFC (takeoff)      | `pint[g/(kN*s)]`        |
    | Fuel Flow (takeoff)    | `pint[kg/s]`            |
    | Fuel Flow (climbout)   | `pint[kg/s]`            |
    | Fuel Flow (approach)   | `pint[kg/s]`            |
    | Fuel Flow (idle)       | `pint[kg/s]`            |
    | B/P Ratio              | `pint[dimensionless]`   |
    | Pressure Ratio         | `pint[dimensionless]`   |
    | Rated Thrust           | `pint[kN]`              |

    See Also
    --------
    - [`src.aircraftdetective.calculations.engines.determine_takeoff_to_cruise_tsfc_ratio`][]
    - [ICAO Aircraft Engine Emissions Databank](https://www.easa.europa.eu/en/domains/environment/icao-aircraft-engine-emissions-databank)

    Parameters
    ----------
    path_excel_engine_data_icao_in : str
        Path or URL to the ICAO Aircraft Engine Emissions Databank Excel file.
        Equivalent to parameter `io` in [`pandas.read_excel`](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html#pandas-read-excel).
    scaling_polynomial : np.polynomial.Polynomial
        [`numpy.polynomial.polynomial.Polynomial`](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.html#numpy.polynomial.polynomial.Polynomial) Polynomial to scale TSFC (takeoff) to TSFC (cruise).
    """

    df_engines = pd.read_excel(
        io=path_excel_engine_data_icao_in,
        sheet_name='Gaseous Emissions and Smoke',
        header=0,
        engine='openpyxl',
    )
    df_engines['Final Test Date'] = df_engines['Final Test Date'].dt.year.astype('Int64')

    df_engines = tabular.rename_columns_and_set_units(
        df=df_engines,
        return_only_renamed_columns=True,
        column_names_and_units=[
            ("Engine Identification", "Engine Identification", str),
            ("Final Test Date", "Final Test Date", "Int64"), # 'Int64' to ensure null values do not raise an error
            ("Fuel Flow T/O (kg/sec)", "Fuel Flow (takeoff)", "pint[kg/s]"),
            ("Fuel Flow C/O (kg/sec)", "Fuel Flow (climbout)", "pint[kg/s]"),
            ("Fuel Flow App (kg/sec)", "Fuel Flow (approach)", "pint[kg/s]"),
            ("Fuel Flow Idle (kg/sec)", "Fuel Flow (idle)", "pint[kg/s]"),
            ("B/P Ratio", "B/P Ratio", "pint[dimensionless]"),
            ("Pressure Ratio", "Pressure Ratio", "pint[dimensionless]"),
            ("Rated Thrust (kN)", "Rated Thrust", "pint[kN]")
        ],
    )

    df_engines = df_engines.groupby(['Engine Identification'], as_index=False).agg('mean')

    df_engines['TSFC (takeoff)'] = df_engines['Fuel Flow (takeoff)'] / df_engines['Rated Thrust']
    df_engines['TSFC (takeoff)'] = df_engines['TSFC (takeoff)'].astype("pint[g/(kN*s)]") # commonly used unit for TSFC, to ensure compatibility with the polynomial

    df_engines['TSFC (cruise)'] = df_engines['TSFC (takeoff)'].apply(lambda x: scaling_polynomial(x))
    df_engines['TSFC (cruise)'] = df_engines['TSFC (cruise)'].astype("pint[g/(kN*s)]") # commonly used unit for TSFC

    return df_engines