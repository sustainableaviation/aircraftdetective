# %%
from aircraftdetective import ureg

import pandas as pd
import numpy as np

import re

from aircraftdetective.utility import plotting
from aircraftdetective.utility import tabular
from aircraftdetective.utility.statistics import r_squared
from aircraftdetective import config


def determine_takeoff_to_cruise_tsfc_ratio(
    path_excel_engine_data_for_calibration: str
) -> tuple[np.polynomial.Polynomial, np.polynomial.Polynomial, float, float]:
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


def scale_engine_data_from_icao_emissions_database(
    path_excel_engine_data_icao_in: str,
    path_excel_engine_data_icao_out: str,
    scaling_polynomial: np.polynomial.Polynomial
) -> pd.DataFrame:
    r"""_summary_

    Notes
    -----

    The Excel file must have a sheet named `Data` with at least the column `TSFC (takeoff)`.
    The first row of the sheet must contain the column names.
    The second row must contain the units of the column in square brackets `[]`, [in a format supported by Pint](https://pint.readthedocs.io/en/stable/user/formatting.html#pint-format-types).

    | TSFC (takeoff) |
    |----------------|
    | **[g/(kN*s)]** |
    | 13.87952222    |
    | 8.101108889    |
    | 10.45213       |

    Parameters
    ----------
    path_excel_engine_data_icao_in : str
        _description_
    path_excel_engine_data_icao_out : str
        _description_
    scaling_polynomial : np.polynomial.Polynomial
        _description_
    """

    df_engines = pd.read_excel(
        io=path_excel_engine_data_icao_in,
        sheet_name='Gaseous Emissions and Smoke',
        header=0,
        converters={'Final Test Date': lambda x: int(pd.to_datetime(x).year)},
        engine='openpyxl',
    )

    df_engines = tabular.rename_columns_and_set_units(
        df=df_engines,
        return_only_renamed_columns=True,
        column_names_and_units=[
            ("Engine Identification", "Engine Identification", str),
            ("Final Test Date", "Final Test Date", float),
            ("Fuel Flow T/O (kg/sec)", "Fuel Flow (takeoff)", "pint[kg/s]"),
            ("Fuel Flow C/O (kg/sec)", "Fuel Flow (climbout)", "pint[kg/s]"),
            ("Fuel Flow App (kg/sec)", "Fuel Flow (approach)", "pint[kg/s]"),
            ("Fuel Flow Idle (kg/sec)", "Fuel Flow (idle))", "pint[kg/s]"),
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

    tabular.export_typed_dataframe_to_excel(
        df=df_engines,
        path=path_excel_engine_data_icao_out
    )

    return df_engines