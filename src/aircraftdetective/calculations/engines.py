# %%
import pandas as pd
import numpy as np
from typing import Any
import math

import pint
ureg = pint.get_application_registry()
from aircraftdetective.utility import tabular
from aircraftdetective.utility.statistics import (
    _compute_polynomials_from_dataframe
)
from aircraftdetective.utility.physics import _calculate_atmospheric_conditions

from aircraftdetective.data.hyperlinks import (
    PATH_ZENODO_ENGINE_TSFC_CALIBRATION_FILE,
    PATH_EASA_ENGINE_EMISSIONS_DATABANK_FILE,
)


def determine_takeoff_to_cruise_tsfc_ratio(
    degree: int = 2,
    plot: bool = False,
    path_excel_engine_data_for_calibration: str = PATH_ZENODO_ENGINE_TSFC_CALIBRATION_FILE,
) -> dict[str, Any]:
    r"""
    Given a path to an Excel file with engine TSFC data for takeoff and cruise,
    computes two fitted polynomials (linear and quadratic).
    Also returns the $R^2$ values for both fits and the cleaned DataFrame with engine data.
    
    A degree 1 polynomial (linear fit) is implemented as:
    $$
    TSFC_{cruise} = \beta_0 + \beta_1 \cdot TSFC_{takeoff} + \epsilon
    $$
    A degree 2 polynomial (quadratic fit) is implemented as:
    $$
    TSFC_{cruise} = \beta_0 + \beta_1 \cdot TSFC_{takeoff} + \beta_2 \cdot TSFC_{takeoff}^2 + \epsilon
    $$
    etc., where

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
    - [`aircraftdetective.calculations.engines.scale_engine_data_from_icao_emissions_database`][]
    
    References
    ----------
    - [Section 4.6.2 in Sadraey, 2nd Edition (2023)](https://doi.org/10.1201/9781003279068)
    - [Thrust-Specific Fuel Consumption on Wikipedia](https://en.wikipedia.org/wiki/Thrust-specific_fuel_consumption)

    Notes
    -----
    With no parameters passed, the function will download the relevant Excel file 
    from the [`aircraftdetective` Zenodo repository](https://doi.org/10.5281/zenodo.14382100).

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
    ValueError
        If the degree parameter is not a positive integer.
    """
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be a positive integer.")

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

    return _compute_polynomials_from_dataframe(
        df=df_engines_grouped,
        col_name_x='TSFC (takeoff)',
        list_col_names_y=['TSFC (cruise)'],
        degree=degree,
        plot=plot
    )


def scale_engine_data_from_icao_emissions_database(
    scaling_polynomial: np.polynomial.Polynomial,
    path_excel_engine_data_icao_in: str = PATH_EASA_ENGINE_EMISSIONS_DATABANK_FILE,
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
    - [`aircraftdetective.calculations.engines.determine_takeoff_to_cruise_tsfc_ratio`][]
    - [ICAO Aircraft Engine Emissions Databank](https://www.easa.europa.eu/en/domains/environment/icao-aircraft-engine-emissions-databank)

    Parameters
    ----------
    path_excel_engine_data_icao_in : str
        Path or URL to the ICAO Aircraft Engine Emissions Databank Excel file.
        Equivalent to parameter `io` in [`pandas.read_excel`](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html#pandas-read-excel).
    scaling_polynomial : np.polynomial.Polynomial
        [`numpy.polynomial.polynomial.Polynomial`](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.html#numpy.polynomial.polynomial.Polynomial) Polynomial to scale TSFC (takeoff) to TSFC (cruise).

    Returns
    -------
    pd.DataFrame
        DataFrame with engine data and scaled TSFC (cruise) values.

    Raises
    ------
    ValueError
        If the provided scaling_polynomial is not a [`numpy.polynomial.polynomial.Polynomial`](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.html#numpy.polynomial.polynomial.Polynomial) object.
    """
    if not isinstance(scaling_polynomial, np.polynomial.Polynomial):
        raise ValueError("scaling_polynomial must be a numpy Polynomial object.")

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

    models4['Air Mass Flow [kg/s]'] = (air_density* flight_vel * math.pi * models4['Fan diameter,float,metre']**2)/4


def calculate_air_mass_flow_rate(
    df: pd.DataFrame,
    altitude: float = 12000.0 * ureg.meter,
) -> pd.DataFrame:
    r"""
    Calculates the air mass flow rate for each engine in the provided DataFrame.
    
    Air mass flow rate $m$ is defined as:
    $$
    m = r^2 \pi C_a \rho(h)
    $$
    as a dimensionality analysis shows:
    $$
    [kg/s] = [m^2][m/s][kg/m^3] = [m^3/s][kg/m^3]
    $$
    where
    
    | Symbol       | Units         | Description                                |
    |--------------|---------------|--------------------------------------------|
    | $m$          | kg/s          | Air mass flow rate                         |
    | $\rho$       | kg/m³         | Air density                                |
    | $C_a$        | m/s           | Cruise speed = intake air velocity         |
    | $r$          | m             | Fan radius                                 |
    | $h$          | m             | Altitude                                   |

    Warnings
    --------
    Air density $\rho(h)$ is dependent on altitude $h$. 
    The default altitude is 12'000 meters, but can be changed via the `altitude` parameter.

    See Also
    --------
    - [Mass Flow Rate on Wikipedia](https://en.wikipedia.org/wiki/Mass_flow_rate#Air_mass_flow_rate)
    - [`aircraftdetective.utility.physics._calculate_atmospheric_conditions`][]

    Parameters
    ----------
    df : pd.DataFrame
        [`pint-pandas`](https://pint-pandas.readthedocs.io/en/latest/) DataFrame containing the columns:
        
        | Column Name   | Dimension     |
        |---------------|---------------|
        | `Cruise Speed`| [length/time] |
        | `Fan Diameter`| [length]      |

    altitude : float [distance], optional
        Altitude above sea level, by default 12000.0 * ureg.meter   

    Raises
    ------
    ValueError
        If `df` is empty or does not contain the required columns.

    Returns
    -------
    pd.DataFrame
        [`pint-pandas`](https://pint-pandas.readthedocs.io/en/latest/) DataFrame with a new column `Air Mass Flow` [weight/time] added.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
    required_columns = [
        'Cruise Speed',
        'Fan Diameter',
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"DataFrame is missing required columns: {missing_columns}")
    
    df_func = df.copy()

    air_density = _calculate_atmospheric_conditions(altitude)['density']
    df_func['Air Mass Flow'] = (air_density * df['Cruise Speed'] * math.pi * (df['Fan Diameter']/2)**2)
    return df_func


def calculate_engine_efficiencies(
    df: pd.DataFrame
) -> pd.DataFrame:
    r"""
    Calculates the overall engine efficiency, propulsive efficiency and thermal efficiency 
    for each engine in the provided DataFrame. The calculation combines multiple equations describing turbofan engine performance.
    
    Overall engine efficiency $\eta_0$ is defined as:
    $$
    \eta_0 = \frac{C_a}{TSFC} \frac{1}{Q}
    $$
    Propulsive efficiency $\eta_P$ is defined as:
    $$
    \eta_p = \frac{2}{1 + \frac{C_j}{C_a}}
    $$
    Since the exhaust jet velocity $C_j$ is not typically known,
    the thrust equation
    $$
    F = m (C_j - C_a)
    $$
    is rearranged to solve for $C_j$:
    $$
    C_j = \frac{F}{m} + C_a
    $$
    which gives
    $$
    \eta_p = \frac{2}{2 + \frac{F}{m C_a}}
    $$
    Using a different definition for overall efficiency $\eta_0$:
    $$
    \eta_0 = \frac{FC_a}{m_fQ}
    $$
    net thrust $F$ can be obtained:
    $$
    F = \frac{\eta_0 m_f Q}{C_a}
    $$
    which for propulsive efficiency $\eta_P$ finally gives: 
    $$
    \eta_p = \frac{2}{2 + \frac{\eta_0 m_f Q}{m C_a^2}}
    $$

    | Symbol       | Units         | Description                                |
    |--------------|---------------|--------------------------------------------|
    | $\eta_0$     | dimensionless | Overall engine efficiency                  |
    | $C_a$        | m/s           | Cruise speed = intake air velocity         |
    | $C_j$        | m/s           | Jet velocity = exhaust air velocity        |
    | $TSFC$       | kg/(N*s)      | Thrust-Specific Fuel Consumption at cruise |
    | $m$          | kg/s          | Air mass flow rate                         |
    | $m_f$        | kg/s          | Fuel mass flow rate                        |
    | $F$          | N             | Net engine thrust                          |
    | $Q$          | J/kg          | Fuel Heating Value                         |

    Warnings
    --------
    In $F = \frac{\eta_0 m_f Q}{C_a}$, the fuel mass flow rate $m_f$
    is assumed to be for _all_ engines of the aircraft. 
    This means that air mass flow rate $m$ is also calculated for _all_ engines of the aircraft.

    References
    ----------
    - Overall engine efficiency $\eta_0$: [Eqn. (3.5)-(3.7) in Saravanamuttoo et al. (2017)](http://books.google.com/books?vid=ISBN9781292093093)
    - Propulsive efficiency $\eta_P$: [Eqn. (3.3) in Saravanamuttoo et al. (2017)](http://books.google.com/books?vid=ISBN9781292093093)
    - Net engine thrust $F$: [Eqn. (3.1) in Saravanamuttoo et al. (2017)](http://books.google.com/books?vid=ISBN9781292093093)
    
    See Also
    --------
    - [Jet Fuel on Wikipedia](https://en.wikipedia.org/wiki/Aviation_fuel)
    
    Parameters
    ----------
    df : pd.DataFrame
        [`pint-pandas`](https://pint-pandas.readthedocs.io/en/latest/) DataFrame containing the columns:

        | Column Name         | Dimension           |
        |---------------------|---------------------|
        | `Cruise Speed`      | [length/time]       |
        | `TSFC (cruise)`     | [mass/(force*time)] |
        | `Fuel Flow`         | [mass/time]         |
        | `Air Mass Flow`     | [mass/time]         |
        | `Number of Engines` | [dimensionless]     |
        | `B/P Ratio`         | [dimensionless]     |

    Returns
    -------
    pd.DataFrame
        [`pint-pandas`](https://pint-pandas.readthedocs.io/en/latest/) DataFrame with a new column `Engine Efficiency` [dimensionless] added.
    """
    required_columns = [
        'Cruise Speed',
        'TSFC (cruise)',
        'Fuel Flow',
        'Air Mass Flow',
        'Number of Engines',
        'B/P Ratio'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")
    
    jeta1_energydensity = 34.7*1E6 * ureg.J / ureg.l # https://en.wikipedia.org/wiki/Jet_fuel
    jeta1_density = 0.804 * ureg.kg / ureg.l # https://en.wikipedia.org/wiki/Jet_fuel
    jeta1_specificenergy = jeta1_energydensity / jeta1_density

    df['Engine Efficiency'] = df['Cruise Speed'] / (jeta1_specificenergy * df['TSFC (cruise)'])

    df['Propulsive Efficiency'] = 2 / (2 + (df['Engine Efficiency'] * df['Fuel Flow'] * jeta1_density * jeta1_specificenergy) / (df['Air Mass Flow'] * df['Cruise Speed']**2 * df['Number of Engines']))
    
    df['Thermal Efficiency'] = df['Engine Efficiency']/df['Propulsive Efficiency']
    
    df.loc[df['B/P Ratio']<=2, 'Thermal Efficiency'] = np.nan
    df.loc[df['B/P Ratio']<=2, 'Propulsive Efficiency'] = np.nan
    
    return df

