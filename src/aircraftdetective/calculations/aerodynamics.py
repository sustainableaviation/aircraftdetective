# %%
import pandas as pd
import numpy as np
import pint_pandas
import pint
ureg = pint.get_application_registry()
from aircraftdetective.utility.tabular import _validate_dataframe_columns_with_units
from aircraftdetective.data.constants import g_acceleration

def compute_lift_to_drag_ratio(
    df: pd.DataFrame,
    beta: dict[str, float],
) -> pd.DataFrame:
    r"""
    Given points from a payload/range diagram of an aircraft,
    calculates the lift-to-drag ratio (=aerodynamic efficiency)
    based on the Breguet range equation.

    ![Payload/Range Diagram](../../_static/payload_range_generic.svg)
    **Figure 1:** Payload/Range diagram of the Airbus A350-900 (data derived [from manufacturer information](https://web.archive.org/web/20211129144142/https://www.airbus.com/sites/g/files/jlcbta136/files/2021-11/Airbus-Commercial-Aircraft-AC-A350-900-1000.pdf)).
    Note that in this figure, the y-axis shows _total_ aircraft weight, not _payload weight_.
    Total aircraft weight can be computed by adding the operating empty weight (OEW) to the payload weight and fuel weight.

    The range equation for a jet aircraft can be written as
    $$
    R = \frac{V}{c \cdot g} \frac{L}{D} \ln \bigg( \frac{m_1}{m_2} \bigg)
    $$
    where

    | Symbol | Unit        | Description                                |
    |--------|-------------|--------------------------------------------|
    | $R$    | m           | Aircraft range                             |
    | $V$    | m/s         | Cruise speed                               |
    | $c$    | g/kNs       | Thrust-specific fuel consumption (average) |
    | $g$    | m/s$^2$     | Acceleration due to gravity                |
    | $L/D$  | -           | Lift-to-drag ratio                         |
    | $m_1$  | kg          | weight at start of cruise segment          |
    | $m_2$  | kg          | weight at end of cruise segment            |

    The right hand side is composed of the logarithmic weight ratio term and a 
    term also known as the range factor $K$:
    $$
    K = \frac{V}{c \cdot g} \frac{L}{D}
    $$
    which is
    > (...) assumed to be constant during the cruise phase, 
    > represents an average performance of the aircraft 
    > and is related to the mean aerodynamic and propulsion characteristics (...)  
    > — [Martinez-Val et al. (2008)](https://doi.org/10.1243/09544100JAERO338)

    The masses $m_1$ and $m_2$ can be expressed in terms of the maximum takeoff weight (MTOW) 
    and the maximum zero fuel weight (MZFW) as
    $$
    \frac{m_1}{m_2} \propto \frac{(1-\beta)MTOW}{OEW + \text{Payload}} = \frac{(1-\beta)MTOW}{MZFW}
    $$
    where the correction factor $\beta$ 

    > (...) accounts for the fuel fractions burnt during the take-off and climbing phases (...), 
    > the fuel fractions burnt during the descent and landing phases (...) 
    > [and] the reserve fuel.  
    > — [Martinez-Val et al. (2008)](https://doi.org/10.1243/09544100JAERO338)

    With that, the lift-to-drag ratio can be computed as
    $$
    \frac{L}{D} = \frac{K \cdot g \cdot TSFC_{cruise}}{V} = \frac{R \cdot g \cdot TSFC_{cruise}}{V \ln(\frac{MTOW}{MZFW} (1-\beta))}
    $$
    where

    | Symbol             | Unit            | Description                                      |
    |--------------------|-----------------|--------------------------------------------------|
    | $MTOW$             | [kg]            | Maximum takeoff weight                           |
    | $MZFW$             | [kg]            | Maximum zero fuel weight                         |
    | $\beta$            | -               | Correction factor for the Breguet range equation |

    References
    ----------
    - [Young (2018), Eqn. (13.36)](https://doi.org/10.1002/9781118534786)
    - [Martinez-Val et al. (2008)](https://doi.org/10.1243/09544100JAERO338)
    - [Martinez-Val et al. (2005)](https://doi.org/10.2514/6.2005-121)

    See Also
    --------
    [Range (aeronautics) on Wikipedia](https://en.wikipedia.org/wiki/Range_(aeronautics))

    Parameters
    ----------
    df : pd.DataFrame
        [`pint-pandas`](https://pint-pandas.readthedocs.io/en/latest/) DataFrame containing the columns:

        | Column Name                       | Dimension           |
        |-----------------------------------|---------------------|
        | `Payload/Range: Range at Point B` | [length]            |
        | `Payload/Range: Range at Point C` | [length]            |
        | `Payload/Range: MZFW at Point B`  | [mass]              |
        | `Payload/Range: MZFW at Point C`  | [mass]              |
        | `Cruise Speed`                    | [length/time]       |
        | `TSFC (cruise)`                   | [mass/(force*time)] |   

    beta : dict[str, float]
        Correction factor for the Breguet range equation, typically between 0.4 and 0.6.  
        Specific to aircraft type, e.g.:

        ```python
        beta = {
            'Narrow': 0.06,
            'Wide': 0.04,
        }
        ```

    Returns
    -------
    pd.DataFrame
        [`pint-pandas`](https://pint-pandas.readthedocs.io/en/latest/) DataFrame with an additional column `L/D` [dimensionless] added.

    Raises
    ------
    ValueError
        If the input DataFrame does not contain the required columns, or if `beta` is not between 0 and 1.

    Example
    -------
    ```pyodide install='aircraftdetective'
    ```
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
    
    _validate_dataframe_columns_with_units(
        df,
        {
            'Payload/Range: Range at Point B': '[length]',
            'Payload/Range: Range at Point C': '[length]',
            'Payload/Range: MZFW at Point B': '[mass]',
            'Payload/Range: MZFW at Point C': '[mass]',
            'MTOW': '[mass]',
            'Cruise Speed': '[length]/[time]',
            'TSFC (cruise)': '[time]/[length]',
        }
    )
    if 'Type' not in df.columns:
            raise ValueError("The DataFrame must contain a 'Type' column when `beta` is a dict.")

    df_func = df.copy()

    beta_series = df_func['Type'].map(beta)
    if beta_series.isna().any():
        missing_types = sorted(df_func.loc[beta_series.isna(), 'Type'].astype(str).unique())
        raise ValueError(f"Missing beta for Type(s): {missing_types}. Provide all required mappings.")

    if not ((beta_series > 0) & (beta_series < 1)).all():
        bad_rows = beta_series.index[~((beta_series > 0) & (beta_series < 1))]
        raise ValueError(f"`beta` values must be between 0 and 1 for all rows. Bad indices: {list(bad_rows)}")

    K_B: pd.Series = df_func['Payload/Range: Range at Point B'] / np.log(
        (df_func['MTOW'] / df_func['Payload/Range: MZFW at Point B']) * (1 - beta_series)
    )
    K_C: pd.Series = df_func['Payload/Range: Range at Point C'] / np.log(
        (df_func['MTOW'] / df_func['Payload/Range: MZFW at Point C']) * (1 - beta_series)
    )
    K_average = (K_B + K_C) / 2

    df_func['L/D'] = K_average * g_acceleration * df_func['TSFC (cruise)'] / df_func['Cruise Speed']
    df_func['L/D'] = df_func['L/D'].pint.to_base_units()

    if not df_func['L/D'].pint.units.dimensionless:
        raise ValueError("Calculated L/D is not dimensionless, please check the input units.")

    return df_func


def compute_aspect_ratio(
    df: pd.DataFrame,
) -> pd.DataFrame:
    r"""
    Given the wingspan $b$ and the wing area $S$, returns the aspect ratio $A$ of an aircraft.
    $$
    A = \frac{b^2}{S}
    $$
    where

    | Symbol | Dimension | Description  |
    |--------|-----------|--------------|
    | $A$    | -         | Aspect ratio |
    | $b$    | [length]  | Wingspan     |
    | $S$    | [area]    | Wing area    |

    
    References
    ----------
    [Eqn. (3.3) in Young (2018)](https://doi.org/10.1002/9781118534786)
    
    See Also
    --------
    [Aspect Ratio on Wikipedia](https://en.wikipedia.org/wiki/Aspect_ratio_(aeronautics))

    
    Parameters
    ----------
    df : pd.DataFrame
        [`pint-pandas`](https://pint-pandas.readthedocs.io/en/latest/) DataFrame containing the columns:

        | Column Name | Dimension  |
        |-------------|------------|
        | `Wingspan`  | [length]   |
        | `Wing Area` | [area]     |

    Returns
    -------
    pd.DataFrame
        [`pint-pandas`](https://pint-pandas.readthedocs.io/en/latest/) DataFrame with an additional column `Aspect Ratio` [dimensionless] added.

    Raises
    ------
    ValueError
        If the input DataFrame does not contain the required columns.

    Example
    -------
    ```pyodide install='aircraftdetective'
    import pandas as pd
    from aircraftdetective.calculations.aerodynamics import compute_aspect_ratio
    df = pd.DataFrame(
        {
            'Wingspan': pd.Series([60.3, 64.8], dtype='pint[m]'),
            'Wing Area': pd.Series([361, 443], dtype='pint[m**2]'),
        }
    )
    compute_aspect_ratio(df=df)
    ```
    """
    df_func = df.copy()
    required_columns = ['Wingspan', 'Wing Area']
    missing_columns = [col for col in required_columns if col not in df_func.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")
    df_func['Aspect Ratio'] = (df_func['Wingspan']**2) / df_func['Wing Area']
    df_func['Aspect Ratio'] = df_func['Aspect Ratio'].pint.to_base_units()
    return df_func