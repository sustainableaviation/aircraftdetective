# %%
import math
import pandas as pd
import numpy as np
from aircraftdetective.utility.statistics import r_squared
from typing import Any

def _compute_polynomials_from_dataframe(
    df: pd.DataFrame,
    col_name_x: str,
    degree: int
) -> dict[str, Any]:
    r"""
    Computes polynomial fits of a given degree for each column in a dataframe.

    Given a dataframe with at least two columns, computes a polynomial fit of the specified degree
    for each column against the specified x-axis column. Returns a dictionary containing the polynomial
    fits and their corresponding R-squared values.

    See Also
    --------
    [`numpy.polynomial.Polynomial.fit`](https://numpy.org/doc/2.0/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html)  
    [`aircraftdetective.utility.statistics.r_squared`][]

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data for polynomial fitting.
    col_name_x : str
        Name of the column to be used as the x-axis for polynomial fitting.
    degree : int
        Degree of the polynomial to fit.
    
    Returns
    -------
    dict[str, Any]
        A dictionary where keys are column names and values are the corresponding polynomial fits.  
        Of the kind:  
        ```
        {
            'Column1': Polynomial object,
            'Column1_r2': float,
            'Column2': Polynomial object,
            'Column2_r2': float,
            ...
        }
        ```
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a Pandas DataFrame")
    if df.empty:
        raise ValueError("df cannot be empty")
    if col_name_x not in df.columns:
        raise ValueError(f"col_name_x '{col_name_x}' not found in df columns")
    if not isinstance(degree, int) or degree < 0:
        raise ValueError("degree must be a non-negative integer")
    if len(df) <= degree:
        raise ValueError("number of data points must be greater than degree")
    
    dict_polynomials = {}
    for col in df.columns:
        x = df[col_name_x].astype("float64")
        y = df[col].astype("float64")
        polynomial_fit = np.polynomial.Polynomial.fit(
            x=x,
            y=y,
            deg=degree,
        )
        r_squared_polynomial = r_squared(y, polynomial_fit(x))
        dict_polynomials[col] = polynomial_fit
        dict_polynomials[f'{col}_r2'] = r_squared_polynomial

    return dict_polynomials
      

def _compute_efficiency_improvement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    Computes the relative improvement $\Delta\%x$ between times $t=0$ and $t=T$
    $$
    \Delta\%x=\frac{x(t=T)-x(t=0)}{x(t=0)} \times 100 [\%]
    $$
    of relevant aircraft efficiency metrics.

    $\eta$...

    Given a dataframe of the kind:

    | YOI | EU | TSFC | OEW/Exit Limit | L/D |
    |-----|----|------|----------------|-----|
    | 1958| ...| ...  | ...            | ... |

    computes relative improvements in energy use, thrust-specific fuel consumption,
    weight per available seat and lift-to-drag ratio compared to the 1958 baseline.

    
    Notes
    -----
    Aircraft energy use $E_U$ as per Eqn. (4) in [Babikian et al. (2002)](https://doi.org/10.1016/S0969-6997(02)00020-0)
    is proportional to the product of thrust-specific fuel consumption,
    the weight weight per available seat and the lift-to-drag ratio:
    $$
    E_U [\text{J/ASK}] \sim TSFC \times \frac{W}{pax} \times \bigg(\frac{L}{D}\bigg)^{-1} 
    $$
    
    abc
    $$
    \Delta C = \Delta C_1 + \Delta C_2 + ... + \Delta C_n
    $$
 
    This means that for the percentage change in aircraft energy use $\Delta E_U (t=T)$, we get:
    $$
    \Delta \% E_U(t=T) = \frac{E_U(t=T) - E_U(t=1958)}{E_U(t=1958)} \times 100 [\%]
    $$

    Warning
    -------
    Since $l\D$ is in the denominator of the equation for $E_U$, its improvement
    is inverted before calculating the relative improvement:
    $$
    L/D_{improved} = 100 / (L/D_{1958} / L/D_{current})
    $$

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the aircraft sub-efficiency data.
    
    Returns
    -------
    
    """
    
    

def _compute_lmdi_factor_contributions(
    aggregate_t1: float,
    aggregate_t2: float,
    factor_t1: float,
    factor_t2: float
) -> float:
    r"""
    Computes the contributions of changes in factors to the change in an aggregate
    according to the additive logarithmic mean Divisia index method I (LMDI-I).

    Given values for an aggregate $C$ and a factor $C_i$
    $$
    C = C_1 \times C_2 \times ... \times C_n
    $$
    at two points in time $(t_1, t_2)$,
    computes the contribution $\Delta C_i$ of the change in factor $C_i$
    to the change in total $\Delta C$ through the
    additive logarithmic mean Divisia index method I (LMDI-I).
    $$
    \Delta C_i = \left(\frac{C(t_2) - C(t_1)}{\ln C(t_2) - \ln C(t_1)}\right) \times \ln\left(\frac{C_i(t_2)}{C_i(t_1)}\right)
    $$
    where

    | Symbol | Unit     | Description        |
    |--------|----------|--------------------|
    | $C$    | -        | Efficiency         |
    | $C_i$  | -        | Sub-efficiency $i$ |
    
    Notes
    -----
    The LDMI-I method is _additive_, which means that the contributions sum up to the total change:
    $$
    \Delta C = \sum_i \Delta C_i
    $$

    References
    ----------
    Ang & Goh (2019), "Index decomposition analysis for comparing emission scenarios: Applications and challenges", 
    _Energy Economics_, doi:[10.1016/j.eneco.2019.06.013](https://doi.org/10.1016/j.eneco.2019.06.013)  
    Eqn. (28) and Table 5 in Ang & Zhang (2000), "A survey of index decomposition analysis in energy and environmental studies",
    _Energy_, doi:[10.1016/S0360-5442(00)00039-6](https://doi.org/10.1016/S0360-5442(00)00039-6)

    Parameters
    ----------
    aggregate_t1 : float
        Value of the aggregate $C$ at time $t_1$
    aggregate_t2 : float
        Value of the aggregate $C$ at time $t_2$
    factor_t1 : float
        Value of the factor $C_i$ at time $t_1$
    factor_t2 : float
        Values of the factor $C_i$ at time $t_2$

    Returns
    -------
    float
        Contribution of change in sub-efficiency to change in total efficiency

    Example
    -------
    ```pyodide install='aircraftdetective'
    aggregate_t1 = 8.0
    aggregate_t2 = 64.0
    factor_1_t1 = 2.0
    factor_1_t2 = 4.0
    factor_2_t1 = 4.0
    factor_2_t2 = 16.0
    delta_factor_1 = _compute_lmdi_factor_contributions(aggregate_t1, aggregate_t2, factor_1_t1, factor_1_t2)
    delta_factor_2 = _compute_lmdi_factor_contributions(aggregate_t1, aggregate_t2, factor_2_t1, factor_2_t2)
    ```
    """
    delta_aggregate = aggregate_t2 - aggregate_t1
    if delta_aggregate == 0:
        return 0.0
    log_mean_aggregate = delta_aggregate / (math.log(aggregate_t2) - math.log(aggregate_t1))
    delta_factor = log_mean_aggregate * math.log(factor_t2 / factor_t1)
    return delta_factor