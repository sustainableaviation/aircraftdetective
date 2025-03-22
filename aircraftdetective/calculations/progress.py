# %%

import pandas as pd
import pint
import pint_pandas
ureg = pint.get_application_registry() # https://pint-pandas.readthedocs.io/en/latest/user/common.html#using-a-shared-unit-registry
import numpy as np

def compute_normalized_aircraft_efficiency_metrics(
    df: pd.DataFrame,
    baseline_aircraft_designation: str,
) -> pd.DataFrame:
    r"""
    Converts aircraft efficiency metrics into "% improvement over the earliest aircraft in the dataset"
    XXXXXXXXXXX
    XXXXXXXXXXX

    $\frac{W}{G} = \frac{p}{100}$

    usually solved for $p$:

    $p = \frac{W}{G} \times 100$

    where

    - $W$ percent value
    - $G$ baseline value
    - $p$ percentage improvement

    For instance, overall efficiency η is determined by the metric _Energy Use_ [MJ/ASK].

    Arbitrarily setting the aircraft released in 1960 in the dataset as the baseline,
    the percentage improvement in Overall Efficiency of an aircraft released in 2020 is calculated as:

    p = 100 * (1/EU(t=2020))/(1/EU(t=1960)) - 100 = 100 * (EU(t=1960) / EU(t=2020)) - 100

    Notes
    -----

    Since an aircraft is MORE efficient if it has a LOWER Energy Use value, overall efficiency is proportional to the inverse of Energy Use:

    $\eta = \frac{1}{EU}$

    Since an aircraft is MORE efficient if it has a HIGHER L/D value, aerodynamic efficiency is proportional to L/D:

    $\eta = L/D$

    See Also
    --------
    - [Wikipedia: Percentage](https://en.wikipedia.org/wiki/Percentage#Variants_of_the_percentage_calculation)

    """

    row_baseline_aircraft: pd.DataFrame = df[df['Aircraft Designation'] == baseline_aircraft_designation]
    
    eu_baseline = row_baseline_aircraft['Energy Use'].values[0]
    tsfc_baseline = row_baseline_aircraft['TSFC (cruise)'].values[0]
    ld_baseline = row_baseline_aircraft['L/D'].values[0]
    struct_baseline = row_baseline_aircraft['OEW/Pax Exit Limit'].values[0]

    df['Overall Efficiency Improvement (%)'] = 100 * (eu_baseline/df['Energy Use']) - 100
    df['Engine Efficiency Improvement (%)'] = 100 * (tsfc_baseline/df['TSFC (cruise)']) - 100
    df['Structural Efficiency Improvement (%)'] = 100 * (struct_baseline/df['OEW/Pax Exit Limit']) - 100
    df['Aerodynamic Efficiency Improvement (%)'] = 100 * (df['L/D']/ld_baseline) - 100 # see the Notes section of docstrings
    
    return df


def determine_polynomial_fit(
    df: pd.DataFrame,
    column: str,
    degree: int,
) -> np.polynomial.Polynomial:
    r"""
    Given a DataFrame with Pint unit columns, a column name, and a polynomial degree, return a polynomial fit of the data in the column.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Pint unit columns
    column : str
        Name of the column to fit
    degree : int
        Degree of the polynomial fit

    See Also
    --------
    - [`np.polynomial.Polynomial.fit`](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html#numpy-polynomial-polynomial-polynomial-fit)

    Returns
    -------
    np.polynomial.Polynomial
        Polynomial fit of the data in the column
    """
    
    df = df.dropna(subset=[column])
    df = df.sort_values(by='YOI', ascending=True)

    polynomial = np.polynomial.Polynomial.fit(
        x=df['YOI'],
        y=df[column].values.numpy_data, # polynomial.fit() requires a numpy array, not a typed (=Pint) Pandas series
        deg=degree,
    )

    return polynomial


def create_efficiency_dataframe_from_polynomials(
    polynomial_overall: np.polynomial.Polynomial,
    polynomial_engine: np.polynomial.Polynomial,
    polynomial_structural: np.polynomial.Polynomial,
    polynomial_aerodynamic: np.polynomial.Polynomial,
    yoi_range: tuple[int, int],
) -> pd.DataFrame:
    r"""
    Given four polynomials and a range of years of introduction, create a DataFrame with efficiency metrics for each year in the range.

    Parameters
    ----------
    polynomial_overall : np.polynomial.Polynomial
        Polynomial fit of the overall efficiency data
    polynomial_engine : np.polynomial.Polynomial
        Polynomial fit of the engine efficiency data
    polynomial_structural : np.polynomial.Polynomial
        Polynomial fit of the structural efficiency data
    polynomial_aerodynamic : np.polynomial.Polynomial
        Polynomial fit of the aerodynamic efficiency data
    yoi_range : tuple[int, int]
        Range of years of introduction

    Returns
    -------
    pd.DataFrame
        DataFrame with efficiency metrics for each year in the range
    """
    
    df = pd.DataFrame()
    df['YOI'] = np.arange(yoi_range[0], yoi_range[1] + 1)

    df['Overall Efficiency Improvement (%)'] = polynomial_overall(df['YOI'])
    df['Engine Efficiency Improvement (%)'] = polynomial_engine(df['YOI'])
    df['Structural Efficiency Improvement (%)'] = polynomial_structural(df['YOI'])
    df['Aerodynamic Efficiency Improvement (%)'] = polynomial_aerodynamic(df['YOI'])

    return df


def compute_log_mean_divisia_index_of_efficiency(
    df: pd.DataFrame,
) -> float:
    r"""
    Given total and sub-efficiency values at two points in time, compute the log mean Divisia index.

    $$
    \Delta C = \Delta C_1 + \Delta C_2 + ... + \Delta C_n$
    $$

    where

    | Symbol | Unit     | Description      |
    |--------|----------|------------------|
    | $C$    | -        | Efficiency       |
    | $C_i$  | -        | Sub-efficiency i |


    See Also
    --------
    Ang, Beng Wah, and Tian Goh. "Index decomposition analysis for comparing emission scenarios: Applications and challenges."
    _Energy Economics_ 83 (2019): 74-87. doi:[10.1016/j.eneco.2019.06.013](https://doi.org/10.1016/j.eneco.2019.06.013)

    Parameters
    ----------
    eff_total_t0 : float
        Total efficiency at time t=0
    eff_total_t1 : float
        Total efficiency at time t=1
    eff_sub_t0 : float
        Sub-efficiency at time t=0
    eff_sub_t1 : float
        Sub-efficiency at time t=1

    Returns
    -------
    float
        Contribution of change in sub-efficiency to change in total efficiency
    """

    for col in df.columns:
        df[f'{col} + 100'] = df[col] + 100

    df['delta_overall_efficiency'] = df['Overall Efficiency Improvement (%) + 100'] - df['Overall Efficiency Improvement (%) + 100'].iloc[0]
    df['log_average_overall_efficiency'] = (df['Overall Efficiency Improvement (%) + 100'] - df['Overall Efficiency Improvement (%) + 100'].iloc[0]) / (np.log(df['Overall Efficiency Improvement (%) + 100']) - np.log(df['Overall Efficiency Improvement (%) + 100'].iloc[0]))
    
    df['lmdi_contrib_engine_efficiency'] =  df['log_average_overall_efficiency'] * np.log(df['Engine Efficiency Improvement (%) + 100'] / df['Engine Efficiency Improvement (%) + 100'].iloc[0])
    df['lmdi_contrib_structural_efficiency'] =  df['log_average_overall_efficiency'] * np.log(df['Structural Efficiency Improvement (%) + 100'] / df['Structural Efficiency Improvement (%) + 100'].iloc[0])
    df['lmdi_contrib_aerodynamic_efficiency'] =  df['log_average_overall_efficiency'] * np.log(df['Aerodynamic Efficiency Improvement (%) + 100'] / df['Aerodynamic Efficiency Improvement (%) + 100'].iloc[0])
    df['lmdi_contrib_redisual'] = df['delta_overall_efficiency'] - df['lmdi_contrib_engine_efficiency'] - df['lmdi_contrib_structural_efficiency'] - df['lmdi_contrib_aerodynamic_efficiency']

    return df