# %%

import pandas as pd
import pint
import pint_pandas
ureg = pint.get_application_registry() # https://pint-pandas.readthedocs.io/en/latest/user/common.html#using-a-shared-unit-registry
import numpy as np

def normalize_aircraft_efficiency_metrics(
    df: pd.DataFrame,
    baseline_aircraft_designation_wide: str,
    baseline_aircraft_designation_narrow: str,
) -> tuple[np.polynomial.Polynomial, np.polynomial.Polynomial, np.polynomial.Polynomial, np.polynomial.Polynomial]:
    """
    Converts aircraft efficiency metrics into "% improvement over the earliest aircraft in the dataset"
    """

    row_baseline_aircraft_wide: pd.DataFrame = df[df['Aircraft Designation'] == baseline_aircraft_designation_wide]
    row_baseline_aircraft_narrow: pd.DataFrame = df[df['Aircraft Designation'] == baseline_aircraft_designation_narrow]
    
    eu_baseline = row_baseline_aircraft_wide['Energy Use'].values[0]
    tsfc_baseline = row_baseline_aircraft_wide['TSFC (cruise)'].values[0]
    ld_baseline = row_baseline_aircraft_wide['L/D'].values[0]
    struct_baseline_wide = row_baseline_aircraft_wide['Structural Efficiency'].values[0]
    struct_baseline_narrow = row_baseline_aircraft_narrow['Structural Efficiency'].values[0]

    df['Energy Use (relative)'] = 100 * (eu_baseline - df['Energy Use']) / eu_baseline
    df['Engine Efficiency (relative)'] = 100 * (tsfc_baseline - df['TSFC (cruise)']) / tsfc_baseline
    df['Aerodynamic Efficiency (relative)'] = 100 * (df['L/D'] - ld_baseline) / ld_baseline

    df['struct_eff_rel_wide'] = 100 * (struct_baseline_wide - df[df['Type'] == 'Wide']['Structural Efficiency']) / struct_baseline_wide
    df['struct_eff_rel_narrow'] = 100 * (struct_baseline_narrow - df[df['Type'] == 'Narrow']['Structural Efficiency']) / struct_baseline_narrow
    df['Structural Efficiency (relative)'] = df['struct_eff_rel_wide'].combine_first(df['struct_eff_rel_narrow'])
    df = df.drop(columns=['struct_eff_rel_wide', 'struct_eff_rel_narrow'])

    return df


def determine_polynomial_fit(
    df: pd.DataFrame,
    column: str,
    degree: int,
) -> np.polynomial.Polynomial:
    """
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


def compute_log_mean_divisia_index(
    eff_total_t0: float,
    eff_total_t1: float,
    eff_sub_t0: float,
    eff_sub_t1: float,
) -> float:
    """
    Given total and sub-efficiency values at two points in time, compute the log mean Divisia index.

    ΔC₁ + ΔC₂ + ... + ΔCₙ = ΔC

    See Also
    --------
    - [Ang & Goh (2019), eqn. (2) ff.](https://doi.org/10.1016/j.eneco.2019.06.013)

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
    eff_total_log_mean = (eff_total_t1 - eff_total_t0) / (np.log(eff_sub_t1) - np.log(eff_sub_t0))
    return eff_total_log_mean * np.log(eff_sub_t1 / eff_sub_t0)