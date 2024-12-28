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

    row_baseline_aircraft_wide = df[df['Aircraft Designation'] == baseline_aircraft_designation_wide]
    row_baseline_aircraft_narrow = df[df['Aircraft Designation'] == baseline_aircraft_designation_narrow]
    
    eu_baseline = row_baseline_aircraft_wide['Fuel/Available Seat Distance'].values[0]
    tsfc_baseline = row_baseline_aircraft_wide['TSFC (cruise)'].values[0]
    ld_baseline = row_baseline_aircraft_wide['L/D'].values[0]
    mtow_baseline_wide = row_baseline_aircraft_wide['MTOW'].values[0]
    mtow_baseline_narrow = row_baseline_aircraft_narrow['MTOW'].values[0]

    df['Energy Use (relative)'] = 100 * (eu_baseline - df['Fuel/Available Seat Distance']) / eu_baseline
    df['Engine Efficiency (relative)'] = 100 * (tsfc_baseline - df['TSFC (cruise)']) / tsfc_baseline
    df['Aerodynamic Efficiency (relative)'] = 100 * (ld_baseline - df['L/D']) / ld_baseline
    df['Structural Efficiency'] = df['MTOW'] / df['Pax Exit Limit']
    df['Structural Efficiency (relative)'] = 100 * (mtow_baseline_wide - df[df['Type'] == 'Wide']['Structural Efficiency']) / mtow_baseline_wide
    df['Structural Efficiency (relative)'] = 100 * (mtow_baseline_narrow - df[df['Type'] == 'Narrow']['Structural Efficiency']) / mtow_baseline_narrow

    """
    polynomial_energy_use = np.polynomial.Polynomial.fit(
        x=df['YOI'],
        y=df['Energy Use (relative)'],
        deg=4,
    )
    polynomial_engine_efficiency = np.polynomial.Polynomial.fit(
        x=df['YOI'],
        y=df['Engine Efficiency (relative)'],
        deg=4,
    )
    polynomial_aerodynamic_efficiency = np.polynomial.Polynomial.fit(
        x=df['YOI'],
        y=df['Aerodynamic Efficiency (relative)'],
        deg=4,
    )
    polynomial_structural_efficiency = np.polynomial.Polynomial.fit(
        x=df['YOI'],
        y=df['Structural Efficiency (relative)'],
        deg=4,
    )
    """
    return df #, polynomial_energy_use, polynomial_engine_efficiency, polynomial_aerodynamic_efficiency, polynomial_structural_efficiency


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