# %%
import math
import pandas as pd

def compute_subefficiency_contributions(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    Aircraft energy use $E_U$ as per Eqn. (4) in [Babikian et al. (2002)](https://doi.org/10.1016/S0969-6997(02)00020-0)
    is proportional to the product of thrust-specific fuel consumption,
    the weight weight per available seat and the lift-to-drag ratio:
    $$
    E_U [\text{J/ASK}] \sim TSFC \times \frac{W}{pax} \times \frac{L}{D} 
    $$
    Relative improvements:
    $$
    \Delta\%x=\frac{x(t=T)-x(t=0)}{x(t=0)} \times 100 [\%]
    $$
    abc
    $$
    \Delta C = \Delta C_1 + \Delta C_2 + ... + \Delta C_n
    $$
 
    This means that for the percentage change in aircraft energy use $\Delta E_U (t=T)$, we get:
    $$
    \Delta \% E_U(t=T) = \frac{E_U(t=T) - E_U(t=1958)}{E_U(t=1958)} \times 100 [\%]
    $$
    for col in df.columns:
    df[f'{col} + 100'] = df[col] + 100

    df['delta_overall_efficiency'] = df['Overall Efficiency Improvement (%) + 100'] - df['Overall Efficiency Improvement (%) + 100'].iloc[0]
    df['log_average_overall_efficiency'] = (df['Overall Efficiency Improvement (%) + 100'] - df['Overall Efficiency Improvement (%) + 100'].iloc[0]) / (math.log(df['Overall Efficiency Improvement (%) + 100']) - math.log(df['Overall Efficiency Improvement (%) + 100'].iloc[0]))

    df['lmdi_contrib_engine_efficiency'] =  df['log_average_overall_efficiency'] * math.log(df['Engine Efficiency Improvement (%) + 100'] / df['Engine Efficiency Improvement (%) + 100'].iloc[0])
    df['lmdi_contrib_structural_efficiency'] =  df['log_average_overall_efficiency'] * math.log(df['Structural Efficiency Improvement (%) + 100'] / df['Structural Efficiency Improvement (%) + 100'].iloc[0])
    df['lmdi_contrib_aerodynamic_efficiency'] =  df['log_average_overall_efficiency'] * math.log(df['Aerodynamic Efficiency Improvement (%) + 100'] / df['Aerodynamic Efficiency Improvement (%) + 100'].iloc[0])
    df['lmdi_contrib_redisual'] = df['delta_overall_efficiency'] - df['lmdi_contrib_engine_efficiency'] - df['lmdi_contrib_structural_efficiency'] - df['lmdi_contrib_aerodynamic_efficiency']

    return df
    """
    pass
    

def _compute_lmdi_factor_contributions(
    aggregate_t1: float,
    aggregate_t2: float,
    factor_t1: float,
    factor_t2: float
) -> float:
    r"""
    Computes the contributions of changes in factors to the change in an aggregate
    according to the additive logarithmic mean Divisia index method I (LMDI-I).

    Given values for an aggregate $C$ and a factor $C_i$ at two points in time $(t_1, t_2)$,
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