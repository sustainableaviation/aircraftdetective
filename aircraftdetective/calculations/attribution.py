import math
import pandas as pd


def compute_lmdi_efficiency_contributions(
    eff_total_t1: float,
    eff_total_t2: float,
    dict_subeff_t1: dict,
    dict_subeff_t2: dict
) -> float:
    r"""
    Given a value for total efficiency $C$ and an arbitrary number of values for
    sub-efficiencies $C_i$ at two points in time $(t_1, t_2)$,
    computes the contribution $X$ of the change in each sub-efficiency $\Delta C_i$
    to the change in total $\Delta C$ through the
    additive logarithmic mean Divisia index method I (LMDI-I).

    The method is _additive_, meaning that the identity

    $$
    \Delta C = \Delta C_1 + \Delta C_2 + ... + \Delta C_n
    $$
 
    where

    | Symbol | Unit     | Description        |
    |--------|----------|--------------------|
    | $C$    | -        | Efficiency         |
    | $C_i$  | -        | Sub-efficiency $i$ |

    holds.

    Notes
    -----
    Sub-efficiencies must be given as a dictionary of the form:

    ```python
    dict_subeff = {
        'subeff1': (30,45)
        'subeff2': (80,83)
    }
    ```

    Where tuples indicated values $(C_i(t_1), C_i(t_2)$.

    References
    ----------
    Ang, Beng Wah, and Tian Goh. "Index decomposition analysis for comparing emission scenarios: Applications and challenges."
    _Energy Economics_ 83 (2019): 74-87. doi:[10.1016/j.eneco.2019.06.013](https://doi.org/10.1016/j.eneco.2019.06.013)
    Eqn. (28) and Table 5 in doi:[10.1016/S0360-5442(00)00039-6](https://doi.org/10.1016/S0360-5442(00)00039-6)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the following columns:
        - 'Overall Efficiency Improvement (%)'
        - 'Engine Efficiency Improvement (%)'
        - 'Structural Efficiency Improvement (%)'
        - 'Aerodynamic Efficiency Improvement (%)'

    Returns
    -------
    float
        Contribution of change in sub-efficiency to change in total efficiency
    """
    dict_subeff_contrib = dict_subeff.copy()
    for subeff in dict_subeff.keys():
        log_mean_total_eff = ((eff_total_t2 - eff_total_t1) / (math.ln(eff_total_t2) - math.ln(eff_total_t1)))
        delta_subeff = log_mean_total_eff * math.ln(dict_subeff[subeff][1] / dict_subeff[subeff][0])
        dict_subeff_contrib[subeff] = delta_subeff
    return dict_subeff_contrib




for col in df.columns:
    df[f'{col} + 100'] = df[col] + 100

df['delta_overall_efficiency'] = df['Overall Efficiency Improvement (%) + 100'] - df['Overall Efficiency Improvement (%) + 100'].iloc[0]
df['log_average_overall_efficiency'] = (df['Overall Efficiency Improvement (%) + 100'] - df['Overall Efficiency Improvement (%) + 100'].iloc[0]) / (math.log(df['Overall Efficiency Improvement (%) + 100']) - math.log(df['Overall Efficiency Improvement (%) + 100'].iloc[0]))

df['lmdi_contrib_engine_efficiency'] =  df['log_average_overall_efficiency'] * math.log(df['Engine Efficiency Improvement (%) + 100'] / df['Engine Efficiency Improvement (%) + 100'].iloc[0])
df['lmdi_contrib_structural_efficiency'] =  df['log_average_overall_efficiency'] * math.log(df['Structural Efficiency Improvement (%) + 100'] / df['Structural Efficiency Improvement (%) + 100'].iloc[0])
df['lmdi_contrib_aerodynamic_efficiency'] =  df['log_average_overall_efficiency'] * math.log(df['Aerodynamic Efficiency Improvement (%) + 100'] / df['Aerodynamic Efficiency Improvement (%) + 100'].iloc[0])
df['lmdi_contrib_redisual'] = df['delta_overall_efficiency'] - df['lmdi_contrib_engine_efficiency'] - df['lmdi_contrib_structural_efficiency'] - df['lmdi_contrib_aerodynamic_efficiency']

return df
