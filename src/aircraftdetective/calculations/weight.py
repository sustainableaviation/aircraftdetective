# %%
import pandas as pd
import numpy as np
import pint
ureg = pint.get_application_registry()

def calculate_weight_metrics(
    df: 'pd.DataFrame'
) -> 'pd.DataFrame':
    r"""
    Calculates weight-related metrics for aircraft data.

    Calculates the OEW/MTOW and OEW/Exit Limit ratios for each aircraft in the provided DataFrame, 
    where

    | Variable     | Description                          |
    |--------------|--------------------------------------|
    | OEW          | Operating Empty Weight               |
    | MTOW         | Maximum Takeoff Weight               |
    | Exit Limit   | Maximum number of pax allowed as per [emergency evacuation tests](https://www.easa.europa.eu/en/light/topics/aircraft-emergency-evacuation) |

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing aircraft data with necessary weight columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional weight metrics:

        - `OEW/MTOW`
        - `OEW/Exit Limit`

    Raises
    ------
    ValueError
        If required columns are missing in the input DataFrame.
    """
    df_func = df.copy()
    required_columns = ['OEW', 'MTOW', 'Pax Exit Limit']
    for col in required_columns:
        if col not in df_func.columns.get_level_values(0):
            raise ValueError(f"Missing required column: {col}")

    df_func['OEW/MTOW'] = df_func['OEW'] / df_func['MTOW']
    df_func['OEW/Exit Limit'] = df_func['OEW'] / df_func['Pax Exit Limit']

    
    grouped = df_func.groupby('Type', group_keys=False)

    return df_func