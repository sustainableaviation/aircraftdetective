import pint
import pint_pandas
ureg = pint.get_application_registry()

import pandas as pd

def compute_overall_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    fuel_heating_value = 44.1 * ureg('MJ/kg')
    fuel_density = 0.803 * ureg('kg/L')

    df["Energy Use"] = df["Fuel/Available Seat Distance"] * fuel_heating_value * fuel_density
    df["Energy Use"] = df["Energy Use"].astype('pint[MJ/km]')
    df["Fuel/Available Seat Distance"] = df["Fuel/Available Seat Distance"].astype('pint[L/km]')

    return df