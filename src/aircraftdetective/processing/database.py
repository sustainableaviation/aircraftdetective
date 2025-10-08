# %%

import pandas as pd
import pint
import pint_pandas
ureg = pint.get_application_registry()
from aircraftdetective import ureg
import importlib.resources as pkg_resources
from aircraftdetective import data
from aircraftdetective.data.hyperlinks import (
    PATH_ZENODO_AIRCRAFT_DATABASE_FILE,
    PATH_ZENODO_ENGINE_DATABASE_FILE,
    PATH_ZENODO_BABIKIAN_FILE
)
from aircraftdetective.calculations.engines import (
    determine_takeoff_to_cruise_tsfc_ratio,
    scale_engine_data_from_icao_emissions_database
)

from aircraftdetective.processing.acftdb import _read_engine_database
from aircraftdetective.utility.tabular import left_merge_wildcard

df = pd.read_excel(
    io=PATH_ZENODO_AIRCRAFT_DATABASE_FILE,
    sheet_name='Raw Data',
    header=[0,1],
    engine='openpyxl'
)
df = df.pint.quantify(level=-1)


df_babikian = pd.read_excel(
    io=PATH_ZENODO_BABIKIAN_FILE,
    sheet_name='Data (Figure 2)',
    header=[0,1],
    engine='openpyxl'
)
df_babikian = df_babikian.pint.quantify(level=-1)
df_babikian = df_babikian[df_babikian['Aircraft Designation'].notna() & (df_babikian['Aircraft Designation'] != '???')]

df_engines = _read_engine_database()

dict_tsfc_scaling = determine_takeoff_to_cruise_tsfc_ratio(degree=2, plot=True)
df_engines_scaled = scale_engine_data_from_icao_emissions_database(
    scaling_polynomial=dict_tsfc_scaling['TSFC (cruise)'],
)
df_engines_scaled.drop(columns=['Final Test Date'], inplace=True)

df_merged = left_merge_wildcard(
    df_left=df,
    df_right=df_engines_scaled,
    left_on='Engine Designation (ICAO)',
    right_on='Engine Identification',
)


df_merged = left_merge_wildcard(
    df_left=df_merged,
    df_right=df_engines,
    left_on='Engine Designation (aircraft-database.com)',
    right_on='Engine Designation',
)

from aircraftdetective.utility.tabular import export_typed_dataframe_to_excel

export_typed_dataframe_to_excel(df=df_merged, path='out.xlsx')

#%%

from aircraftdetective.calculations.engines import (
    calculate_air_mass_flow_rate,
    calculate_engine_efficiencies
)

df_merged = calculate_air_mass_flow_rate(df_merged)

# %%

from aircraftdetective.processing.usdot import process_data_usdot_t2

df_t2 = process_data_usdot_t2()

df_with_dot = pd.merge(
    how='left',
    left=df_merged,
    right=df_t2,
    left_on='Aircraft Designation (US DOT Schedule T2)',
    right_on='Aircraft Designation (US DOT Schedule T2)'
)

# %%

from aircraftdetective.calculations.engines import calculate_engine_efficiencies
calculate_engine_efficiencies(df=df_with_dot)
# %%
