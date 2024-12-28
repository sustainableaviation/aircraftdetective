# %%

from pathlib import Path
import sys
import os

import pandas as pd

module_path = str(Path(__file__).parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)

from aircraftdetective.calculations import engines
from aircraftdetective import config

# %%

df_engines_tsfc, pol1, pol2, _, _ = engines.determine_takeoff_to_cruise_tsfc_ratio(
    path_excel_engine_data_for_calibration=config['aircraft_detective']['url_xlsx_engine_tsfc_database']
)

df_engines = engines.scale_engine_data_from_icao_emissions_database(
    path_excel_engine_data_icao_in=config['ICAO']['url_xlsx_engine_emissions_databank'],
    path_excel_engine_data_icao_out='Scaled.xlsx',
    scaling_polynomial=pol2
)

df_aircraft = pd.read_excel(
    io=config['aircraft_detective']['url_xlsx_aircraft_database'],
    sheet_name='Data',
    header=[0, 1],
    engine='openpyxl',
)
df_aircraft = df_aircraft.pint.quantify(level=1)

engines_wildcard: set = engines.get_engine_designations_with_wildcard(df=df_aircraft)

df_engines = engines.average_engine_data_by_model(
    df=df_engines,
    engine_models_to_average=engines_wildcard
)

df_merged = pd.merge(
    left=df_aircraft,
    right=df_engines,
    how='left',
    left_on='Engine Designation',
    right_on='Engine Identification',
)

df_merged = engines.compute_engine_metrics(
    df=df_merged
)

# %%
