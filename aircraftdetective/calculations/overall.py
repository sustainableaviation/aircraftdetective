# %%

import pint
import pint_pandas
ureg = pint.get_application_registry()


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import sys
import os
module_path = os.path.abspath("/Users/michaelweinold/github/aircraftdetective")
if module_path not in sys.path:
    sys.path.append(module_path)

from aircraftdetective.calculations import engines
from aircraftdetective.auxiliary import dataframe
from aircraftdetective.processing import databases
from aircraftdetective.calculations import aerodynamics
from aircraftdetective import config

# %%

path_aircraft_database = '/Users/michaelweinold/Library/CloudStorage/OneDrive-TheWeinoldFamily/Documents/University/PhD/Data/Aircraft Performance/Aircraft Database.xlsx'
path_engine_tsfc_database = '/Users/michaelweinold/Library/CloudStorage/OneDrive-TheWeinoldFamily/Documents/University/PhD/Data/Aircraft Performance/Engine Database (TSFC Data).xlsx'
url_engine_icao_database = 'https://www.easa.europa.eu/en/downloads/131424/en'

df_aircraft = pd.read_excel(
    io=path_aircraft_database,
    sheet_name='Data',
    header=[0, 1],
    engine='openpyxl',
)
df_aircraft = df_aircraft.pint.quantify(level=1)



# %%

df_aircraft_enriched = pd.merge(
    left=df_aircraft,
    right=df_engines,
    how='left',
    left_on='Engine Designation',
    right_on='Engine Identification'
)


# %%

df_aircraft = pd.read_excel(
    io=config['aircraftdetective']['url_xlsxl_aircraft_database'],
    sheet_name='Data',
    header=[0, 1],
    engine='openpyxl',
)
df_aircraft = df_aircraft.pint.quantify(level=1)


df_aircraft_database_com = databases.enrich_aircraft_database(
    path_json_aircraft_database=config['aircraft-database-com']['url_json_aircraft_types'],
    path_json_properties=config['aircraft-database-com']['url_json_properties'],
    path_json_manufacturers=config['aircraft-database-com']['url_json_manufacturers'],
)



# %%
df_aircraft_database_com = databases.aggregate_aircraft_database(df_aircraft_database_com)

df_aircraft_enriched = pd.merge(
    left=df_aircraft,
    right=df_aircraft_database_com,
    how='left',
    on='Aircraft Designation (aircraft-database.com)',
)

pol1, pol2, _, _ = engines.determine_takeoff_to_cruise_tsfc_ratio(
    path_excel_engine_data_for_calibration=config['aircraftdetective']['url_xlsx_engine_tsfc_database']
)


df_engines =engines.scale_engine_data_from_icao_emissions_database(
    path_excel_engine_data_icao_in='https://www.easa.europa.eu/en/downloads/131424/en',
    path_excel_engine_data_icao_out='Scaled.xlsx',
    scaling_polynomial=pol2
)

df_aircraft_enriched = pd.merge(
    left=df_aircraft_enriched,
    right=df_engines,
    how='left',
    left_on='Engine Designation',
    right_on='Engine Identification'
)

# %%

aerodynamics.compute_aerodynamic_efficiency(
    df=df_aircraft_enriched,
    beta=0.8,
)
