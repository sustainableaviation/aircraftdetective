# %%

import pandas as pd
import pint
import pint_pandas
ureg = pint.get_application_registry() # https://pint-pandas.readthedocs.io/en/latest/user/common.html#using-a-shared-unit-registry

import sys
import os
module_path = os.path.abspath("/Users/michaelweinold/github/aircraftdetective")
if module_path not in sys.path:
    sys.path.append(module_path)

from aircraftdetective.calculations import engines
from aircraftdetective.auxiliary import dataframe
from aircraftdetective.processing import databases

from aircraftdetective import config

from aircraftdetective.calculations import aerodynamics
from aircraftdetective.calculations import weights
from aircraftdetective.calculations import engines

# %%


df_aircraft = pd.read_excel(
        io=config['aircraft_detective']['url_xlsx_aircraft_database'],
        sheet_name='Data',
        header=[0, 1],
        engine='openpyxl',
    )
df_aircraft = df_aircraft.pint.quantify(level=1)


df_engines = pd.read_excel(
    io=config['ICAO']['url_xlsx_engine_emissions_databank'],
    sheet_name='Gaseous Emissions and Smoke',
    header=0,
    converters={'Final Test Date': lambda x: int(pd.to_datetime(x).year)},
    engine='openpyxl',
)

pd.merge(
    left=df_aircraft,
    right=df_engines,
    how='left',
    left_on='Engine Designation',
    right_on='Engine Identification'
)

# %%
def analyze_engines(
    
    path_output_engines_scaled: str



def analyze_aircraft_and_engines(
    path_output_engines_scales: str
) -> None:

    df_aircraft = pd.read_excel(
        io=config['aircraft_detective']['url_xlsx_aircraft_database'],
        sheet_name='Data',
        header=[0, 1],
        engine='openpyxl',
    )
    df_aircraft = df_aircraft.pint.quantify(level=1)

    linear_fit, polynomial_fit, r_squared_linear, r_squared_polynomial = engines.determine_takeoff_to_cruise_tsfc_ratio(
        path_excel_engine_data_for_calibration=config['aircraft_detective']['url_xlsx_engine_tsfc_database'],

    )

    df_engines = engines.scale_engine_data_from_icao_emissions_database(
        path_excel_engine_data_icao=config['ICAO']['url_xlsx_engine_data'],
        scaling_polynomial=polynomial_fit
    )

    df_aircraft_enriched = pd.merge(
        left=df_aircraft,
        right=df_engines,
        how='left',
        left_on='Engine Designation',
        right_on='Engine Identification'
    )

    df_aircraft_enriched = weights.compute_weight_metrics(df=df_aircraft_enriched)
    df_aircraft_enriched = aerodynamics.compute_aerodynamic_metrics(df=df_aircraft_enriched)
    df_aircraft_enriched = engines.compute_engine_metrics(df=df_aircraft_enriched)


