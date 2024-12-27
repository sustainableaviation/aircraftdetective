# %%

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
module_path = Path(__file__).resolve().parent.parent.parent
module_path = str(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)

from aircraftdetective.processing import databases

from aircraftdetective import config


df = enrich_aircraft_database(
    path_json_aircraft_database=config['aircraft_database_com']['url_json_aircraft_types'],
    path_json_engine_database=config['aircraft_database_com']['url_json_engine_models'],
    path_json_properties=config['aircraft_database_com']['url_json_properties'],
    path_json_manufacturers=config['aircraft_database_com']['url_json_manufacturers'],
)
# %%

def aggregate_aircraft_database(
    df_aircraft: pd.DataFrame
) -> pd.DataFrame:
    
    df_aircraft_aggregated = df_aircraft.groupby(
        [
            'Aircraft Designation',
            'Aircraft Manufacturer'
        ],
        as_index=False
    ).agg(
        {
            'Wingspan': 'mean',
            'Wingspan (Winglets)' : 'mean',
            'Wing Area': 'mean',
        }
    )


    df_aircraft_aggregated['Wingspan'] = df_aircraft_aggregated.apply(
        lambda row:
        row['Wingspan (Winglets)']
        if pd.notna(row['Wingspan (Winglets)'])
        else row['Wingspan'],
        axis=1
    )
    # https://pint-pandas.readthedocs.io/en/latest/user/common.html#units-in-cells-object-dtype-columns
    df_aircraft_aggregated = df_aircraft_aggregated.pint.convert_object_dtype()


    return df_aircraft_aggregated