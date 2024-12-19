# %%

from pathlib import Path
import pandas as pd
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

dict_properties = databases.read_properties_database(
    path_json_properties=config['aircraft-database-com']['url_json_properties']
)

dict_manufacturers = databases.read_manufacturers_database(
    path_json_manufacturers=config['aircraft-database-com']['url_json_manufacturers']
)

df_aircraft_types = databases.read_aircraft_database(
    path_json_aircraft_database=config['aircraft-database-com']['url_json_aircraft_types'],
    dict_properties=dict_properties,
    dict_manufacturers=dict_manufacturers
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
    df = df.pint.convert_object_dtype()


    return df_aircraft_aggregated