# %%

from pathlib import Path
import sys
import os

import pandas as pd

module_path = str(Path(__file__).parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)

from aircraftdetective.calculations import engines
from aircraftdetective.processing import operations
from aircraftdetective import config

# %%

df = operations.process_data_usdot_t2(
    path_csv_t2=config['USDOT']['url_csv_schedule_t2'],
    path_csv_aircraft_types=config['USDOT']['url_csv_aircraft_types'],
)

df = df.groupby('Aircraft Designation (US DOT Schedule T2)')['Fuel/Available Seat Distance'].mean().reset_index()