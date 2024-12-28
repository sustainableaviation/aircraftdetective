# %%

import numpy as np
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
from aircraftdetective.calculations import overall
from aircraftdetective.processing import statistics
from aircraftdetective.calculations import progress


df_aircraft = pd.read_excel(
    io=config['aircraft_detective']['url_xlsx_aircraft_database'],
    sheet_name='Raw Data',
    header=[0, 1],
    engine='openpyxl',
)
df_aircraft = df_aircraft.pint.quantify(level=1)


df_stats = statistics.process_data_usdot_t2(
    path_csv_t2=config['US_DOT']['url_csv_schedule_t2'],
    path_csv_aircraft_types=config['US_DOT']['url_csv_aircraft_types'],
)
df_stats = df_stats.groupby('Aircraft Designation (US DOT Schedule T2)')['Fuel/Available Seat Distance'].mean().reset_index()

df_aircraft_enriched = pd.merge(
    left=df_aircraft,
    right=df_stats,
    how='left',
    on='Aircraft Designation (US DOT Schedule T2)',
)

_, linear_fit, polynomial_fit, r_squared_linear, r_squared_polynomial = engines.determine_takeoff_to_cruise_tsfc_ratio(
    path_excel_engine_data_for_calibration=config['aircraft_detective']['url_xlsx_engine_tsfc_database'],
)

df_engines = engines.scale_engine_data_from_icao_emissions_database(
    path_excel_engine_data_icao_in=config['ICAO']['url_xlsx_engine_emissions_databank'],
    path_excel_engine_data_icao_out='Scaled.xlsx',
    scaling_polynomial=polynomial_fit
)

df_aircraft_enriched = pd.merge(
    left=df_aircraft_enriched,
    right=df_engines,
    how='left',
    left_on='Engine Designation',
    right_on='Engine Identification'
)

df_aircraft_enriched = weights.compute_weight_metrics(df=df_aircraft_enriched)
df_aircraft_enriched = aerodynamics.compute_aerodynamic_metrics(df=df_aircraft_enriched)
df_aircraft_enriched = engines.compute_engine_metrics(df=df_aircraft_enriched)
df_aircraft_enriched = overall.compute_overall_metrics(df=df_aircraft_enriched)


df_aircraft_literature = pd.read_excel(
    io=config['aircraft_detective']['url_xlsx_aircraft_database'],
    sheet_name='Literature Data',
    header=[0, 1],
    engine='openpyxl',
)
df_aircraft_literature = df_aircraft_literature.pint.quantify(level=1)

list_columns=['Energy Use', 'L/D', 'TSFC (cruise)', 'OEW/MTOW']
merge_column='Aircraft Designation (Literature)'

df_updated = pd.merge(
    left=df_aircraft_enriched,
    right=df_aircraft_literature[list_columns + [merge_column]],
    on=merge_column,
    how='left',
    suffixes=('', '_update')
)

df_updated = dataframe.update_column_data(
    df_main=df_aircraft_enriched,
    df_other=df_aircraft_literature,
    merge_column='Aircraft Designation (Literature)',
    list_columns=list_columns,
)

df_progress = progress.normalize_aircraft_efficiency_metrics(
    df=df_updated,
    baseline_aircraft_designation_wide='B707-120',
    baseline_aircraft_designation_narrow='B737-200C',
)

# %%

pol = progress.determine_polynomial_fit(
    df=df_progress,
    column='Engine Efficiency (relative)',
    degree=4,
)

import matplotlib.pyplot as plt
# Generate x values for plotting the polynomial
x_values = np.linspace(df_progress['YOI'].min(), df_progress['YOI'].max(), 500)
y_values = pol(x_values)

# Plot the polynomial
plt.plot(x_values, y_values, color='green', linestyle='--', label='Energy Use (polynomial fit)')
plt.legend(loc='upper left')


# %%

# plotting
import matplotlib.pyplot as plt
# unit conversion
cm = 1/2.54 # for inches-cm conversion
# time manipulation
from datetime import datetime
# data science
import numpy as np
import pandas as pd

fig, ax = plt.subplots(
    num = 'main',
    nrows = 1,
    ncols = 1,
    dpi = 300,
    figsize=(20*cm, 10*cm), # A4=(210x297)mm,
)

ax.set_xlim(1950, 2030)
ax.set_ylim(-30,150)

plt.scatter(
    df_progress['YOI'],
    df_progress['Engine Efficiency (relative)'],
    color = 'black',
    marker = 'o',
)
plt.scatter(
    df_progress['YOI'],
    df_progress['Aerodynamic Efficiency (relative)'],
    color = 'blue',
    marker = 'o',
)
plt.scatter(
    df_progress['YOI'],
    df_progress['Structural Efficiency (relative)'],
    color = 'red',
    marker = 'o',
)
plt.scatter(
    df_progress['YOI'],
    df_progress['Energy Use (relative)'],
    color = 'green',
    marker = 'o',
)

plt.legend(
    [
        'Engine Efficiency',
        'Aerodynamic Efficiency',
        'Structural Efficiency',
        'Energy Use',
    ],
    loc='upper left',
)