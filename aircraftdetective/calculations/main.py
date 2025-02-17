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

df_aircraft = df_aircraft[df_aircraft['Type'] != 'Regional']
# df_aircraft = df_aircraft[df_aircraft['Type'] != 'Narrow']

df_stats = statistics.process_data_usdot_t2(
    path_csv_t2=config['US_DOT']['url_csv_schedule_t2'],
    path_csv_aircraft_types=config['US_DOT']['url_csv_aircraft_types'],
)
df_stats = df_stats.groupby('Aircraft Designation (US DOT Schedule T2)')['Fuel/Available Seat Distance'].mean().reset_index()
df_stats = statistics.remove_outliers_usdot_t2(
    df=df_stats,
    aircraft_designations=['Airbus Industrie A310-200C/F', 'Boeing 777-300/300ER/333ER']
)

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

engines_wildcard: set = engines.get_engine_designations_with_wildcard(df=df_aircraft)

df_engines = engines.average_engine_data_by_model(
    df=df_engines,
    engine_models_to_average=engines_wildcard
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


df_updated = dataframe.update_column_data(
    df_main=df_aircraft_enriched,
    df_other=df_aircraft_literature,
    merge_column='Aircraft Designation (Literature)',
    list_columns=list_columns,
)


# %%

df_progress = progress.compute_normalized_aircraft_efficiency_metrics(
    df=df_updated,
    baseline_aircraft_designation='Comet 4',
)

pol_engine = progress.determine_polynomial_fit(
    df=df_progress,
    column='Engine Efficiency Improvement (%)',
    degree=2,
)
pol_aero = progress.determine_polynomial_fit(
    df=df_progress,
    column='Aerodynamic Efficiency Improvement (%)',
    degree=2,
)
pol_struct = progress.determine_polynomial_fit(
    df=df_progress,
    column='Structural Efficiency Improvement (%)',
    degree=2,
)
pol_overall = progress.determine_polynomial_fit(
    df=df_progress,
    column='Overall Efficiency Improvement (%)',
    degree=2,
)


df_pol = progress.create_efficiency_dataframe_from_polynomials(
    polynomial_overall=pol_overall,
    polynomial_engine=pol_engine,
    polynomial_aerodynamic=pol_aero,
    polynomial_structural=pol_struct,
    yoi_range=(1960, 2020),
)

df_ldmi = progress.compute_log_mean_divisia_index_of_efficiency(
    df=df_pol,
)


import matplotlib.pyplot as plt
# Generate x values for plotting the polynomial

ax, fig = plt.subplots()

plt.xlim(1958, 2030)
plt.ylim(-15,300)

# Plot the polynomial
plt.plot(
    df_pol['YOI'],
    df_pol['Overall Efficiency Improvement (%)'],
    color='green',
    linestyle='--',
    label='Overall Efficiency Improvement (%)',
)
plt.scatter(
    df_progress['YOI'],
    df_progress['Overall Efficiency Improvement (%)'],
    color = 'green',
    marker = 'o',
)

plt.plot(
    df_pol['YOI'],
    df_pol['Engine Efficiency Improvement (%)'],
    color='black',
    linestyle='--',
    label='Engine Efficiency Improvement (%)',
)
plt.scatter(
    df_progress['YOI'],
    df_progress['Engine Efficiency Improvement (%)'],
    color = 'black',
    marker = 'o',
)

plt.plot(
    df_pol['YOI'],
    df_pol['Aerodynamic Efficiency Improvement (%)'],
    color='blue',
    linestyle='--',
    label='Aerodynamic Efficiency Improvement (%)',
)
plt.scatter(
    df_progress['YOI'],
    df_progress['Aerodynamic Efficiency Improvement (%)'],
    color = 'blue',
    marker = 'o',
)


plt.legend(loc='upper left')

list_cols_for_plot = [
    'YOI',
    'lmdi_contrib_engine_efficiency',
    'lmdi_contrib_structural_efficiency',
    'lmdi_contrib_aerodynamic_efficiency',
    'lmdi_contrib_redisual'
]

df_ldmi[list_cols_for_plot].plot(
    kind='bar',
    x='YOI',
    stacked=True,
)

# %%

import plotly.io as pio
import plotly.express as px

pio.renderers.default = "notebook"

df_plot = df_progress.dropna(subset=['Aerodynamic Efficiency Improvement (%)'])

# https://plotly.com/python-api-reference/generated/plotly.express.scatter
fig = px.scatter(
    x=df_progress['YOI'],
    y=df_progress['Aerodynamic Efficiency Improvement (%)'].astype('float64'),
    hover_name=df_progress['Aircraft Designation'],
)

# Show the figure
fig.show()

# %%

df_plot = df_progress.dropna(subset=['Overall Efficiency Improvement (%)'])

# https://plotly.com/python-api-reference/generated/plotly.express.scatter
fig = px.scatter(
    x=df_progress['YOI'],
    y=df_progress['Overall Efficiency Improvement (%)'].astype('float64'),
    hover_name=df_progress['Aircraft Designation'],
)

# Show the figure
fig.show()
