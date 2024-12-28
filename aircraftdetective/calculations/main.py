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
from aircraftdetective.calculations import overall
from aircraftdetective.processing import statistics
from aircraftdetective.calculations import progress

# %%


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

# %%

df_aircraft_literature = pd.read_excel(
    io=config['aircraft_detective']['url_xlsx_aircraft_database'],
    sheet_name='Literature Data',
    header=[0, 1],
    engine='openpyxl',
)
df_aircraft_literature = df_aircraft_literature.pint.quantify(level=1)

list_columns=['Energy Use', 'L/D', 'TSFC (cruise)']
merge_column='Aircraft Designation'

df_updated = pd.merge(
    left=df_aircraft_enriched,
    right=df_aircraft_literature[list_columns + [merge_column]],
    on=merge_column,
    how='left',
    suffixes=('', '_update')
)

df_updated["Energy Use"].combine_first(df_updated["Energy Use_update"])

df_updated["TSFC (cruise)"].combine_first(df_updated["TSFC (cruise)_update"])

df_updated_form = dataframe.update_column_data(
    df_main=df_aircraft_enriched,
    df_other=df_aircraft_literature,
    merge_column='Aircraft Designation',
    list_columns=list_columns,
)


# %%

list_columns=['Energy Use', 'L/D', 'TSFC (cruise)']

df_aircraft_enriched = df_aircraft_enriched.set_index('Aircraft Designation')
df_aircraft_literature = df_aircraft_literature.set_index('Aircraft Designation')

# %%

df_aircraft_enriched.combine(
    other=df_aircraft_literature[list_columns],
    func=lambda x, y: x if pd.isna(y) else y,
    overwrite=False,
)


# %%

df_progress = progress.normalize_aircraft_efficiency_metrics(
    df=df_aircraft_enriched,
    baseline_aircraft_designation_wide='B707-120',
    baseline_aircraft_designation_narrow='B737-200C',
)
# %%

df = df_aircraft_enriched

row_baseline_aircraft_wide = df[df['Aircraft Designation'] == 'B707-120']
row_baseline_aircraft_narrow = df[df['Aircraft Designation'] == 'B737-200C']
    
eu_baseline = row_baseline_aircraft_wide['Fuel/Available Seat Distance'].values[0]
tsfc_baseline = row_baseline_aircraft_wide['TSFC (cruise)'].values[0]
ld_baseline = row_baseline_aircraft_wide['L/D'].values[0]
mtow_baseline_wide = row_baseline_aircraft_wide['MTOW'].values[0]
mtow_baseline_narrow = row_baseline_aircraft_narrow['MTOW'].values[0]

# %%

df['Energy Use (relative)'] = 100 * (eu_baseline - df['Fuel/Available Seat Distance']) / eu_baseline
df['Engine Efficiency (relative)'] = 100 * (tsfc_baseline - df['TSFC (cruise)']) / tsfc_baseline
df['Aerodynamic Efficiency (relative)'] = 100 * (ld_baseline - df['L/D']) / ld_baseline
df['Structural Efficiency'] = df['MTOW'] / df['Pax Exit Limit']
df['Structural Efficiency (relative)'] = 100 * (mtow_baseline_wide - df[df['Type'] == 'Wide']['Structural Efficiency']) / mtow_baseline_wide
df['Structural Efficiency (relative)'] = 100 * (mtow_baseline_narrow - df[df['Type'] == 'Narrow']['Structural Efficiency']) / mtow_baseline_narrow