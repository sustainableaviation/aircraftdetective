# %%

import pandas as pd
from pathlib import Path
import processing
import processing.statistics
import processing.literature
import calculations.engines

path_csv_t2 = Path("/Users/michaelweinold/github/Aircraft-Performance/database/rawdata/USDOT/T_SCHEDULE_T2.csv")
path_csv_aircraft_types = Path("/Users/michaelweinold/github/Aircraft-Performance/database/rawdata/USDOT/L_AIRCRAFT_TYPE (1).csv")
path_xlsx_babikian = Path('/Users/michaelweinold/Library/CloudStorage/OneDrive-TheWeinoldFamily/Documents/University/PhD/Data/Aircraft Performance/Eficiency Data Babikian et al (2002).xlsx')
path_xlsx_concordance = Path('/Users/michaelweinold/Library/CloudStorage/OneDrive-TheWeinoldFamily/Documents/University/PhD/Data/Aircraft Performance/Aircraft Name Concordance.xlsx')

path_xlsx_engines_tsfc = Path('/Users/michaelweinold/Library/CloudStorage/OneDrive-TheWeinoldFamily/Documents/University/PhD/Data/Aircraft Performance/Engine Database (TSFC Data).xlsx')
path_xlsx_engines_icao = Path('/Users/michaelweinold/Downloads/edb-emissions-databank_v30__web_.xlsx')

# %%

linear_fit, polynomial_fit, r_squared_linear, r_squared_polynomial = calculations.engines.determine_takeoff_to_cruise_tsfc_ratio(path_xlsx_engines_tsfc)

calculations.engines.scale_engine_data_from_icao_emissions_database(
    path_excel_engine_data_icao_in=path_xlsx_engines_icao,
    path_excel_engine_data_icao_out='test.xlsx',
    scaling_polynomial=linear_fit
)

# %%

df_concordance = pd.read_excel(
    io=path_xlsx_concordance,
    sheet_name='Data',
    engine='openpyxl'
)

df_t2 = processing.statistics.process_data_usdot_t2(
    path_csv_t2=path_csv_t2,
    path_csv_aircraft_types=path_csv_aircraft_types
)


# %%

df_t2_grouped = df_t2.groupby('Aircraft Designation (US DOT Schedule T2)').median()

df_t2_grouped_our_designation = pd.merge(
    left=df_t2_grouped,
    right=df_concordance[['Aircraft Designation (OUR)', 'Aircraft Designation (US DOT Schedule T2)']],
    on='Aircraft Designation (US DOT Schedule T2)',
    how='left',
).drop(columns=['Aircraft Designation (US DOT Schedule T2)'])

# %%

df_babikian = processing.literature.process_data_babikian_figures(path_xlsx_babikian)
df_babikian_our_designation = pd.merge(
    left = df_babikian,
    right=df_concordance[['Aircraft Designation (OUR)', 'Aircraft Designation (Babikian et al.)']],
    on='Aircraft Designation (Babikian et al.)',
    how='left',
).drop(columns=['Aircraft Designation (Babikian et al.)'])

df_overall = pd.merge(
    left=df_t2_grouped_our_designation,
    right=df_babikian_our_designation,
    on='Aircraft Designation (OUR)',
    how='outer',
)