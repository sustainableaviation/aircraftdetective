# %%

import pandas as pd
import pint
import pint_pandas
from aircraftdetective import ureg
import importlib.resources as pkg_resources
from aircraftdetective import data
from aircraftdetective.data.hyperlinks import (
    PATH_ZENODO_AIRCRAFT_DATABASE_FILE,
    PATH_ZENODO_ENGINE_DATABASE_FILE,
    PATH_ZENODO_BABIKIAN_FILE
)

from aircraftdetective.processing.acftdb import _read_engine_database
from aircraftdetective.utility.tabular import left_merge_wildcard

df = pd.read_excel(
    io=PATH_ZENODO_AIRCRAFT_DATABASE_FILE,
    sheet_name='Raw Data',
    header=[0,1],
    engine='openpyxl'
)
df = df.pint.quantify(level=-1)


df_babikian = pd.read_excel(
    io=PATH_ZENODO_BABIKIAN_FILE,
    sheet_name='Data (Figure 2)',
    header=[0,1],
    engine='openpyxl'
)
df_babikian = df_babikian.pint.quantify(level=-1)
df_babikian = df_babikian[df_babikian['Aircraft Designation'].notna() & (df_babikian['Aircraft Designation'] != '???')]

df_engines = _read_engine_database()

merged_wildcard = left_merge_wildcard(
    df_left=df,
    df_right=df_engines,
    left_on='Engine Designation',
    right_on='Engine Designation',
)
