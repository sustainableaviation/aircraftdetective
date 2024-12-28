# %%

import pandas as pd
import pint
import pint_pandas
ureg = pint.get_application_registry() # https://pint-pandas.readthedocs.io/en/latest/user/common.html#using-a-shared-unit-registry


df_aircraft_literature = pd.read_excel(
    io=config['aircraft_detective']['url_xlsx_aircraft_database'],
    sheet_name='Literature Data',
    header=[0, 1],
    engine='openpyxl',
)
df_aircraft_literature = df_aircraft_literature.pint.quantify(level=1)


# %%

# Example DataFrames
df1 = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
}, index=[0, 1, 2])

df2 = pd.DataFrame({
    'A': [7, 8],
    'B': [9, 10],
}, index=[1, 3])  # Different indices

# Update df1 using df2
#df1.update(df2)

print(df1)
