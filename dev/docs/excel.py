# %%
import pandas as pd
df = pd.read_excel(
    "https://zenodo.org/records/13119393/files/LEDCOM2003.xlsx",
    sheet_name="Dry_Etch",
    header=0,
    usecols="A:D",
    skiprows=0,
    engine='openpyxl'
)
