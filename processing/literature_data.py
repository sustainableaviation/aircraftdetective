# %%

from pathlib import Path
import pandas as pd
import pint_pandas


def process_data_babikian_figures(
    path_xlsx_babikian: Path,
) -> pd.DataFrame:
    """
    """
    df = pd.read_excel(
        io=path_xlsx_babikian,
        sheet_name='Data (Figure 2)',
        header=0,
        index_col=None,
        usecols='B:C',
        names=['EU', 'Aircraft Designation (Babikian)'],
        engine='openpyxl',
    )
    df = df.astype(
        {
            'EU': 'pint[MJ/km]',
            'Aircraft Designation (Babikian)': 'pint[]',
        }
    )
    return df


path_xlsx_babikian = Path('/Users/michaelweinold/Library/CloudStorage/OneDrive-TheWeinoldFamily/Documents/University/PhD/Data/Aircraft Performance/Data from Literature.xlsx')

df = process_data_babikian_figures(path_xlsx_babikian)