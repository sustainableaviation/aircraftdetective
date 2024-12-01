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
        names=['EU (Babikian et al.)', 'Aircraft Designation (Babikian et al.)'],
        engine='openpyxl',
    )
    df = df.astype(
        {
            'EU (Babikian et al.)': 'pint[MJ/km]',
            'Aircraft Designation (Babikian et al.)': 'string',
        }
    )
    df = df.dropna(how='any', axis=0)
    df = df[df['Aircraft Designation (Babikian et al.)'] != '???']
    
    return df