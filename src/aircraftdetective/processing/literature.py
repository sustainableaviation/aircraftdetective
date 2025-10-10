# %%
import pandas as pd
import pint
ureg = pint.get_application_registry()

from aircraftdetective.data.hyperlinks import PATH_ZENODO_BABIKIAN_FILE

def process_data_babikian_figures(
    path_xlsx_babikian: str = PATH_ZENODO_BABIKIAN_FILE,
) -> pd.DataFrame:
    """
    Processes the Babikian et al. (2002) aircraft efficiency data from the provided Excel file.

    Notes
    -----
    With no parameters passed, the function will download the relevant Excel file 
    from the relevant [Zenodo repository](https://doi.org/10.5281/zenodo.14560913).


    See Also
    --------
    [Figure data from Babikian et al. (2002) in Excel format on Zenodo](https://doi.org/10.5281/zenodo.14560913)
    
    References
    -----------
    [Babikian et al. (2002)](https://doi.org/10.1016/S0969-6997(02)00020-0)

    Returns
    -------
    pd.DataFrame
        [`pint-pandas`](https://pint-pandas.readthedocs.io/en/latest/) DataFrame containing the Babikian et al. (2002) aircraft efficiency data.
    """
    df = pd.read_excel(
        io=path_xlsx_babikian,
        sheet_name='Data (Figure 2)',
        header=[0,1],
        index_col=None,
        engine='openpyxl',
    )
    df = df.pint.quantify(level=-1)
    df = df[df['Aircraft Designation'].notna() & (df['Aircraft Designation'] != '???')]
    df.drop(
        columns=
        [
            'Year',
            'Aircraft Type',
            'Source',
            'Source Section',
            'Comment'
        ],
    inplace=True)
    return df