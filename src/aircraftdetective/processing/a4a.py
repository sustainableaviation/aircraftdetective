# %%
from aircraftdetective.data.hyperlinks import PATH_ZENODO_A4A_TRAFFIC_DATA
import pandas as pd

def process_a4a_traffic_data(
    path_xlsx_a4a: str | None = None,
) -> pd.DataFrame:
    """
    Processes A4A traffic and operations data.  
    Data includes yearly passenger load factors (=seat load factor (SLF)).

    Notes
    -----
    With no parameters passed, the function will download the relevant Excel file 
    from the relevant [Zenodo repository](https://doi.org/10.5281/zenodo.14382100).

    See Also
    --------
    [Weinold (2023) aircraft database in Excel format on Zenodo](https://doi.org/10.5281/zenodo.14382100)

    References
    ----------
    [Airlines for America (A4A) dataset "World Airlines Traffic and Capacity"](https://www.airlines.org/dataset/world-airlines-traffic-and-capacity/)

    Parameters
    ----------
    path_xlsx_a4a : str, optional
        The path or URL to the Excel file containing the A4A traffic data. Default is the Zenodo URL.

    Returns:
        pd.DataFrame: A DataFrame containing A4A global traffic data:

        - `Year`
        - `SLF`
    """
    if path_xlsx_a4a is None:
        path_xlsx_a4a = PATH_ZENODO_A4A_TRAFFIC_DATA
    df = pd.read_excel(PATH_ZENODO_A4A_TRAFFIC_DATA)
    df = df[['Year', 'PLF']]
    df.rename(columns={'PLF': 'SLF'}, inplace=True)
    return df