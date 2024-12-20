# %%

from pathlib import Path
import pandas as pd
import pint_pandas


def rename_columns_and_set_units(
    df: pd.DataFrame,
    return_only_renamed_columns: bool,
    column_names_and_units: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """
    Given a dataframe and a list of tuples describing the column names and units, rename the columns and set the units.

    _extended_summary_

    columns_and_units = [
        ("Engine Identification", "Engine Identification", str),
        ("Final Test Date", "Final Test Date", int),
        ("Fuel Flow T/O (kg/sec)", "Fuel Flow T/O", "pint[kg/s]"),
        ("B/P Ratio", "B/P Ratio", "pint[dimensionless]"),
    ]

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    column_names_and_units : list[tuple[str, str, pint_pandas.pint_array.PintType]]
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """

    for col_name_old, col_name_new, dtype in column_names_and_units:
        if col_name_old in df.columns:
            df = df.rename(columns={col_name_old: col_name_new})
            if dtype != None:
                df[col_name_new] = df[col_name_new].astype(dtype)

    if return_only_renamed_columns == True:
        return df[[col_name_new for col_name_old, col_name_new, dtype in column_names_and_units]]

    return df


def _return_short_units(dtype: pint_pandas.pint_array.PintType) -> str:
    """
    Given a pint_pandas PintType object, return the short unit string.
    If the object does not have a unit attribute, return "No Unit".

    Notes
    -----
    The "No Unit" string which is returned when the object does not have a unit attribute must not be changed.
    This is because `pint_pandas` [will (be default) ignore columns have a "No Unit" unit string](https://pint-pandas.readthedocs.io/en/latest/user/reading.html#pandas-dataframe-accessors).

    See Also
    --------
    - [Pint String Formatting Specifications](https://pint.readthedocs.io/en/stable/user/formatting.html)

    Parameters
    ----------
    dtype : pint_pandas.pint_array.PintType
        PintType object (eg. "kilowatt").

    Returns
    -------
    str
        Short unit string (eg. "kW").
    """
    try:
        return f"{dtype.units:~P}"
    except AttributeError:
        return "No Unit"


def export_typed_dataframe_to_excel(
        df: pd.DataFrame,
        path: Path
) -> None:
    """
    Given a DataFrame with PintArray columns, export it to an Excel file with the short units in the second row.
    If a column is no PintArray (=has no physical unit), "No Unit" is used as a unit description.

    For example, a DataFrame with the following columns and units:

    YOI                               pint[year]
    TSFC       pint[milligram / newton / second]
    Comment                               object

    will be exported to an Excel file with the following structure:

    | YOI   | TSFC   | Comment |
    |-------|--------|---------|
    | a     | mg/N/s | No Unit |
    |-------|--------|---------|
    | 1999  | 14.32  | FooBar  |
    | (...) | (...)  | (...)   |

    Note
    ----
    Simply using the df.ping.dequantify() method will not work, as it will throw units into a second-level column index.
    Exporting this multi-index DataFrame to Excel directly will result in an empty row that breaks pd.read_excel().

    - [Related Pandas GitHub Issue](https://github.com/pandas-dev/pandas/issues/27772)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with PintArray columns.
    path : Path
        Destination (=export) absolute file path.
    """

    short_units: pd.Series = df.dtypes.apply(lambda x: _return_short_units(x))

    df_dequantified = df.pint.dequantify()
    df_dequantified.columns = df_dequantified.columns.droplevel(1)

    df_export = pd.concat(
        objs=[
            pd.DataFrame(short_units).T,
            df_dequantified
        ],
        axis=0,
        ignore_index=True
    )

    df_export.to_excel(
        path,
        freeze_panes=(2, 0),
        sheet_name='Data',
        header=True,
        index=False,
        engine='openpyxl',
    )


def explode_column_with_list_elements(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Given a DataFrame, explode column values of the `propertyValues` column.
    
    For example, from a DataFrame of the kind:

    | id | (...) | propertyValues                                                          |
    |----|-------|-------------------------------------------------------------------------|
    | 1  | (...) | [{"property":"345cde","value":42}, {"property":"456def","value":12.45}] |
    | 2  | (...) | [{"property":"345cde","value":13}, {"property":"456def","value":23.56}] |

    the function generates a dataframe of the kind:

    | id | (...) | 345cde | 456def |
    |----|-------|--------|--------|
    | 1  | (...) | 42     | 12.45  |
    | 2  | (...) | 13     | 23.56  |

    It does so by first generating a Pandas Series of the kind:

    | index | propertyValues                                                        |
    |-------|-----------------------------------------------------------------------|
    | 0     | {"property":"345cde","value":42}, {"property":"456def","value":12.45} |
    | 1     | {"property":"345cde","value":13}, {"property":"456def","value":23.56} |

    and converting it to a DataFrame of the kind:

    | index | 345cde | 456def |
    |-------|--------|--------|
    | 0     | 42     | 12.45  |
    | 1     | 13     | 23.56  |

    which is concatenated with the original DataFrame.
    """

    series_property_values: pd.Series = df["propertyValues"].apply(
        lambda x: {item["property"]: item["value"] for item in x}
    )

    df_exploded_values = pd.DataFrame(series_property_values.tolist())

    df = pd.concat(
        objs=[df.drop(columns=["propertyValues"]), df_exploded_values],
        axis=1
    )

    return df

import re
def average_engine_model_data(
    df: pd.DataFrame,
    list_engine_models_to_average: list[str],
    list_columns_to_average: list[str],
) -> pd.DataFrame:
    """

    _extended_summary_

    For example, given a list of engines to average:

    ['GEnx-1B.*']

    and a DataFrame of the kind:

    | Engine Identification | (...) | Rated Thrust      |
    |-----------------------|-------|-------------------|
    | 'GEnx-1B54'           | (...) | 255.3             |
    | 'GEnx-1B58'           | (...) | 271.3             |
    | 'GEnx-1B64'           | (...) | 298               |

    The function returns a DataFrame of the kind:

    | Engine Identification | (...) | Rated Thrust      |
    |-----------------------|-------|-------------------|
    | 'GEnx-1B*'            | (...) | 274.87            |



    Parameters
    ----------
    df : pd.DataFrame
        _description_
    list_engine_models_to_average : list[str]
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """

    df['Engine Identification'] = df['Engine Identification'].apply(
        lambda engine_identification: next((engine_pattern for engine_pattern in list_engine_models_to_average if re.match(engine_pattern, engine_identification)), None)
    )
    df = df.dropna(subset=['Engine Identification'])
