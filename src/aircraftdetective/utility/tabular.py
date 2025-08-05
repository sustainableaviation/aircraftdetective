# %%
import pint_pandas
from pathlib import Path
import pandas as pd
from aircraftdetective import ureg

def rename_columns_and_set_units(
    df: pd.DataFrame,
    return_only_renamed_columns: bool,
    column_names_and_units: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """
    Given a DataFrame and a list of tuples describing the column names and units, renames the columns and set the units.

    ```
    columns_and_units = [
        ("<old column name>", "<new column name>", <new column unit>),
        (...)
    ```

    For example:

    ```
    columns_and_units = [
        ("Engine Identification", "Engine Identification", str),
        ("B/P Ratio", "B/P Ratio", "pint[dimensionless]"),
        ("Fuel Flow T/O (kg/sec)", "Fuel Flow T/O", "pint[kg/s]"),
    ]
    ```

    Notes
    -----
    Units can be any [Pandas data type](https://pandas.pydata.org/docs/reference/arrays.html) or [Pint unit string](https://pint.readthedocs.io/en/stable/user/formatting.html#pint-format-types).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    return_only_renamed_columns : bool
        If True, only the renamed columns are included in the returned DataFrame.
    column_names_and_units : list[tuple[str, str, pint_pandas.pint_array.PintType]]
        List of tuples describing the column names and units.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns and set units.
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
    Given a Pandas column `dtype` object, returns the short unit string.
    If the object is not of pint_pandas PintType, returns "No Unit".

    Notes
    -----
    The "No Unit" string which is returned when the object does not have a unit attribute should not be changed.
    This is because `pint_pandas` [will (by default) ignore columns have a "No Unit" unit string](https://pint-pandas.readthedocs.io/en/latest/user/reading.html#pandas-dataframe-accessors).

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
    Given a DataFrame with [PintArray columns](https://pint-pandas.readthedocs.io/en/latest/), export it to an Excel file with the [short units](https://pint.readthedocs.io/en/stable/user/formatting.html#pint-format-types) in the second row.
    If a column is no PintArray (=has no physical unit), "No Unit" is used as a unit description.

    For example, a DataFrame with the following columns and units:

    ```
    YOI                               pint[year]
    TSFC       pint[milligram / newton / second]
    Comment                               object
    ```

    will be exported to an Excel file with the following structure:

    | YOI   | TSFC       | Comment     |
    |-------|------------|-------------|
    | **a** | **mg/N/s** | **No Unit** |
    | 1999  | 14.32      | FooBar      |
    | (...) | (...)      | (...)       |

    Note
    ----
    Simply using the df.ping.dequantify() method will not work, as it will throw units into a second-level column index.
    Exporting this multi-index DataFrame to Excel directly will result in an empty row that breaks `pd.read_excel()`.

    - [Related Pandas GitHub Issue](https://github.com/pandas-dev/pandas/issues/27772)

    See Also
    --------
    - [Pint String Formatting Specifications](https://pint.readthedocs.io/en/stable/user/formatting.html)

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


def update_column_data(
    df_main: pd.DataFrame,
    df_other: pd.DataFrame,
    merge_column: str,
    list_columns: list[str],
) -> pd.DataFrame:
    """_summary_

    _extended_summary_

    Given a first DataFrame of the kind:

    | Aircraft Designation | (...) | TSFC (cruise) |
    |----------------------|-------|---------------|
    | B707-120             | (...) |               |
    | B737-200C            | (...) | 0.5           |
    | A380-800             | (...) |               |

    and a second DataFrame of the kind:

    | Aircraft Designation | (...) | TSFC (cruise) |
    |----------------------|-------|---------------|
    | B707-120             | (...) | 0.6           |
    | A380-800             | (...) | 0.7           |

    and a list of columns to update:

    list_columns = ['TSFC (cruise)']

    and a merge column:

    merge_column = 'Aircraft Designation'

    the function will update the first DataFrame with the values from the second DataFrame:

    | Aircraft Designation | (...) | TSFC (cruise) |
    |----------------------|-------|---------------|
    | B707-120             | (...) | 0.6           |
    | B737-200C            | (...) | 0.5           |
    | A380-800             | (...) | 0.7           |


    Parameters
    ----------
    df_main : pd.DataFrame
        _description_
    df_other : pd.DataFrame
        _description_
    merge_column : str
        _description_
    list_columns : list[str]
        _description_

    See Also
    --------
    - [GitHub Issues/PR at `pint-pandas`](https://github.com/hgrecco/pint-pandas/pull/247)

    Returns
    -------
    pd.DataFrame
        _description_
    """
    df_main_updated = pd.merge(
        left=df_main,
        right=df_other[list_columns + [merge_column]],
        on=merge_column,
        how='left',
        suffixes=('', '_update')
    )

    for col in list_columns:
        df_main_updated[col] = df_main_updated[f"{col}_update"].astype(float).fillna(
            df_main_updated[col].astype(float)
        ).astype(df_main_updated[col].dtype)

    df_main_updated = df_main_updated.drop(columns=[f"{col}_update" for col in list_columns])

    return df_main_updated