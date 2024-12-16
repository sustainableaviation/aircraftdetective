# %%

from pathlib import Path
import pandas as pd
import pint_pandas


def rename_columns_and_set_units(
    df: pd.DataFrame,
    column_names_and_units: list[tuple[str, str, pint_pandas.pint_array.PintType]]
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

    subset = [name_old for name_old, _, _ in column_names_and_units]
    df = df[subset]

    for name_old, name_new, dtype in column_names_and_units:
        df = df.rename(columns={name_old: name_new})
        df[name_new] = df[name_new].astype(dtype)

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