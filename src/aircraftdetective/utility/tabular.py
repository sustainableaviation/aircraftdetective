# %%
import pint_pandas
import pint
ureg = pint.get_application_registry()
from pathlib import Path
import pandas as pd
from pint import DimensionalityError
from pint_pandas.pint_array import is_pint_type


def _validate_dataframe_columns_with_units(
        df: pd.DataFrame,
        required_schema: dict[str, str]
) -> None:
    r"""
    Validates the presence and dimensions of [`pint-pandas`](https://pint-pandas.readthedocs.io/en/latest/) 
    DataFrame columns.

    Parameters
    ----------
    df : pd.DataFrame
        The pint-pandas DataFrame to validate.
    required_schema : dict[str, str]
        A dictionary where keys are the required column names (str) and
        values are their expected dimension strings. Of the form  
         
        ```
        {
            "foo": "[length]",
            "bar": "[mass]/[time]",
            ...
        }
        ```

    Notes
    -----
    Valid `pint` dimension strings can be listed using this syntax:

    ```pyodide install='aircraftdetective'
    import pint
    ureg = pint.UnitRegistry()
    sorted(list(ureg._dimensions.keys()))
    ```

    See Also
    --------
    [`pint` Documentation: Checking Dimensionality](https://pint.readthedocs.io/en/stable/advanced/wrapping.html#checking-dimensionality)

    Returns
    -------
    None
        This function does not return anything if validation is successful.

    Raises
    ------
    ValueError
        If columns are missing or if any column has incorrect dimensions.
    """
    missing_columns = [col for col in required_schema if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")

    for col, dim_str in required_schema.items():
        if not is_pint_type(df[col]):
             raise TypeError(f"Column '{col}' is not a pint-dtype Series and cannot be validated.")
        
        if df[col].pint.check(dim_str) is False:
            raise ValueError(
                f"Column '{col}' has incorrect units. "
                f"Expected dimensionality of '{dim_str}', but got '{pint.Unit(df[col].pint.dimensionality)}'."
            )


def _rename_columns_and_set_units(
    df: pd.DataFrame,
    return_only_renamed_columns: bool,
    column_names_and_units: list[tuple[str, str, str]],
) -> pd.DataFrame:
    r"""
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


def update_column_data(
    df_main: pd.DataFrame,
    df_other: pd.DataFrame,
    merge_column: str,
    list_columns: list[str],
) -> pd.DataFrame:
    r"""
    Given two DataFrames, updates the values in the first DataFrame 
    with the values from the second DataFrame for the specified columns, 
    based on a common merge column.

    Given a first DataFrame of the kind:

    | Aircraft Designation | (...) | TSFC (cruise) |
    |----------------------|-------|---------------|
    | B707-120             | (...) | NaN           |
    | B737-200C            | (...) | 0.5           |
    | A380-800             | (...) | NaN           |

    and a second DataFrame of the kind:

    | Aircraft Designation | (...) | TSFC (cruise) |
    |----------------------|-------|---------------|
    | B707-120             | (...) | 0.6           |
    | A380-800             | (...) | 0.7           |

    for the merge column `Aircraft Designation`, 
    and the list of columns to update `['TSFC (cruise)']`,
    the function will update the first DataFrame with the values from the second DataFrame:

    | Aircraft Designation | (...) | TSFC (cruise) |
    |----------------------|-------|---------------|
    | B707-120             | (...) | **0.6**       |
    | B737-200C            | (...) | 0.5           |
    | A380-800             | (...) | **0.7**       |

    Parameters
    ----------
    df_main : pd.DataFrame
        Main DataFrame to be updated.
    df_other : pd.DataFrame
        Other DataFrame to update from.
    merge_column : str
        Column name to merge on.
    list_columns : list[str]
        List of column names to update. These columns must exist in both DataFrames.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame.

    Raises
    ------
    KeyError
        If the `merge_column` or any of the `list_columns` do not exist in either DataFrame.
    """
    df_main = df_main.copy()
    df_other = df_other.copy()

    if merge_column not in df_main.columns:
        raise KeyError(f"'{merge_column}' not in main DataFrame columns")
    if merge_column not in df_other.columns:
        raise KeyError(f"'{merge_column}' not in other DataFrame columns")
    for col in list_columns:
        if col not in df_main.columns:
            raise KeyError(f"'{col}' not in main DataFrame columns")
        if col not in df_other.columns:
            raise KeyError(f"'{col}' not in other DataFrame columns")

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


def left_merge_wildcard(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_on: str,
    right_on: str,
) -> pd.DataFrame:
    """
    Given two DataFrames, merges them on columns that may contain wildcard characters.  
    Rows are aggregated by taking the mean of all matching rows.

    For a left dataframe of the form:

    | Engine Designation | ... |
    |--------------------|-----|
    | CFM56-5*           | ... |
    | V2500-A1           | ... |

    and a right dataframe of the form:

    | Engine Designation | Thrust (kN) | Non-Numeric Column |
    |--------------------|-------------|--------------------|
    | CFM56-5A1          | 60          | Foo                |
    | CFM56-5A2          | 62          | Bar                |
    | CFM56-5A3          | 64          | Baz                |
    | V2500-A1           | 100         | Qux                |

    the function returns a merged dataframe of the form:

    | Engine Designation | ... | Thrust (kN)  | Non-Numeric Column |
    |--------------------|-----|--------------|--------------------|
    | CFM56-5*           | ... | (60+62+64)/3 | Foo                |
    | V2500-A1           | ... | 100          | Qux                |

    Notes
    -----
    Wildcard characters are supported in the left DataFrame only.

    See Also
    --------
    [`pandas.DataFrame.merge`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html)

    Parameters
    ----------
    df_left : pd.DataFrame
        Left DataFrame to merge.
    df_right : pd.DataFrame
        Right DataFrame to merge.
    left_on : str
        Column name in the left DataFrame to merge on.
    right_on : str
        Column name in the right DataFrame to merge on.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    if left_on not in df_left.columns:
        raise KeyError(f"'{left_on}' not in left DataFrame columns")
    if right_on not in df_right.columns:
        raise KeyError(f"'{right_on}' not in right DataFrame columns")

    df_l = df_left.copy()
    df_r = df_right.copy()

    left_keys = df_l[left_on].drop_duplicates().tolist()
    right_keys = df_r[right_on].drop_duplicates().tolist()

    # Pre-process left keys for efficient lookup
    wildcard_patterns = {
        key: key.split('*')[0]
        for key in left_keys
        if isinstance(key, str) and '*' in key
    }
    # Use a set for fast O(1) lookups of exact keys
    exact_keys = {key for key in left_keys if key not in wildcard_patterns}

    # Build a definitive map from each right_key to its single BEST left_key match
    key_map = {}
    for rk in right_keys:
        # Priority 1: An exact match is always the best match.
        if rk in exact_keys:
            key_map[rk] = rk
            continue

        # Priority 2: If no exact match, find the best wildcard match.
        # This only applies if the right key is a string.
        if not isinstance(rk, str):
            continue

        matching_wildcards = [
            lk for lk, pattern in wildcard_patterns.items() if rk.startswith(pattern)
        ]

        # If any wildcard matches were found, the best is the most specific (longest).
        if matching_wildcards:
            best_match = max(matching_wildcards, key=len)
            key_map[rk] = best_match

    # Use the unambiguous map to create the new join key in the right DataFrame.
    # Keys in df_r that have no match in the map will be assigned NaN.
    df_r['key_wildcard'] = df_r[right_on].map(key_map)

    # The groupby will automatically drop rows where 'key_wildcard' is NaN,
    # satisfying the "silent discarding is ok" requirement.
    numeric_cols = df_r.select_dtypes(include='number').columns
    object_cols = df_r.select_dtypes(exclude='number').columns.drop([right_on, 'key_wildcard'], errors='ignore')

    agg_rules = {col: 'mean' for col in numeric_cols}
    agg_rules.update({col: 'first' for col in object_cols})

    if 'key_wildcard' not in df_r or df_r['key_wildcard'].isnull().all():
        # Handle case where no matches were found at all
        df_r_agg = pd.DataFrame(columns=df_r.columns.drop(right_on)).set_index('key_wildcard' if 'key_wildcard' in df_r.columns else [])
    elif not agg_rules:
        # Handle case where there are no columns to aggregate
        unique_matched_keys = df_r['key_wildcard'].dropna().unique()
        df_r_agg = pd.DataFrame(index=unique_matched_keys)
        df_r_agg.index.name = 'key_wildcard'
    else:
        df_r_agg = df_r.groupby('key_wildcard').agg(agg_rules)

    return pd.merge(
        left=df_l,
        right=df_r_agg,
        how='left',
        left_on=left_on,
        right_index=True,
    )