# %%
import pandas as pd
import pint
import pint_pandas
ureg = pint.get_application_registry()
from pathlib import Path
from aircraftdetective import ureg
import importlib.resources as pkg_resources
from aircraftdetective import data
from aircraftdetective.data.hyperlinks import (
    PATH_ZENODO_USDOT_T2_FILE,
    PATH_ZENODO_USDOT_ACFT_TYPES_FILE
)

def process_data_usdot_t2(
    path_csv_t2: str = PATH_ZENODO_USDOT_T2_FILE,
    path_csv_aircraft_types: str = PATH_ZENODO_USDOT_ACFT_TYPES_FILE
) -> pd.DataFrame:
    """
    Given a CSV file containing the T2 data from Form 41 Schedule T-100, 
    processes the data and returns a DataFrame with relevant statistics.

    Parameters
    ----------
    path_csv_t2 : str
        Path or URL to the CSV file containing the T2 data from Form 41 Schedule T-100
    path_csv_aircraft_types : str, optional
        Path or URL to the CSV file containing the aircraft types data,
        by default None. If None, the function will use the local file
        `src/aircraftdetective/data/USDOT/L_AIRCRAFT_TYPES.csv`.

    Notes
    -----
    Required data can be downloaded from the [`aircraftdetective` Zenodo repository](https://doi.org/10.5281/zenodo.17255995):

    - [`05-10-2025_T_SCHEDULE_T2.csv`]()
    - [`05-10-2025_L_AIRCRAFT_TYPES.csv`]()

    Terminology
    -----
    - **Form 41**
    
        Form 41 is a report that the U.S. Department of Transportation (DOT) generates
        based on data which large certified air carriers are required to provide.

        The _Financial Report_ part of this form includes balance sheet, cash flow, employment, income statement, fuel cost and consumption,
        aircraft operating expenses, and operating expenses.
        The _Air Carrier Statistics_ part of this form includes data on
        passengers, freight and mail transported.
        It also includes aircraft type, service class, available capacity and seats, and aircraft hours ramp-to-ramp and airborne.

        The reporting requirements of Schedule T-100 of Form 41 are defined in federal law: 

        | Reglation | Scope |
        | --------- | ----- |
        | [14 CFR 291.45](https://www.ecfr.gov/current/title-14/section-291.45) | General |
        | [Appendix A to Subpart E of Part 291, Title 14](https://www.ecfr.gov/current/title-14/part-291/appendix-Appendix%20A%20to%20Subpart%20E%20of%20Part%20291) | US Air Carriers |
        | [Appendix A to Part 217, Title 14](https://www.ecfr.gov/current/title-14/part-217/appendix-Appendix A to Part 217) | Foreign Air Carriers |

    - **Schedule** (in the context of Form 41)

        A schedule is a specific section of the Form 41 that contains
        a particular type of data:

        > "the Air Carrier Financial Reports (Form 41 Financial Data) (...)
        > Each table in this database contains a different type of financial report or “schedule” (...)"

        [Durso (2007) "An Introduction to DOT Form 41 web resources for airline financial analysis"](https://rosap.ntl.bts.gov/view/dot/16264/dot_16264_DS1.pdf)

    - Schedule **T100**

        > "Form 41 Schedule T-100(f) provides flight stage data covering both passenger/cargo
        > and all cargo operations in scheduled and nonscheduled services.
        > The schedule is used to report all flights which serve points in the United States
        > or its territories as defined in this part."

        [Appendix A to Part 217, Title 14](https://www.ecfr.gov/current/title-14/part-217/appendix-Appendix A to Part 217)
    
    - (table) **T2**

        > This table summarizes the T-100 traffic data reported by U.S. air carriers. The quarterly summary (...)

    - (table) **T1**

        > This table summarizes the T-100 traffic data reported by U.S. air carriers. The monthly summary (...)
        

    See Also
    --------
    [`jetfuelburn.statistics.usdot`](https://jetfuelburn.readthedocs.io/en/latest/api/statistics/#jetfuelburn.statistics.usdot)

    References
    ----------
    [US DOT: BTS: Air Carrier Summary Data: T2 (U.S. Air Carrier Traffic And Capacity Statistics by Aircraft Type)](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FIH)

    Warning
    -------
    Form 41 Schedule T-100 Table T2 data can be downloaded from the US Department of Transportation website.
    Unfortunately, there are no permalinks to the data files, and even regular URLs are
    generated dynamically based on some JavaScript magic.
    
    The best way to obtain the correct files is to search by name, according to the following hierarchy:

    ```
    Database Name: Air Carrier Summary Data (Form 41 and 298C Summary Data)
    T2: U.S. Air Carrier TRAFFIC And Capacity Statistics by Aircraft Type
    ```

    On the `T2: (...)` page, click on the "Download" button in the "Data Tools" sidebar (left).
    Now select _at least_ the following "Field Names":

    - `Year`
    - `CarrierGroup`
    - `AircraftConfig`
    - `AircraftType`
    - `AirHours`
    - `AvailSeatMiles`
    - `AirHoursRamp`
    - `RevPaxMiles`
    - `AircraftFuel`
    - `AircraftType`

    The download should be an archive, containing the following file:

    ```
    T_SCHEDULE_T2.csv
    ```

    The aircraft types are abbreviated using a numeric code.
    The full names of the aircraft types is defined in a file `L_AIRCRAFT_TYPE.csv`,
    which can be downloaded on the same "Download" page by opening the "Get Lookup Table"
    link in the `AircraftType` row.

    Returns
    -------
    pd.DataFrame
        A [`pint-pandas`](https://pint-pandas.readthedocs.io/en/latest/) DataFrame with support for physical units, containing the processed T2 data.  

        | Aircraft Designation (US DOT Schedule T2)  | Fuel/Available Seat Distance | Fuel/Revenue Seat Distance | Fuel Flow          | Airborne Efficiency | SLF    |
        |--------------------------------------------|------------------------------|----------------------------|--------------------|---------------------|--------|
        | string                                     | pint[gallons/mile]           | pint[gallons/mile]         | pint[gallons/hour] | pint[]              | pint[] |
        
    Example
    -------
    ```python
    >>> from aircraftdetective.processing.statistics import process_data_usdot_t2
    >>> df_t2 = process_data_usdot_t2(
    ...     path_csv_t2='path/to/T_SCHEDULE_T2.csv',
    ...     path_csv_aircraft_types='path/to/L_AIRCRAFT_TYPE.csv'
    ... )
    >>> print(df_t2.head())
    ```
    """

    if path_csv_aircraft_types is None:
        with pkg_resources.path('aircraftdetective.data.USDOT', 'L_AIRCRAFT_TYPES.csv') as file_path:
            path_csv_aircraft_types = file_path

    df_t2 = pd.read_csv(
        filepath_or_buffer=path_csv_t2,
        header=0,
        index_col=None,
        sep=',',
    )
    
    df_aircraft_types = pd.read_csv(
        path_csv_aircraft_types,
        sep='|', # separator not present in csv file, to ensure entire line is read into a single column
        header=0,
        encoding='utf-8'
    )
    single_col_name = df_aircraft_types.columns[0]
    new_columns = single_col_name.split(',', 1) # split only on the first comma
    df_aircraft_types[new_columns] = df_aircraft_types[single_col_name].str.split(',', n=1, expand=True)
    df_aircraft_types = df_aircraft_types[new_columns]
    df_aircraft_types = (df_aircraft_types
        .rename(
            columns={
                "Code": "AIRCRAFT_TYPE",
                "Description": "Aircraft Designation (US DOT Schedule T2)"
            }
        )
        .astype({
            "AIRCRAFT_TYPE": "int64",
            "Aircraft Designation (US DOT Schedule T2)": "string"
        })
    )

    df_t2 = pd.merge(
        left=df_t2,
        right=df_aircraft_types,
        on='AIRCRAFT_TYPE',
        how='left',
    )

    # SETTING COLUMN NAMES AND UNITS

    """
    Column names in the T2 dataset sometimes have numbers appended.
    In order to unify the column names, these numbers are removed.

    For example, as of 11-2024, the dictionary generated below will look like:

    dict_columns_for_renaming = {
        'AVL_SEAT_MILES_320': 'AVL_SEAT_MILES',
        'REV_PAX_MILES_140': 'REV_PAX_MILES',
        'AIRCRAFT_FUELS_921': 'AIRCRAFT_FUELS',
        'CARRIER_GROUP': 'CARRIER_GROUP',
        'AIRCRAFT_CONFIG': 'AIRCRAFT_CONFIG',
        'AIRCRAFT_TYPE': 'AIRCRAFT_TYPE',
        'HOURS_AIRBORNE_650': 'HOURS_AIRBORNE',
        'ACRFT_HRS_RAMPTORAMP_630': 'ACRFT_HRS_RAMPTORAMP'
    }
    """

    dict_columns_and_units = {
        'Year': int,
        'AVL_SEAT_MILES': 'pint[miles]',
        'REV_PAX_MILES': 'pint[miles]',
        'AIRCRAFT_FUELS': 'pint[gallons]',
        'CARRIER_GROUP': 'pint[]',
        'AIRCRAFT_CONFIG': 'pint[]',
        'AIRCRAFT_TYPE': 'pint[]',
        'HOURS_AIRBORNE': 'pint[hours]',
        'ACRFT_HRS_RAMPTORAMP': 'pint[hours]',
    }
    # 1. Create an empty dictionary to populate safely
    dict_columns_for_renaming = {}

    # 2. Loop through the desired column names and check for matches
    for column_name in dict_columns_and_units.keys():
        # Find all columns in the DataFrame that contain the base name
        matching_cols = df_t2.filter(like=column_name).columns

        # 3. IMPORTANT: Only proceed if a match was actually found
        if not matching_cols.empty:
            # Get the full original column name (e.g., 'HOURS_AIRBORNE_650')
            original_col = matching_cols[0]
            # Add the entry to our renaming dictionary
            dict_columns_for_renaming[original_col] = column_name

    # 4. Rename the columns that were found
    df_t2 = df_t2.rename(columns=dict_columns_for_renaming)

    # 5. Filter the dtype dictionary to only include columns that now exist in the DataFrame
    final_dtype_map = {k: v for k, v in dict_columns_and_units.items() if k in df_t2.columns}

    # 6. Safely apply the dtypes
    df_t2 = df_t2.astype(final_dtype_map)
    
    # DATA FILTERING
    
    df_t2 = df_t2.loc[df_t2['CARRIER_GROUP'] == 3] # major carriers only
    df_t2 = df_t2.loc[df_t2['AIRCRAFT_CONFIG'] == 1] # passenger aircraft only
    df_t2 = df_t2.drop(columns=['CARRIER_GROUP', 'AIRCRAFT_CONFIG'])

    list_numeric_columns = [
        'AVL_SEAT_MILES',
        'REV_PAX_MILES',
        'AIRCRAFT_FUELS',
        'HOURS_AIRBORNE',
    ]
    df_t2[list_numeric_columns] = df_t2[list_numeric_columns].replace(
        to_replace=0,
        value=pd.NA
    )

    # COLUMN RENAMING

    df_t2 = df_t2.rename(
        columns={
            'YEAR': 'Year',
        }
    )

    # CUSTOM COLUMN CALCULATIONS

    df_t2['Fuel/Available Seat Distance'] = df_t2['AIRCRAFT_FUELS']/df_t2['AVL_SEAT_MILES']
    df_t2['Fuel/Revenue Seat Distance'] = df_t2['AIRCRAFT_FUELS']/df_t2['REV_PAX_MILES']
    df_t2['Fuel Flow'] = df_t2['AIRCRAFT_FUELS']/df_t2['HOURS_AIRBORNE']
    df_t2['Airborne Efficiency'] = df_t2['HOURS_AIRBORNE']/df_t2['ACRFT_HRS_RAMPTORAMP']
    df_t2['SLF']= df_t2['REV_PAX_MILES']/df_t2['AVL_SEAT_MILES']

    # SANITY CHECKS

    df_t2 = df_t2.loc[df_t2['REV_PAX_MILES'] <= df_t2['AVL_SEAT_MILES']]
    df_t2 = df_t2.loc[df_t2['HOURS_AIRBORNE'] <= df_t2['ACRFT_HRS_RAMPTORAMP']]

    # RETURN

    list_return_columns = [
        'Year',
        'Aircraft Designation (US DOT Schedule T2)',
        'Fuel/Available Seat Distance',
        'Fuel/Revenue Seat Distance',
        'Fuel Flow',
        'Airborne Efficiency',
        'SLF',
    ]

    df_t2 = df_t2[list_return_columns]
    df_t2 = df_t2.reset_index(drop=True)
    return df_t2