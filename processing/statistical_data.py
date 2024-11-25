# %%

from pathlib import Path
import pandas as pd
import pint_pandas


def process_data_usdot_t2(
    path_csv_t2: Path,
    path_csv_aircraft_types: Path,
) -> pd.DataFrame:
    """_summary_

    _extended_summary_

    Notes
    -----
    Required data must be downloaded from the US Department of Transport:
    - ["AircraftType": `L_AIRCRAFT_TYPE.csv`]()
    - ["Data Tools: Download": `T_SCHEDULE_T2.csv`]()

    See Also
    --------
    Additional information can be found at:
    - [US DOT: BTS: Air Carrier Summary Data (Form 41 and 298C Summary Data)](https://www.transtats.bts.gov/Tables.asp?QO_VQ=EGD&QO)
    - [US DOT: BTS: Air Carrier Summary Data: T2 (U.S. Air Carrier Traffic And Capacity Statistics by Aircraft Type)](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FIH)
    """

    df_t2 = pd.read_csv(
        filepath_or_buffer=path_csv_t2,
        header=0,
        index_col=None,
        sep=',',
    )
    df_aircraft_types = pd.read_csv(
        filepath_or_buffer=path_csv_aircraft_types,
        header=0,
        index_col=None,
        sep=',',
        names=['AIRCRAFT_TYPE', 'Aircraft Designation'],
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
        'AVL_SEAT_MILES': 'pint[miles]',
        'REV_PAX_MILES': 'pint[miles]',
        'AIRCRAFT_FUELS': 'pint[gallons]',
        'CARRIER_GROUP': 'pint[]',
        'AIRCRAFT_CONFIG': 'pint[]',
        'AIRCRAFT_TYPE': 'pint[]',
        'HOURS_AIRBORNE': 'pint[hours]',
        'ACRFT_HRS_RAMPTORAMP': 'pint[hours]',
    }
    dict_columns_for_renaming = {df_t2.filter(like=column_name).columns[0]: column_name for column_name in dict_columns_and_units.keys()}
    df_t2 = df_t2.rename(columns=dict_columns_for_renaming)
    df_t2 = df_t2.astype(dict_columns_and_units)
    
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
        'Aircraft Designation',
        'Fuel/Available Seat Distance',
        'Fuel/Revenue Seat Distance',
        'Fuel Flow',
        'Airborne Efficiency',
        'SLF',
    ]

    df_t2 = df_t2[list_return_columns]
    df_t2 = df_t2.reset_index(drop=True)
    return df_t2