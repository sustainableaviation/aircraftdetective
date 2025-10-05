# %%
import json
import pandas as pd

from pathlib import Path

def read_aircraft_database(
    path_json_aircraft_database: Path,
    dict_properties: dict,
    dict_manufacturers: dict
) -> pd.DataFrame:
    """
    Reads a JSON backup of the the entire [aircraft-database.com](https://web.archive.org/web/20231201220700/https://aircraft-database.com/)
    (discontinued as of 01-2024) and returns a DataFrame with all aircraft models and their properties.
    
    Since aircraft can have multiple engines and `propertyValues`,
    the function will explode these columns to have one row per engine and property.
    For example, the following JSON object:

    ```
    {
    "id":"1edbc3ab-ed02-6b78-b000-951ebcf1a8fa",
    "aircraftFamily":"airplane",
    "engineCount":1,
    "engineFamily":"piston",
    "engineModels":["1ed869d8-bac4-67a4-a133-df00b6f38f52","1ed869d8-e2ad-6d92-a133-799194a3de38"],
    "iataCode":null,
    "icaoCode":null,
    "manufacturer":"1edbc3a6-8e32-6c24-b000-9ff27b37bf11",
    "name":"1",
    "propertyValues":[
        {"property":"1ec96f93-22b5-66f0-9933-45bd403e4df0","value":839},
        {"property":"1ec96f94-5471-66de-9933-a93eae676780","value":839},
        {"property":"1ec96f9c-4371-6098-9933-2d909f93f3cd","value":91}
    ],
    "tags":[],
    "url":"https:\/\/aircraft-database.com\/database\/aircraft-types\/1"
    },
    ```

    will be transformed into:

    | id                                   | (...) | engineModels                         | propertyValues                       |
    |--------------------------------------|-------|--------------------------------------|--------------------------------------|
    | 1edbc3ab-ed02-6b78-b000-951ebcf1a8fa | (...) | 1ed869d8-bac4-67a4-a133-df00b6f38f52 | 1ec96f93-22b5-66f0-9933-45bd403e4df0 |
    | 1edbc3ab-ed02-6b78-b000-951ebcf1a8fa | (...) | 1ed869d8-bac4-67a4-a133-df00b6f38f52 | 1ec96f94-5471-66de-9933-a93eae676780 |
    | 1edbc3ab-ed02-6b78-b000-951ebcf1a8fa | (...) | 1ed869d8-bac4-67a4-a133-df00b6f38f52 | 1ec96f9c-4371-6098-9933-2d909f93f3cd |
    | 1edbc3ab-ed02-6b78-b000-951ebcf1a8fa | (...) | 1ed869d8-e2ad-6d92-a133-799194a3de38 | 1ec96f93-22b5-66f0-9933-45bd403e4df0 |
    | 1edbc3ab-ed02-6b78-b000-951ebcf1a8fa | (...) | 1ed869d8-e2ad-6d92-a133-799194a3de38 | 1ec96f94-5471-66de-9933-a93eae676780 |
    | 1edbc3ab-ed02-6b78-b000-951ebcf1a8fa | (...) | 1ed869d8-e2ad-6d92-a133-799194a3de38 | 1ec96f9c-4371-6098-9933-2d909f93f3cd |

    See Also
    --------
    - [aircraft-database.com (archived)](https://web.archive.org/web/20231201220700/https://aircraft-database.com/)

    Parameters
    ----------
    path_json_aircraft_models : Path
        Path to the JSON file containing the aircraft models and their properties.

    Returns
    -------
    pd.DataFrame
        _description_
    """

    df = pd.read_json(path_json_aircraft_database)
    df = df.loc[df['aircraftFamily'].isin(['airplane'])] # remove helicopters, amphibious, etc.
    df = df[df['tags'].str.len() == 0] # remove entries with tags (e.g. ['military'])

    df['manufacturer'] = df['manufacturer'].map(dict_manufacturers)

    df = df.explode('engineModels').reset_index(drop=True)
    df = df.explode('propertyValues').reset_index(drop=True)

    df_properties = pd.json_normalize(df['propertyValues'])
    df_properties['property'] = df_properties['property'].map(dict_properties)

    df_properties_pivot = df_properties.pivot(columns='property', values='value')

    df = pd.concat(
        objs=[
            df,
            df_properties_pivot
        ],
        axis=1
    )

    dict_column_names = {
        'id': '_id_aircraft',
        'engineModels': '_id_engine',
        'manufacturer': 'Aircraft Manufacturer',
        'name': 'Aircraft Designation',
        'engineCount': 'Engine Count',
        'Fuel Capacity [L]': 'Fuel Capacity [l]',
        'Mlw [kg]': 'MLW [kg]',
        'Mtow [kg]': 'MTOW [kg]',
        'Mtw [kg]': 'MTW [kg]',
        'Mzfw [kg]': 'MZFW [kg]',
        'Mmo': 'MMO',
        'Maximum Operating Altitude [ft]': 'Maximum Operating Altitude [ft]',
        'Oew [kg]': 'OEW [kg]',
        'Wing Area [m2]': 'Wing Area [m²]',
        'Wingspan (Canard) [m]': 'Wingspan (Canard) [m]',
        'Wingspan (Winglets) [m]': 'Wingspan (winglets) [m]',
        'Wingspan [m]': 'Wingspan [m]',
        'Height [m]': 'Height [m]',
    }

    df = df.rename(columns=dict_column_names)
    df = df[dict_column_names.values()]

    df_grouped = df.groupby(
        by=[
            'Aircraft Designation',
            '_id_engine',
        ],
        as_index=False,
    ).first()
    
    return df_grouped


def read_engine_database(
    path_json_engine_database: Path,
    dict_properties: dict,
    dict_manufacturers: dict,
) -> pd.DataFrame:
    """
    Given the path to a JSON file containing the backup of the [aircraft-database.com](https://web.archive.org/web/20231201220700/https://aircraft-database.com/)
    (discontinued as of 01-2024) engine database, a dictionary of property IDs and their names, and a dictionary of manufacturer IDs and their names,
    returns a DataFrame with the engine models and their properties.

    The `propertyValues` column contains a list of dictionaries with the properties and their values.
    For example, one entry in the `propertyValues` column might look like this:

    | name | propertyValues |
    |------|----------------|
    | foo  | [{'property': 'alpha', 'value': 107.95}, {'property': 'beta', 'value': 'Air'}] |
    | bar  | [{'property': 'alpha', 'value': 12} |

    | name | alpha | beta |
    |------|-------|------|
    | foo  | 107.95| Air  |
    | bar  | 12    | NaN  |

    [
        {'property': '1ecd84bb-a0fc-6e28-8b2d-99154ce73d02', 'value': 107.95},
        {'property': '1ecd84bf-2ea5-6c86-8b2d-0b9aa407e174', 'value': 'Air'}
    ]
    Exploding the `propertyValues` column will create one row per property and value:

    | name    | (...) | propertyValues                                                        |
    |---------|-------|-----------------------------------------------------------------------|
    | 110 7DF | (...) | {'property': '1ecd84bb-a0fc-6e28-8b2d-99154ce73d02', 'value': 107.95} |
    | 110 7DF | (...) | {'property': '1ecd84bf-2ea5-6c86-8b2d-0b9aa407e174', 'value': 'Air'}  |

    Using `pd.json_normalize` will create two columns `property` and `value`:

    | name    | (...) | property                             | value  |
    |---------|-------|--------------------------------------|--------|
    | 110 7DF | (...) | 1ecd84bb-a0fc-6e28-8b2d-99154ce73d02 | 107.95 |
    | 110 7DF | (...) | 1ecd84bf-2ea5-6c86-8b2d-0b9aa407e174 | Air    |

    Using the dictionary of property names, we can map the `property` column to the actual property name:

    | name    | (...) | property       | value  |
    |---------|-------|----------------|--------|
    | 110 7DF | (...) | Bore [mm]      | 107.95 |
    | 110 7DF | (...) | Cooling System | Air    |

    Finally, we can pivot the `property` column to have one column per property:

    | name    | (...) | Bore [mm] | Cooling System |
    |---------|-------|-----------|----------------|
    | 110 7DF | (...) | 107.95    | NaN            |
    | 110 7DF | (...) | NaN       | Air            |

    Which we can group by engine designation (=name) and aggregate the values:

    | name    | (...) | Bore [mm] | Cooling System |
    |---------|-------|-----------|----------------|
    | 110 7DF | (...) | 107.95    | Air            |

    Parameters
    ----------
    path_json_engine_models : Path
        Path to the JSON file containing the engine models and their properties.
    dict_properties : dict
        Dictionary mapping the property IDs to their names.
    dict_manufacturers : dict
        Dictionary mapping the manufacturer IDs to their names.

    Returns
    -------
    pd.DataFrame
        DataFrame with the engine models and their properties.
    """

    df = pd.read_json(path_json_engine_database)
    df = df.loc[df['engineFamily'].isin(['turbofan', 'turbojet', 'turboprop'])]

    df['manufacturer'] = df['manufacturer'].map(dict_manufacturers)

    df = df.explode('propertyValues').reset_index(drop=True)

    df_properties = pd.json_normalize(df['propertyValues'])
    df_properties['property'] = df_properties['property'].map(dict_properties)

    df_properties_pivot = df_properties.pivot(columns='property', values='value')

    df = pd.concat(
        objs=[
            df,
            df_properties_pivot
        ],
        axis=1
    )

    dict_column_names = {
        'id': '_id_engine',
        'name': 'Engine Designation',
        'engineFamily': 'Engine Family',
        'manufacturer': 'Engine Manufacturer',
        'Bypass Ratio': 'Bypass Ratio',
        'Overall Pressure Ratio': 'Overall Pressure Ratio',
        'Dry Weight [kg]': 'Dry Weight [kg]',
        'Fan Diameter [m]': 'Fan Diameter [m]',
        'Max. Continuous Power [kW]': 'Max. Continuous Power [kW]',
        'Max. Continuous Thrust [kN]': 'Max. Continuous Thrust [kN]',
    }

    df = df.rename(columns=dict_column_names)
    df = df[dict_column_names.values()]

    df_grouped = df.groupby(
        by='Engine Designation',
        as_index=False,
    ).first()

    return df_grouped



def read_properties_database(path_json_properties: Path) -> dict:
    """
    Reads a JSON backup of the the [aircraft-database.com](https://web.archive.org/web/20231201220700/https://aircraft-database.com/)
    (discontinued as of 01-2024) table of `properties` and their values and returns a dictionary with the properties and their values.

    For instance, the following JSON object:

    ```
    {
        '1ec882d7-fc13-611c-b99a-5724c0f1cad1': 'Afterburner [None]',
        '1ed95aff-38d7-6552-9ae8-b35cc598320e': 'Battery Capacity [kilowatt-hour]',
        '1ed95b07-4149-69c6-9ae8-c3b0ecc17a8d': 'Battery Technology [None]',
        (...)
    }
    ```

    Parameters
    ----------
    path_json_properties : Path
        Path to the JSON file containing the properties and their values.

    Returns
    -------
    pd.DataFrame
        _description_
    """

    dict_units = {
        None: 'dimensionless',
        'kilowatt-hour': 'kWh',
        'volt': 'V',
        'millimetre': 'mm',
        'cubic-centimetre': 'cm³',
        'kilogram': 'kg',
        'metre': 'm',
        'litre': 'L',
        'horsepower': 'hp',
        'kilowatt': 'kW',
        'kilonewton': 'kN',
        'decanewton-metre': 'daN*m',
        'foot': 'ft',
        'knot': 'kn',
        'cubic-metre': 'm³',
        'square-metre': 'm²',
    }

    df_properties = pd.read_json(path_json_properties)

    # capitalize words in name column
    df_properties['name'] = df_properties['name'].apply(lambda x:x.title()) 
    # replace unit names with abbreviations
    df_properties['unit'] = df_properties['unit'].map(dict_units)
    
    df_properties['value'] = df_properties.apply(
        lambda row: f"{row['name']} [{row['unit']}]"
        if row['unit'] != 'dimensionless'
        else row['name'],
        axis=1
    )
    
    dict_properties = df_properties.set_index('id')['value'].to_dict()

    return dict_properties


def read_manufacturers_database(path_json_manufacturers: Path) -> pd.DataFrame:
    """_summary_

    For instanct, the following JSON object:

    ```
    {
        '1edc26ff-b373-6418-bf55-35eb9f006d3f': 'AAMSA',
        '1ed62800-0b15-6f68-b5ec-05fb44d7d394': 'ABC Motors',
        '1ed62836-1733-6ee0-b5ec-8352ea69874f': 'ADC',
        (...)
    }
    ```

    Parameters
    ----------
    path_json_manufacturers : Path
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    df_manufacturers = pd.read_json(path_json_manufacturers)
    df_manufacturers = df_manufacturers[['id', 'name']]
    dict_manufacturers = df_manufacturers.set_index('id')['name'].to_dict()

    return dict_manufacturers


path_json_manufacturers = Path('manufacturers.json')
dict_manufacturers = read_manufacturers_database(path_json_manufacturers)

path_json_properties = Path('properties.json')
dict_properties = read_properties_database(path_json_properties)

path_json_engine_models = Path('engine-models.json')
df_engines = read_engine_database(path_json_engine_models, dict_properties, dict_manufacturers)

path_json_aircraft_models = Path('aircraft-types.json')
df_aircraft = read_aircraft_database(path_json_aircraft_models, dict_properties, dict_manufacturers)
# %%
