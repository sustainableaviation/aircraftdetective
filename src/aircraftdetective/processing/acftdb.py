# %%
from pathlib import Path
import pandas as pd
import pint
ureg = pint.get_application_registry()
from aircraftdetective.utility.tabular import _rename_columns_and_set_units

from aircraftdetective.data.hyperlinks import (
    PATH_ZENODO_AIRCRAFT_DATABASE_AIRCRAFT_TYPES_FILE,
    PATH_ZENODO_AIRCRAFT_DATABASE_ENGINE_MODELS_FILE,
    PATH_ZENODO_AIRCRAFT_DATABASE_MANUFACTURERS_FILE,
    PATH_ZENODO_AIRCRAFT_DATABASE_PROPERTIES_FILE
)


def _read_properties_database(
    path_json_properties: str = PATH_ZENODO_AIRCRAFT_DATABASE_PROPERTIES_FILE
) -> dict:
    """
    Reads a JSON backup of the the [aircraft-database.com](https://web.archive.org/web/20231201220700/https://aircraft-database.com/)
    (discontinued as of 01-2024) table of `properties` and their values and returns it as a dictionary.

    The JSON object and returned dictionary are of the form:

    ```
    {
        '1ec882d7-fc13-611c-b99a-5724c0f1cad1': 'Afterburner [None]',
        '1ed95aff-38d7-6552-9ae8-b35cc598320e': 'Battery Capacity [kilowatt-hour]',
        '1ed95b07-4149-69c6-9ae8-c3b0ecc17a8d': 'Battery Technology [None]',
        (...)
    }
    ```

    Notes
    -----
    With no parameters passed, the function will download the relevant JSON file 
    from the [aircraft-database.com Backup Zenodo repository](https://doi.org/10.5281/zenodo.14382244).

    Parameters
    ----------
    path_json_properties : str
        Path to the JSON file containing the properties and their values.

    Returns
    -------
    dict
        Dictionary with the `aircraft-database.com` property IDs as keys and the property names with units as values.
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


def _read_manufacturers_database(
    path_json_manufacturers: str = PATH_ZENODO_AIRCRAFT_DATABASE_MANUFACTURERS_FILE
) -> dict:
    """
    Reads a JSON backup of the the [aircraft-database.com](https://web.archive.org/web/20231201220700/https://aircraft-database.com/)
    (discontinued as of 01-2024) table of `manufacturers` and their values and returns it as a dictionary.

    The JSON object and returned dictionary are of the form:

    ```
    {
        '1edc26ff-b373-6418-bf55-35eb9f006d3f': 'AAMSA',
        '1ed62800-0b15-6f68-b5ec-05fb44d7d394': 'ABC Motors',
        '1ed62836-1733-6ee0-b5ec-8352ea69874f': 'ADC',
        (...)
    }
    ```

    Notes
    -----
    With no parameters passed, the function will download the relevant JSON file 
    from the [aircraft-database.com Backup Zenodo repository](https://doi.org/10.5281/zenodo.14382244).

    Parameters
    ----------
    path_json_manufacturers : str
        Path or URL of the JSON file containing the manufacturers and their values.

    Returns
    -------
    dict
        Dictionary with the `aircraft-database.com` manufacturer IDs as keys and the manufacturer names as values.
    """
    df_manufacturers = pd.read_json(path_json_manufacturers)
    df_manufacturers = df_manufacturers[['id', 'name']]
    dict_manufacturers = df_manufacturers.set_index('id')['name'].to_dict()

    return dict_manufacturers


def _read_engine_database(
    path_json_engine_database: str = PATH_ZENODO_AIRCRAFT_DATABASE_ENGINE_MODELS_FILE,
    dict_properties: dict | None = None,
    dict_manufacturers: dict | None = None,
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

    Notes
    -----
    With no parameters passed, the function will download the relevant JSON file 
    from the [aircraft-database.com Backup Zenodo repository](https://doi.org/10.5281/zenodo.14382244).

    Parameters
    ----------
    path_json_engine_database : str
        Path to the JSON file containing the engine models and their properties.
    dict_properties : dict, optional
        Dictionary mapping the property IDs to their names. If None, loaded from default path.
    dict_manufacturers : dict, optional
        Dictionary mapping the manufacturer IDs to their names. If None, loaded from default path.

    Returns
    -------
    pd.DataFrame
        [`pint-pandas`](https://pint-pandas.readthedocs.io/en/latest/) DataFrame with the engine models and their properties.  
    """
    if dict_properties is None:
        dict_properties = _read_properties_database()
    
    if dict_manufacturers is None:
        dict_manufacturers = _read_manufacturers_database()

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

    df = _rename_columns_and_set_units(
        df=df,
        column_names_and_units=[
            ('id', '_id_engine', 'str'), # aircraft-database.com internal id, not needed here
            ('name', 'Engine Designation', 'str'),
            ('engineFamily', 'Engine Family', 'str'),
            ('manufacturer', 'Engine Manufacturer', 'str'),
            # ('Bypass Ratio', 'Bypass Ratio', 'pint[dimensionless]'), # bypass ration of EASA/ICAO Engine Database seems more reliable
            ('Overall Pressure Ratio', 'Overall Pressure Ratio', 'pint[dimensionless]'),
            ('Dry Weight [kg]', 'Dry Weight', 'pint[kg]'),
            ('Fan Diameter [m]', 'Fan Diameter', 'pint[m]'),
            ('Max. Continuous Thrust [kN]', 'Max. Continuous Thrust', 'pint[kN]'),
        ],
        return_only_renamed_columns=True,
    )

    df_grouped = df.groupby(
        by='Engine Designation',
        as_index=False,
    ).first()

    return df_grouped


def _read_aircraft_database(
    path_json_aircraft_database: str = PATH_ZENODO_AIRCRAFT_DATABASE_AIRCRAFT_TYPES_FILE,
    dict_properties: dict | None = None,
    dict_manufacturers: dict | None = None,
) -> pd.DataFrame:
    r"""
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

    Notes
    -----
    With no parameters passed, the function will download the relevant JSON file 
    from the [aircraft-database.com Backup Zenodo repository](https://doi.org/10.5281/zenodo.14382244).

    Parameters
    ----------
    path_json_aircraft_database : str
        Path to the JSON file containing the aircraft models and their properties.
    dict_properties : dict, optional
        Dictionary mapping the property IDs to their names. If None, loaded from default path.
    dict_manufacturers : dict, optional
        Dictionary mapping the manufacturer IDs to their names. If None, loaded from default path.

    Returns
    -------
    pd.DataFrame
        [`pint-pandas`](https://pint-pandas.readthedocs.io/en/latest/) DataFrame with the aircraft models and their properties.
    """
    if dict_manufacturers is None:
        dict_manufacturers = _read_manufacturers_database()
    
    if dict_properties is None:
        dict_properties = _read_properties_database()

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

    df = _rename_columns_and_set_units(
        df=df,
        column_names_and_units=[
            ('id', '_id_aircraft', 'str'),
            ('name', 'Aircraft Designation', 'str'),
            ('manufacturer', 'Aircraft Manufacturer', 'str'),
            ('engineModels', '_id_engine', 'str'),
            ('engineCount', 'Engine Count', 'Int64'),
            ('Fuel Capacity [L]', 'Fuel Capacity', 'pint[l]'),
            ('Mlw [kg]', 'MLW', 'pint[kg]'),
            ('Mtow [kg]', 'MTOW', 'pint[kg]'),
            ('Mtw [kg]', 'MTW', 'pint[kg]'),
            ('Mzfw [kg]', 'MZFW', 'pint[kg]'),
            ('Mmo', 'MMO', 'pint[dimensionless]'),
            ('Maximum Operating Altitude [ft]', 'Maximum Operating Altitude', 'pint[ft]'),
            ('Oew [kg]', 'OEW', 'pint[kg]'),
            ('Wing Area [m²]', 'Wing Area', 'pint[m**2]'),
            ('Wingspan (Canard) [m]', 'Wingspan (Canard)', 'pint[m]'),
            ('Wingspan (Winglets) [m]', 'Wingspan (winglets)', 'pint[m]'),
            ('Wingspan [m]', 'Wingspan', 'pint[m]'),
            ('Height [m]', 'Height', 'pint[m]'),
        ],
        return_only_renamed_columns=True,
    )

    df_grouped = df.groupby(
        by=[
            'Aircraft Designation',
            '_id_engine',
        ],
        as_index=False,
    ).first()
    
    return df_grouped


def enrich_aircraft_database(
    path_json_aircraft_database: str = PATH_ZENODO_AIRCRAFT_DATABASE_AIRCRAFT_TYPES_FILE,
    path_json_engine_database: str = PATH_ZENODO_AIRCRAFT_DATABASE_ENGINE_MODELS_FILE,
    path_json_properties: str = PATH_ZENODO_AIRCRAFT_DATABASE_PROPERTIES_FILE,
    path_json_manufacturers: str = PATH_ZENODO_AIRCRAFT_DATABASE_MANUFACTURERS_FILE,
) -> pd.DataFrame:
    r"""
    Enriches the aircraft database with engine properties by merging the aircraft and engine databases
    from the [aircraft-database.com](https://web.archive.org/web/20231201220700/https://aircraft-database.com/)
    (discontinued as of 01-2024).

    This function serves as a high-level wrapper that first processes the aircraft and engine
    datasets separately using helper functions and then combines them.

    See Also
    --------
    [`aircraftdetective.processing.acftdb._read_aircraft_database`][]
    [`aircraftdetective.processing.acftdb._read_engine_database`][]
    [`aircraftdetective.processing.acftdb._read_properties_database`][]
    [`aircraftdetective.processing.acftdb._read_manufacturers_database`][]

    Notes
    -----
    With no parameters passed, the function will download the relevant JSON files 
    from the [aircraft-database.com Backup Zenodo repository](https://doi.org/10.5281/zenodo.14382244).

    Parameters
    ----------
    path_json_aircraft_database : str
        Path to the JSON file containing the aircraft models and their properties.
    path_json_engine_database : str
        Path to the JSON file containing the engine models and their properties.
    path_json_properties : str
        Path to the JSON file containing the property IDs and their names.
    path_json_manufacturers : str
        Path to the JSON file containing the manufacturer IDs and their names.

    Returns
    -------
    pd.DataFrame
        A [`pint-pandas`](https://pint-pandas.readthedocs.io/en/latest/) DataFrame containing the merged 
        aircraft and engine data. Each row represents a specific aircraft model equipped with a 
        specific engine model.
    """
    df_aircraft = _read_aircraft_database(
        path_json_aircraft_database=path_json_aircraft_database,
    )
    df_engines = _read_engine_database(
        path_json_engine_database=path_json_engine_database,
        dict_properties=_read_properties_database(path_json_properties=path_json_properties),
        dict_manufacturers=_read_manufacturers_database(path_json_manufacturers=path_json_manufacturers),
    )
    df_aircraft_enriched = pd.merge(
        left=df_aircraft,
        right=df_engines,
        how='left',
        on='_id_engine',
    )
    return df_aircraft_enriched