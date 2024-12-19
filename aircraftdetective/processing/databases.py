# %%
from pathlib import Path

import pandas as pd

import pint
ureg = pint.get_application_registry()
import pint_pandas


import sys
import os
module_path = os.path.abspath("/Users/michaelweinold/github/aircraftdetective")
if module_path not in sys.path:
    sys.path.append(module_path)

from aircraftdetective.auxiliary import dataframe
from aircraftdetective import config

# %%


def read_aircraft_database(
    path_json_aircraft_database: Path,
    dict_manufacturers: dict,
    list_column_names_and_units_properties: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """
    Reads a JSON backup of the the entire [aircraft-database.com](https://doi.org/10.5281/zenodo.14382244)
    `aircraft-types` table and returns a DataFrame with all aircraft models and their properties.
    
    Since aircraft can have multiple engines and `propertyValues`,
    the function explodes these columns to have one row per engine and property.
    
    For example, the following JSON object:

    ```
    {
        "id":"123abc",
        "aircraftFamily":"airplane",
        "name":"Aircraft 333",
        "engineModels":["alpha","beta"],
        "propertyValues":[
            {"property":"345cde","value":42},
            {"property":"456def","value":12.45},
        ],
        (...)
    },
    ```

    will be transformed into a DataFrame of the kind:

    | id     | Name         | Engine Model | Property 1 | Property 2 | (...) |
    |--------|--------------|--------------|------------|------------|-------|
    | 123abc | Aircraft 333 | alpha        | 42         | 12.45      | (...) |
    | 123abc | Aircraft 333 | beta         | 42         | 12.45      | (...) |

    See Also
    --------
    - [aircraft-database.com (archived)](https://web.archive.org/web/20231201220700/https://aircraft-database.com/)
    - [aircraft-database.com (Zenodo data re-upload)](https://doi.org/10.5281/zenodo.14382244)

    Parameters
    ----------
    path_json_aircraft_models : Path
        Path to the JSON file containing the aircraft models and their properties.
    dict_properties : dict
        Dictionary mapping the property IDs to their names.
    list_column_names_and_units_properties : list[tuple[str, str, str]]
        List of tuples with the property id, property name and property unit.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with the aircraft models and their properties. Columns are of `pint_pandas` data types.
    """

    df = pd.read_json(path_json_aircraft_database)

    # remove helicopters, amphibious, etc.
    df = df.loc[df['aircraftFamily'].isin(['airplane'])]
    # remove entries with tags (e.g. ['military'])
    df = df[df['tags'].str.len() == 0].drop(columns=['tags'])
    # add manufacturer names
    df['manufacturer'] = df['manufacturer'].map(dict_manufacturers)

    """
    Explode engine models. From a dataframe of the kind:

    | id | (...) | engineModels        |
    |----|-------|---------------------|
    | 1  | (...) | ['alpha', 'beta']   |

    generate a dataframe of the kind:

    | id | (...) | engineModels |
    |----|-------|--------------|
    | 1  | (...) | alpha        |
    | 1  | (...) | beta         |
    """
    df = df.explode('engineModels').reset_index(drop=True)
    
    """
    Explode propertyValues. From a dataframe of the kind:

    | id | (...) | propertyValues                                                          |
    |----|-------|-------------------------------------------------------------------------|
    | 1  | (...) | [{"property":"345cde","value":42}, {"property":"456def","value":12.45}] |
    | 2  | (...) | [{"property":"345cde","value":13}, {"property":"456def","value":23.56}] |

    generate a dataframe of the kind:

    | id | (...) | 345cde | 456def |
    |----|-------|--------|--------|
    | 1  | (...) | 42     | 12.45  |
    | 2  | (...) | 13     | 23.56  |

    by generating a Pandas Series of the kind:

    | index | propertyValues                                                        |
    |-------|-----------------------------------------------------------------------|
    | 0     | {"property":"345cde","value":42}, {"property":"456def","value":12.45} |
    | 1     | {"property":"345cde","value":13}, {"property":"456def","value":23.56} |

    and converting it to a DataFrame of the kind:

    | index | 345cde | 456def |
    |-------|--------|--------|
    | 0     | 42     | 12.45  |
    | 1     | 13     | 23.56  |
    """
    property_values: pd.Series = df["propertyValues"].apply(
        lambda x: {item["property"]: item["value"] for item in x}
    )
    properties_df = pd.DataFrame(property_values.tolist())
    df = pd.concat(
        objs=[df.drop(columns=["propertyValues"]), properties_df],
        axis=1
    )

    df = dataframe.rename_columns_and_set_units(
        df=df,
        column_names_and_units=[
            ("id", "_id_aircraft", str),
            ("engineModels", "_id_engine", str),
            ("manufacturer", "Aircraft Manufacturer", str),
            ("name", "Aircraft Designation", str),
            ("engineCount", "Engine Count", int),
        ]
    )
    df = dataframe.rename_columns_and_set_units(
        df=df,
        column_names_and_units=list_column_names_and_units_properties
    )

    df = df[
        [
            "_id_aircraft",
            "_id_engine",
            "Aircraft Manufacturer",
            "Aircraft Designation",
            "Engine Count",
            "MTOW",
            "MZFW",
            "OEW",
            "Wing Area",
            "Wingspan (Winglets)",
            "Wingspan",
        ]
    ]

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
    dict_manufacturers: dict,
    list_column_names_and_units_properties: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """
    Reads a JSON backup of the the entire [aircraft-database.com](https://doi.org/10.5281/zenodo.14382244)
    `engine-models` table and returns a DataFrame with all engine models and their properties.

    For example, the following JSON object:

    ```
    {
        "id":"123abc",
        "engineFamily":"turbofan",
        "name":"Engine 444",
        "propertyValues":[
            {"property":567def","value":42},
            {"property":"678fgh","value":12.45},
        ],
        (...)
    },
    ```

    will be transformed into a DataFrame of the kind:

    | id     | Name         | Property 1 | Property 2 | (...) |
    |--------|--------------|------------|------------|-------|
    | 123abc | Engine 444   | 42         | 12.45      | (...) |
    
    See Also
    --------
    - [aircraft-database.com (archived)](https://web.archive.org/web/20231201220700/https://aircraft-database.com/)
    - [aircraft-database.com (Zenodo data re-upload)](https://doi.org/10.5281/zenodo.14382244)

    Parameters
    ----------
    path_json_aircraft_models : Path
        Path to the JSON file containing the aircraft models and their properties.
    dict_properties : dict
        Dictionary mapping the property IDs to their names.
    list_column_names_and_units_properties : list[tuple[str, str, str]]
        List of tuples with the property id, property name and property unit.

    Returns
    -------
    pd.DataFrame
        DataFrame with the engine models and their properties. Columns are of `pint_pandas` data types.
    """

    df = pd.read_json(path_json_engine_database)
    # remove turboprop, piston, etc.
    df = df.loc[df['engineFamily'].isin(['turbofan', 'turbojet'])]
    # remove entries with tags (e.g. ['military'])
    df = df[df['tags'].str.len() == 0].drop(columns=['tags'])
    df = df.reset_index(drop=True)
    # add manufacturer names
    df['manufacturer'] = df['manufacturer'].map(dict_manufacturers)

    """
    Explode propertyValues. From a dataframe of the kind:

    | id | (...) | propertyValues                                                          |
    |----|-------|-------------------------------------------------------------------------|
    | 1  | (...) | [{"property":"345cde","value":42}, {"property":"456def","value":12.45}] |
    | 2  | (...) | [{"property":"345cde","value":13}, {"property":"456def","value":23.56}] |

    generate a dataframe of the kind:

    | id | (...) | 345cde | 456def |
    |----|-------|--------|--------|
    | 1  | (...) | 42     | 12.45  |
    | 2  | (...) | 13     | 23.56  |

    by generating a Pandas Series of the kind:

    | index | propertyValues                                                        |
    |-------|-----------------------------------------------------------------------|
    | 0     | {"property":"345cde","value":42}, {"property":"456def","value":12.45} |
    | 1     | {"property":"345cde","value":13}, {"property":"456def","value":23.56} |

    and converting it to a DataFrame of the kind:

    | index | 345cde | 456def |
    |-------|--------|--------|
    | 0     | 42     | 12.45  |
    | 1     | 13     | 23.56  |
    """
    property_values: pd.Series = df["propertyValues"].apply(
        lambda x: {item["property"]: item["value"] for item in x}
    )
    properties_df = pd.DataFrame(property_values.tolist())
    df = pd.concat(
        objs=[df.drop(columns=["propertyValues"]), properties_df],
        axis=1
    )

    df = dataframe.rename_columns_and_set_units(
        df=df,
        column_names_and_units=[
            ("id", "_id_engine", str),
            ("name", "Engine Designation", str),
            ("manufacturer", "Engine Manufacturer", str),
        ]
    )
    df = dataframe.rename_columns_and_set_units(
        df=df,
        column_names_and_units=list_column_names_and_units_properties
    )

    df_grouped = df.groupby(
        by='Engine Designation',
        as_index=False,
    ).first()

    return df_grouped


def read_properties_database(path_json_properties: str) -> list[tuple[str, str, str]]:
    """
    Reads a JSON backup of the the [aircraft-database.com](https://doi.org/10.5281/zenodo.14382244)
    `properties` table and returns a list of tuples with the property id, property name and property unit.

    For instance, the following JSON object:

    ```
    [
        {"id":"1ed5603b-a0c8-6728-9bf8-93b7fcdd8f72","name":"Hull volume","type":"integer","unit":"cubic-metre"},
        {"id":"1ed95aff-38d7-6552-9ae8-b35cc598320e","name":"Battery capacity","type":"float","unit":"kilowatt-hour"},
        {"id":"1ed95b07-4149-69c6-9ae8-c3b0ecc17a8d","name":"Battery technology","type":"string","unit":null},
        (...)
    ]
    ```

    is transformed into a list of tuples of the kind:

    ```
    [
        ("1ed5603b-a0c8-6728-9bf8-93b7fcdd8f72", "Hull Volum", "pint[m ** 3]"),
        ("1ed95aff-38d7-6552-9ae8-b35cc598320e", "Battery Capacity", "pint[kWh]"),
        ("1ed95b07-4149-69c6-9ae8-c3b0ecc17a8d", "Battery Technology", None),
        (...) 
    ]
    ```

    Parameters
    ----------
    path_json_properties : str
        Path or URL of the JSON file containing the aircraft-database.com `properties` metadata.

    Returns
    -------
    list[tuple[str, str, str]]
        List of tuples with the property id, property name and property unit.
    """

    dict_units = {
        None: None,
        'kilowatt-hour': 'pint[kWh]',
        'volt': 'pint[V]',
        'millimetre': 'pint[mm]',
        'cubic-centimetre': 'pint[cm ** 3]',
        'kilogram': 'pint[kg]',
        'metre': 'pint[m]',
        'litre': 'pint[L]',
        'horsepower': 'pint[hp]',
        'kilowatt': 'pint[kW]',
        'kilonewton': 'pint[kN]',
        'decanewton-metre': 'pint[daN * m]',
        'foot': 'pint[ft]',
        'knot': 'pint[kts]',
        'cubic-metre': 'pint[m ** 3]',
        'square-metre': 'pint[m ** 2]',
    }

    df_properties = pd.read_json(path_json_properties)[['id', 'name', 'unit']]
    df_properties['unit'] = df_properties['unit'].map(dict_units)
    list_column_names_and_units = list(df_properties.itertuples(index=False, name=None))
    
    return list_column_names_and_units


def read_manufacturers_database(path_json_manufacturers: Path) -> dict:
    """
    Reads a JSON backup of the the [aircraft-database.com](https://doi.org/10.5281/zenodo.14382244)
    `manufacturers` table and returns a dictionary with the manufacturer id and manufacturer name.

    For instance, the following JSON object:

    ```
    [
        {
            "id": "1edc26ff-b373-6418-bf55-35eb9f006d3f",
            "country": "MX",
            "name": "AAMSA",
            "propertyValues": [],
            "tags": [],
            "url": "https://aircraft-database.com/database/manufacturers/aamsa"
        },
        {
            "id": "1ed62800-0b15-6f68-b5ec-05fb44d7d394",
            "country": "GB",
            "name": "ABC Motors",
            "propertyValues": [],
            "tags": [],
            "url": "https://aircraft-database.com/database/manufacturers/abc-motors"
        },
        (...)
    ]
    ```

    is transformed into a dictionary of the kind:

    ```
    {
        "1edc26ff-b373-6418-bf55-35eb9f006d3f": "AAMSA",
        "1ed62800-0b15-6f68-b5ec-05fb44d7d394": "ABC Motors",
        (...)
    }
    ```

    Parameters
    ----------
    path_json_manufacturers : str
        Path or URL of the JSON file containing the aircraft-database.com `manufacturers` metadata.

    Returns
    -------
    dict
        Dictionary with manufacturer id as keys and manufacturer name as values.
        
    """
    df_manufacturers = pd.read_json(path_json_manufacturers)[['id', 'name']]
    dict_manufacturers = df_manufacturers.set_index('id')['name'].to_dict()

    return dict_manufacturers


def enrich_aircraft_database(
    path_json_aircraft_database: str,
    path_json_properties: str,
    path_json_manufacturers: str
) -> pd.DataFrame:
    df_aircraft = read_aircraft_database(
        path_json_aircraft_database=path_json_aircraft_database,
        dict_properties=read_properties_database(path_json_properties),
        dict_manufacturers=read_manufacturers_database(path_json_manufacturers)
    )
    df_engines = read_engine_database(
        path_json_engine_database=path_json_aircraft_database,
        dict_properties=read_properties_database(path_json_properties),
        dict_manufacturers=read_manufacturers_database(path_json_manufacturers)
    )
    df_aircraft_enriched = pd.merge(
        left=df_aircraft,
        right=df_engines,
        how='left',
        on='_id_engine',
    )
    return df_aircraft_enriched