# %%
import pytest
import pandas as pd
from aircraftdetective.processing.acftdb import (
    _read_properties_database,
    _read_manufacturers_database,
    _read_engine_database,
    _read_aircraft_database,
    enrich_aircraft_database
)

def test_read_properties_database():
    properties = _read_properties_database()
    assert isinstance(properties, dict)
    assert len(properties) == 106

def test_read_manufacturers_database():
    manufacturers = _read_manufacturers_database()
    assert isinstance(manufacturers, dict)
    assert len(manufacturers) == 364

def test_read_engine_database():
    engines_df = _read_engine_database()
    assert isinstance(engines_df, pd.DataFrame)
    assert not engines_df.empty
    assert engines_df.shape[0] == 1346
    assert engines_df.shape[1] == 7
    expected_dtypes = {
        'Engine Designation': 'object',
        'Engine Family': 'object',
        'Engine Manufacturer': 'object',
        'Overall Pressure Ratio': 'pint[dimensionless]',
        'Dry Weight': 'pint[kilogram]',
        'Fan Diameter': 'pint[meter]',
        'Max. Continuous Thrust': 'pint[kilonewton]',
    }
    assert set(engines_df.columns) == set(expected_dtypes.keys())
    for col, expected_dtype in expected_dtypes.items():
        actual_dtype = str(engines_df[col].dtype)
        assert actual_dtype.startswith(expected_dtype), (
            f"Column '{col}' has incorrect dtype. "
            f"Expected: {expected_dtype}, Actual: {actual_dtype}"
        )

def test_read_aircraft_database():
    aircraft_df = _read_aircraft_database()
    assert isinstance(aircraft_df, pd.DataFrame)
    assert not aircraft_df.empty
    assert aircraft_df.shape[0] == 3232
    assert aircraft_df.shape[1] == 18
    expected_dtypes = {
        'Aircraft Designation': 'object',
        'Aircraft Manufacturer': 'object',
        '_id_aircraft': 'object',
        '_id_engine': 'object',
        'Engine Count': 'Int64',
        'Fuel Capacity': 'pint[liter]',
        'MLW': 'pint[kilogram]',
        'MTOW': 'pint[kilogram]',
        'MTW': 'pint[kilogram]',
        'MZFW': 'pint[kilogram]',
        'MMO': 'pint[dimensionless]',
        'Maximum Operating Altitude': 'pint[foot]',
        'OEW': 'pint[kilogram]',
        'Wing Area': 'pint[meter ** 2]',
        'Wingspan (Canard)': 'pint[meter]',
        'Wingspan (winglets)': 'pint[meter]',
        'Wingspan': 'pint[meter]',
        'Height': 'pint[meter]',
    }
    assert set(aircraft_df.columns) == set(expected_dtypes.keys())
    for col, expected_dtype in expected_dtypes.items():
        actual_dtype = str(aircraft_df[col].dtype)
        assert actual_dtype.startswith(expected_dtype), (
            f"Column '{col}' has incorrect dtype. "
            f"Expected: {expected_dtype}, Actual: {actual_dtype}"
        )


def test_enrich_aircraft_database():
    enriched_df = enrich_aircraft_database()
    assert isinstance(enriched_df, pd.DataFrame)
    assert not enriched_df.empty