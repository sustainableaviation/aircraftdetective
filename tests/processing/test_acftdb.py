import pytest
import pandas as pd
from aircraftdetective.processing.acftdb import (
    _read_properties_database,
    _read_manufacturers_database,
    _read_engine_database,
    _read_aircraft_database,
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
    assert engines_df.shape[1] == 10
    expected_columns = [
        'Engine Designation',
        '_id_engine',
        'Engine Family',
        'Engine Manufacturer',
        'Bypass Ratio',
        'Overall Pressure Ratio',
        'Dry Weight [kg]',
        'Fan Diameter [m]',
        'Max. Continuous Power [kW]',
        'Max. Continuous Thrust [kN]',
    ]
    for col in expected_columns:
        assert col in engines_df.columns

def test_read_aircraft_database():
    aircraft_df = _read_aircraft_database()
    assert isinstance(aircraft_df, pd.DataFrame)
    assert not aircraft_df.empty
    assert aircraft_df.shape[0] == 3167
    assert aircraft_df.shape[1] == 18
    expected_columns = [
        'Aircraft Designation',
        '_id_engine',
        '_id_aircraft',
        'Aircraft Manufacturer',
        'Engine Count',
        'Fuel Capacity [l]',
        'MLW [kg]',
        'MTOW [kg]',
        'MTW [kg]',
        'MZFW [kg]',
        'MMO',
        'Maximum Operating Altitude [ft]',
        'OEW [kg]',
        'Wing Area [m²]',
        'Wingspan (Canard) [m]',
        'Wingspan (winglets) [m]',
        'Wingspan [m]',
        'Height [m]'
    ]
    for col in expected_columns:
        assert col in aircraft_df.columns