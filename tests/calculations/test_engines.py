import pandas as pd
import pint_pandas
import pytest
import numpy as np
import numpy.testing as npt

from aircraftdetective.calculations.engines import (
    determine_takeoff_to_cruise_tsfc_ratio,
    scale_engine_data_from_icao_emissions_database
)

import tempfile
from pathlib import Path

def test_determine_takeoff_to_cruise_tsfc_ratio():
    """
    Tests the polynomial fitting for TSFC data by creating a temporary Excel file.
    This test checks that the function returns a dictionary with the correct structure.
    """
    input_data = [
        ['Engine Identification', 'TSFC (takeoff)', 'TSFC (cruise)'],
        ['No Unit', 'g/(kN*s)', 'g/(kN*s)'],
        ["Eng-A", 10, 18],
        ["Eng-B", 12, 22],
        ["Eng-C", 15, 28],
        ["Eng-D", 18, 35]
    ]
    sample_df = pd.DataFrame(input_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "temp_tsfc_calibration.xlsx"
        with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
            sample_df.to_excel(writer, sheet_name='Data', index=False, header=False)

        result_dict = determine_takeoff_to_cruise_tsfc_ratio(
            path_excel_engine_data_for_calibration=temp_path
        )

    expected_keys = [
        'df_engines',
        'pol_linear_fit',
        'pol_quadratic_fit',
        'r_squared_linear_fit',
        'r_squared_quadratic_fit'
    ]

    assert all(key in result_dict for key in expected_keys), "Result dictionary is missing expected keys"

    assert not result_dict['df_engines'].empty, "The returned DataFrame should not be empty"

    assert isinstance(result_dict['pol_linear_fit'], np.polynomial.Polynomial), "Linear fit is not a Polynomial object"
    assert isinstance(result_dict['pol_quadratic_fit'], np.polynomial.Polynomial), "Quadratic fit is not a Polynomial object"

    assert isinstance(result_dict['r_squared_linear_fit'], float), "Linear R-squared is not a float"
    assert isinstance(result_dict['r_squared_quadratic_fit'], float), "Quadratic R-squared is not a float"


def test_scale_engine_data():
    """
    Tests the scaling of engine data by creating and reading a temporary Excel file.
    This test checks for new columns and verifies the scaled value based on a simple polynomial.
    """
    input_data = {
        "Engine Identification": ["TestEngine-01"],
        "Final Test Date": [pd.to_datetime("2023-01-01")],
        "Fuel Flow T/O (kg/sec)": [15.0],
        "Fuel Flow C/O (kg/sec)": [12.0],
        "Fuel Flow App (kg/sec)": [8.0],
        "Fuel Flow Idle (kg/sec)": [2.0],
        "B/P Ratio": [5.0],
        "Pressure Ratio": [30.0],
        "Rated Thrust (kN)": [100.0],
    }
    sample_df = pd.DataFrame(input_data)

    # simple linear polynomial for scaling: f(x) = 2*x + 5
    scaling_poly = np.polynomial.Polynomial([5, 2])
    # TSFC (takeoff) = (15 kg/s / 100 kN) = 0.15 kg/(kN*s) -> 150 g/(kN*s)
    expected_tsfc_takeoff_mag = 150.0
    # TSFC (cruise) = 2 * 150 + 5 = 305
    expected_tsfc_cruise_mag = 305.0

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "temp_engine_data.xlsx"
        with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
            sample_df.to_excel(writer, sheet_name='Gaseous Emissions and Smoke', index=False)

        result_df = scale_engine_data_from_icao_emissions_database(
            path_excel_engine_data_icao_in=temp_path,
            scaling_polynomial=scaling_poly
        )

    assert 'TSFC (takeoff)' in result_df.columns, "Column 'TSFC (takeoff)' should exist"
    assert 'TSFC (cruise)' in result_df.columns, "Column 'TSFC (cruise)' should exist"

    actual_tsfc_cruise_mag = result_df['TSFC (cruise)'].pint.magnitude.iloc[0]
    np.testing.assert_allclose(
        actual_tsfc_cruise_mag,
        expected_tsfc_cruise_mag,
        err_msg="Scaled TSFC (cruise) value is incorrect"
    )