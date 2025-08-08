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


def test_determine_takeoff_to_cruise_tsfc_ratio_1():
    """
    
    """
    output = determine_takeoff_to_cruise_tsfc_ratio('tests/fixtures/fixture_tsfc_data.xlsx')
    assert isinstance(output, dict)
    expected_linear_pol = np.polynomial.polynomial.Polynomial([21.25,  6.25], domain=[25., 40.], window=[-1.,  1.], symbol='x')
    npt.assert_allclose(
        output['pol_linear_fit'].convert().coef,
        expected_linear_pol.convert().coef,
        rtol=1e-5
    )
    expected_quadratic_pol = np.polynomial.polynomial.Polynomial([10.625,  6.25 , 10.625], domain=[25., 40.], window=[-1.,  1.], symbol='x')
    npt.assert_allclose(
        output['pol_quadratic_fit'].convert().coef,
        expected_quadratic_pol.convert().coef,
        rtol=1e-5
    )
    assert output['r_squared_linear_fit'] == pytest.approx(0.892, abs=1e-3)
    assert output['r_squared_quadratic_fit'] == pytest.approx(0.893, abs=1e-3)



def test_determine_takeoff_to_cruise_tsfc_ratio():
    """
    Tests the polynomial fitting for TSFC data by creating a temporary Excel file.
    This test checks that the function returns a dictionary with the correct structure.
    """
    # 1. ARRANGE: Set up test data with a multi-level header for units

    # Create data that can be fitted with a polynomial.
    # The unit strings must not contain brackets for pint-pandas to parse them.
    input_data = {
        ('Engine Identification', 'No Unit'): ["Eng-A", "Eng-B", "Eng-C", "Eng-D"],
        ('TSFC (takeoff)', 'g/(kN*s)'): [10, 12, 15, 18],
        ('TSFC (cruise)', 'g/(kN*s)'): [18, 22, 28, 35],
    }
    sample_df = pd.DataFrame(input_data)

    # Use a temporary directory to safely create the file
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "temp_tsfc_calibration.xlsx"

        # Write the sample DataFrame to the correct sheet with a multi-level header
        with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
            sample_df.to_excel(writer, sheet_name='Data')

        # 2. ACT: Run the function with the path to the temporary file
        result_dict = determine_takeoff_to_cruise_tsfc_ratio(
            path_excel_engine_data_for_calibration=temp_path
        )

    # 3. ASSERT: Check the structure and content of the returned dictionary

    # a) Check that the dictionary contains all the expected keys
    expected_keys = [
        'df_engines',
        'pol_linear_fit',
        'pol_quadratic_fit',
        'r_squared_linear_fit',
        'r_squared_quadratic_fit'
    ]
    assert all(key in result_dict for key in expected_keys), "Result dictionary is missing expected keys"

    # b) Check that the returned DataFrame is not empty
    assert not result_dict['df_engines'].empty, "The returned DataFrame should not be empty"

    # c) Check that the fits are actual numpy Polynomial objects
    assert isinstance(result_dict['pol_linear_fit'], np.polynomial.Polynomial), "Linear fit is not a Polynomial object"
    assert isinstance(result_dict['pol_quadratic_fit'], np.polynomial.Polynomial), "Quadratic fit is not a Polynomial object"

    # d) Check that R-squared values are floats
    assert isinstance(result_dict['r_squared_linear_fit'], float), "Linear R-squared is not a float"
    assert isinstance(result_dict['r_squared_quadratic_fit'], float), "Quadratic R-squared is not a float"



def test_scale_engine_data():
    """
    Tests the scaling of engine data by creating and reading a temporary Excel file.
    This test checks for new columns and verifies the scaled value without mocking.
    """
    input_data = {
        "Engine Identification": ["TestEngine-01"],
        "Final Test Date": ["2023-01-01"],
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