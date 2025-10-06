import pandas as pd
import pint_pandas
import pytest
import numpy as np
import numpy.testing as npt
import tempfile
from pathlib import Path

from aircraftdetective.calculations.engines import (
    determine_takeoff_to_cruise_tsfc_ratio,
    scale_engine_data_from_icao_emissions_database
)

class TestDetermineTakeoffToCruiseTsfcRatio:
    """
    Test suite for the `determine_takeoff_to_cruise_tsfc_ratio` function.
    """

    def test_with_default_data_and_valid_degrees(self):
        """
        Tests the polynomial fitting using the default data source with valid
        degree parameters (1 and 2). This test requires network access.
        """

        result_linear = determine_takeoff_to_cruise_tsfc_ratio(degree=1)

        assert isinstance(result_linear, dict)
        expected_keys = [
            'TSFC (cruise)',
            'TSFC (cruise)_r2'
        ]
        assert all(key in result_linear for key in expected_keys)
        poly_linear = result_linear['TSFC (cruise)']
        assert isinstance(poly_linear, np.polynomial.Polynomial)
        assert poly_linear.degree() == 1
        assert isinstance(result_linear['TSFC (cruise)_r2'], float)

        result_quadratic = determine_takeoff_to_cruise_tsfc_ratio() # Uses degree=2 default
        
        assert isinstance(result_quadratic, dict)
        assert all(key in result_quadratic for key in expected_keys)
        poly_quadratic = result_quadratic['TSFC (cruise)']
        assert isinstance(poly_quadratic, np.polynomial.Polynomial)
        assert poly_quadratic.degree() == 2
        assert isinstance(result_quadratic['TSFC (cruise)_r2'], float)

    def test_invalid_degree_raises_value_error(self):
        """
        Tests that the function raises ValueError for invalid `degree` parameters.
        This check occurs before file access, so it does not require network access.
        """
        with pytest.raises(ValueError, match="degree must be a positive integer."):
            determine_takeoff_to_cruise_tsfc_ratio(degree=0)
        
        with pytest.raises(ValueError, match="degree must be a positive integer."):
            determine_takeoff_to_cruise_tsfc_ratio(degree=-5)
            
        with pytest.raises(ValueError, match="degree must be a positive integer."):
            determine_takeoff_to_cruise_tsfc_ratio(degree=1.5)


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