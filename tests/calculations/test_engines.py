import pandas as pd
import pint_pandas
import pytest
import numpy as np
import numpy.testing as npt
import tempfile
from pathlib import Path
from aircraftdetective import ureg
import math

from aircraftdetective.calculations.engines import (
    determine_takeoff_to_cruise_tsfc_ratio,
    scale_engine_data_from_icao_emissions_database,
    calculate_air_mass_flow_rate
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


@pytest.fixture
def sample_engine_df() -> pd.DataFrame:
    """Provides a sample DataFrame with engine data and pint units."""
    data = {
        'Engine': ['Engine A', 'Engine B'],
        'Cruise Speed': [250 * ureg('m/s'), 240 * ureg('m/s')],
        'Fan Diameter': [3.4 * ureg('meter'), 3.2 * ureg('meter')],
    }
    return pd.DataFrame(data)

class TestCalculateAirMassFlowRate:
    """Test suite for the `calculate_air_mass_flow_rate` function."""

    def test_calculates_correctly_at_default_altitude(self, sample_engine_df):
        """
        Tests if the function calculates the correct air mass flow rate at the
        default altitude of 12000 meters.
        """
        # At 12000m, our model function gives a density of ~0.3119 kg/m^3
        
        # Expected calculations
        # Engine A: 0.31 * 250 * pi * (3.4/2)^2 = 708.3 kg/s
        # Engine B: 0.31 * 240 * pi * (3.2/2)^2 = 600.3 kg/s
        expected_flow_a = (0.31 * 250 * ureg('m/s') * math.pi * (3.4 * ureg.meter / 2)**2).to('kg/s')
        expected_flow_b = (0.31 * 240 * ureg('m/s') * math.pi * (3.2 * ureg.meter / 2)**2).to('kg/s')
        
        result_df = calculate_air_mass_flow_rate(sample_engine_df)
        
        # Check that the results are approximately equal
        assert result_df['Air Mass Flow'].iloc[0].magnitude == pytest.approx(expected_flow_a.magnitude)
        assert result_df['Air Mass Flow'].iloc[1].magnitude == pytest.approx(expected_flow_b.magnitude)
        assert result_df['Air Mass Flow'].iloc[0].units == ureg.kilogram / ureg.second

    def test_calculates_correctly_at_sea_level(self, sample_engine_df):
        """
        Tests the calculation with a different altitude (sea level) to ensure
        the density change is correctly handled.
        """
        # At 0m (sea level), density is ~1.225 kg/m^3
        air_density = _calculate_atmospheric_conditions(0 * ureg.meter)['density']
        
        # Expected calculation for Engine A at sea level
        expected_flow = (air_density * 250 * ureg('m/s') * math.pi * (3.4 * ureg.meter / 2)**2).to('kg/s')

        result_df = calculate_air_mass_flow_rate(sample_engine_df, altitude=0 * ureg.meter)
        
        assert result_df['Air Mass Flow'].iloc[0].magnitude == pytest.approx(expected_flow.magnitude)
        assert result_df['Air Mass Flow'].iloc[0].units == ureg.kilogram / ureg.second

    def test_input_dataframe_is_not_modified(self, sample_engine_df):
        """
        Tests that the original input DataFrame is not mutated by the function,
        as it should operate on a copy.
        """
        df_original = sample_engine_df.copy()
        calculate_air_mass_flow_rate(df_original)
        
        # The original DataFrame should not have the 'Air Mass Flow' column
        assert 'Air Mass Flow' not in df_original.columns
        pd.testing.assert_frame_equal(df_original, sample_engine_df)

    def test_missing_required_columns_raises_keyerror(self, sample_engine_df):
        """
        Tests that a KeyError is raised if essential columns are missing.
        """
        df_no_speed = sample_engine_df.drop(columns=['Cruise Speed'])
        with pytest.raises(KeyError):
            calculate_air_mass_flow_rate(df_no_speed)
            
        df_no_diameter = sample_engine_df.drop(columns=['Fan Diameter'])
        with pytest.raises(KeyError):
            calculate_air_mass_flow_rate(df_no_diameter)

    def test_empty_dataframe_returns_empty_with_new_column(self):
        """
        Tests that passing an empty DataFrame results in an empty DataFrame
        with the new column added.
        """
        empty_df = pd.DataFrame({
            'Cruise Speed': pd.Series(dtype='object'),
            'Fan Diameter': pd.Series(dtype='object')
        })
        
        result_df = calculate_air_mass_flow_rate(empty_df)
        
        assert result_df.empty
        assert 'Air Mass Flow' in result_df.columns
        assert list(result_df.columns) == ['Cruise Speed', 'Fan Diameter', 'Air Mass Flow']