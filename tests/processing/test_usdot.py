import pytest
import pandas as pd
import pint_pandas
from pathlib import Path

from aircraftdetective.processing.usdot import process_data_usdot_t2


@pytest.fixture(scope="module")
def t2_fixture_path() -> Path:
    """A pytest fixture that returns the path to the T2 schedule test data."""
    current_dir = Path(__file__).parent
    return current_dir / "data" / "fixture_t_schedule_t2.csv"


class TestProcessDataUsdotT2:
    """
    Test suite for the `process_data_usdot_t2` function.
    """

    @pytest.fixture(scope="class")
    def processed_df(self, t2_fixture_path: Path) -> pd.DataFrame:
        """
        A class-scoped fixture that runs the data processing function once
        and provides the resulting DataFrame to all tests in this class.
        """
        return process_data_usdot_t2(path_csv_t2=t2_fixture_path)

    def test_output_structure_and_columns(self, processed_df: pd.DataFrame):
        """
        Tests that the output is a non-empty DataFrame with the correct columns.
        """
        assert isinstance(processed_df, pd.DataFrame)
        assert not processed_df.empty, "The resulting DataFrame should not be empty."

        expected_columns = [
            'Year',
            'Aircraft Designation (US DOT Schedule T2)',
            'Fuel/Available Seat Distance',
            'Fuel/Revenue Seat Distance',
            'Fuel Flow',
            'Energy Use (per ASK)',
            'Energy Intensity (per RPK)',
            'Airborne Efficiency',
            'SLF',
            'Revenue Passenger Distance',
        ]
        assert processed_df.columns.tolist() == expected_columns, \
            "DataFrame columns are incorrect or not in the expected order."

    def test_column_dtypes_are_correct(self, processed_df: pd.DataFrame):
        """
        Tests that key columns have the correct pint-pandas dtypes for units.
        """
        assert isinstance(processed_df['Fuel Flow'].dtype, pint_pandas.PintType)
        assert isinstance(processed_df['Fuel/Available Seat Distance'].dtype, pint_pandas.PintType)
        assert isinstance(processed_df['SLF'].dtype, pint_pandas.PintType)
        assert pd.api.types.is_integer_dtype(processed_df['Year'])

    def test_data_filtering_and_constraints(self, processed_df: pd.DataFrame):
        """
        Tests that the data sanity checks and filtering logic were applied correctly.
        """
        assert processed_df['SLF'].pint.magnitude.between(0, 1).all(), \
            "Found SLF values outside the valid range of [0, 1]."

        assert processed_df['Airborne Efficiency'].pint.magnitude.between(0, 1).all(), \
            "Found Airborne Efficiency values outside the valid range of [0, 1]."

    def test_specific_calculation_is_correct(self, processed_df: pd.DataFrame):
        """
        Tests a specific, known calculation for a row from the fixture file
        to verify the correctness of the data processing logic.
        """
        a320_row = processed_df[
            processed_df['Aircraft Designation (US DOT Schedule T2)'].str.contains("A320")
        ]
        assert not a320_row.empty, "Test fixture should contain data for an A320."

        expected_fuel_flow = 966
        actual_fuel_flow = a320_row.iloc[0]['Fuel Flow'].magnitude
        assert actual_fuel_flow == pytest.approx(expected_fuel_flow, abs=10)