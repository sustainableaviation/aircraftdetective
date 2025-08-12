import pytest
import pandas as pd
from pathlib import Path
import io

from aircraftdetective.processing.operations import process_data_usdot_t2

@pytest.fixture
def t2_fixture_path() -> Path:
    """A pytest fixture that returns the path to the T2 schedule test data."""
    # Get the directory of the current test file.
    current_dir = Path(__file__).parent
    # Construct the path to the fixture relative to the test file.
    # This assumes the fixture is in a 'fixtures' subdirectory next to the test file.
    return current_dir / "data" / "fixture_t_schedule_t2.csv"


# --- 3. The test function now requests the fixture as an argument ---
def test_process_data_usdot_t2_with_fixture(t2_fixture_path: Path):
    """
    Tests the data processing function using a real fixture file.
    This test validates the end-to-end logic, including data filtering,
    column renaming, calculations, and sanity checks.

    Args:
        t2_fixture_path (Path): The path to the fixture file, provided by the
                                pytest fixture of the same name.
    """
    # --- Execution: Call the function under test ---
    # The path is now cleanly passed in from the fixture.
    # We are intentionally NOT providing `path_csv_aircraft_types` to test
    # that the function correctly uses the default file bundled with the package.
    result_df = process_data_usdot_t2(path_csv_t2=t2_fixture_path)

    # --- Assertion: Check if the output is correct and valid ---

    # Assert that the function returns a non-empty DataFrame
    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty, "The resulting DataFrame should not be empty."

    # Assert that the final columns are exactly as expected
    expected_columns = [
        'Aircraft Designation (US DOT Schedule T2)',
        'Fuel/Available Seat Distance',
        'Fuel/Revenue Seat Distance',
        'Fuel Flow',
        'Airborne Efficiency',
        'SLF',
    ]
    assert result_df.columns.tolist() == expected_columns, "The DataFrame columns are not correct."

    # --- Sanity checks based on the function's internal logic ---

    # The Seat Load Factor (SLF) must be between 0 and 1.
    # This also implicitly checks that the 'REV_PAX_MILES' <= 'AVL_SEAT_MILES' filter worked.
    assert result_df['SLF'].between(0, 1).all(), "SLF values should be between 0 and 1."

    # The Airborne Efficiency must be between 0 and 1.
    # This also implicitly checks that 'HOURS_AIRBORNE' <= 'ACRFT_HRS_RAMPTORAMP' filter worked.
    assert result_df['Airborne Efficiency'].between(0, 1).all(), "Airborne Efficiency values should be between 0 and 1."

    # Check that numeric columns are indeed numeric
    assert pd.api.types.is_numeric_dtype(result_df['Fuel Flow'])
    assert pd.api.types.is_numeric_dtype(result_df['SLF'])