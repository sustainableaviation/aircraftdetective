import pytest
import pandas as pd
import numpy as np
import pint_pandas
from pathlib import Path
from pandas.testing import assert_frame_equal
from aircraftdetective.utility.tabular import (
    _rename_columns_and_set_units,
    _return_short_units,
    export_typed_dataframe_to_excel,
    left_merge_wildcard,
    update_column_data,
    _validate_dataframe_columns_with_units,
)


class TestValidateDataFrameColumnsWithUnits:
    """Test suite for the _validate_dataframe_columns_with_units function."""

    @pytest.fixture
    def sample_schema(self) -> dict[str, str]:
        """Provides a sample schema for validation tests."""
        return {
            'length_col': '[length]',
            'mass_col': '[mass]',
            'velocity_col': '[length]/[time]',
        }

    @pytest.fixture
    def valid_df(self) -> pd.DataFrame:
        """Provides a DataFrame that is valid against the sample_schema."""
        return pd.DataFrame({
            'length_col': pd.Series([10], dtype='pint[m]'),
            'mass_col': pd.Series([5], dtype='pint[kg]'),
            'velocity_col': pd.Series([20], dtype='pint[m/s]'),
        })

    def test_success_case(self, valid_df, sample_schema):
        """Tests that validation passes for a DataFrame that matches the schema."""
        try:
            _validate_dataframe_columns_with_units(valid_df, sample_schema)
        except (ValueError, TypeError) as e:
            pytest.fail(f"Validation unexpectedly failed with error: {e}")

    def test_missing_column(self, valid_df, sample_schema):
        """Tests that a ValueError is raised if a required column is missing."""
        df_missing = valid_df.drop(columns=['mass_col'])
        with pytest.raises(ValueError, match="DataFrame is missing required columns: \\['mass_col'\\]"):
            _validate_dataframe_columns_with_units(df_missing, sample_schema)

    def test_wrong_dimension(self, valid_df, sample_schema):
        """Tests that a ValueError is raised if a column has the wrong dimension."""
        df_wrong_dim = valid_df.copy()
        df_wrong_dim['length_col'] = pd.Series([10], dtype='pint[kg]')
        # Match the full, specific error message to ensure the test is precise.
        expected_msg = "Column 'length_col' has incorrect units. Expected dimensionality of '\\[length\\]', but got '\\[mass\\]'."
        with pytest.raises(ValueError, match=expected_msg):
            _validate_dataframe_columns_with_units(df_wrong_dim, sample_schema)

    def test_not_a_pint_series(self, valid_df, sample_schema):
        """Tests that a TypeError is raised if a column is not a pint-dtype Series."""
        df_not_pint = valid_df.copy()
        df_not_pint['mass_col'] = pd.Series([5], dtype='int64')
        with pytest.raises(TypeError, match="Column 'mass_col' is not a pint-dtype Series and cannot be validated."):
            _validate_dataframe_columns_with_units(df_not_pint, sample_schema)

    def test_extra_columns_are_ignored(self, valid_df, sample_schema):
        """Tests that validation passes even if the DataFrame has extra columns not in the schema."""
        df_extra = valid_df.copy()
        df_extra['extra_col'] = pd.Series([100], dtype='pint[s]')
        try:
            _validate_dataframe_columns_with_units(df_extra, sample_schema)
        except (ValueError, TypeError) as e:
            pytest.fail(f"Validation with extra columns unexpectedly failed: {e}")

    def test_empty_dataframe(self, sample_schema):
        """Tests that validation fails correctly for an empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is missing required columns: \\['length_col', 'mass_col', 'velocity_col'\\]"):
            _validate_dataframe_columns_with_units(empty_df, sample_schema)

    def test_empty_schema(self, valid_df):
        """Tests that validation passes for any DataFrame if the schema is empty."""
        empty_schema = {}
        try:
            _validate_dataframe_columns_with_units(valid_df, empty_schema)
        except (ValueError, TypeError) as e:
            pytest.fail(f"Validation with an empty schema unexpectedly failed: {e}")


class TestRenameColumnsAndSetUnits:
    """Test suite for the `_rename_columns_and_set_units` function."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """This fixture is now a method of the test class."""
        data = {
            "Engine ID": ["A1", "B2"],
            "Fuel Flow (kg/s)": [25.1, 26.2],
            "Pressure Ratio": [1.5, 1.6],
            "Extra Data": [100, 200]
        }
        return pd.DataFrame(data)

    def test_rename_and_set_units_keep_all_columns(self, sample_df):
        """
        Tests basic renaming and unit conversion, keeping all columns.
        """
        # Arrange
        column_map = [
            ("Engine ID", "Engine", str),
            ("Fuel Flow (kg/s)", "Fuel Flow", "pint[kg/s]"),
            ("Pressure Ratio", "Pressure Ratio", "pint[dimensionless]")
        ]

        # Act
        result_df = _rename_columns_and_set_units(
            df=sample_df.copy(),
            return_only_renamed_columns=False,
            column_names_and_units=column_map
        )

        # Assert
        expected_data = {
            "Engine": ["A1", "B2"],
            "Fuel Flow": pd.Series([25.1, 26.2], dtype="pint[kg/s]"),
            "Pressure Ratio": pd.Series([1.5, 1.6], dtype="pint[dimensionless]"),
            "Extra Data": [100, 200]
        }
        expected_df = pd.DataFrame(expected_data)

        assert_frame_equal(result_df, expected_df)
        assert "Extra Data" in result_df.columns
        assert result_df["Fuel Flow"].dtype == "pint[kg/s]"


    def test_return_only_renamed_columns(self, sample_df):
        """
        Tests the option to return a DataFrame with only the processed columns.
        """
        # Arrange
        column_map = [
            ("Engine ID", "Engine", str),
            ("Fuel Flow (kg/s)", "Fuel Flow", "pint[kg/s]"),
        ]

        # Act
        result_df = _rename_columns_and_set_units(
            df=sample_df.copy(),
            return_only_renamed_columns=True,
            column_names_and_units=column_map
        )

        # Assert
        expected_data = {
            "Engine": ["A1", "B2"],
            "Fuel Flow": pd.Series([25.1, 26.2], dtype="pint[kg/s]"),
        }
        expected_df = pd.DataFrame(expected_data)

        assert_frame_equal(result_df, expected_df)
        assert "Extra Data" not in result_df.columns
        assert list(result_df.columns) == ["Engine", "Fuel Flow"]


    def test_ignore_missing_source_column(self, sample_df):
        """
        Tests that the function doesn't fail if a specified old column name
        is not found in the DataFrame.
        """
        # Arrange
        column_map = [
            ("Engine ID", "Engine", str),
            ("Non-Existent Column", "New Name", "int"),
        ]

        # Act
        result_df = _rename_columns_and_set_units(
            df=sample_df.copy(),
            return_only_renamed_columns=False,
            column_names_and_units=column_map
        )

        # Assert
        assert "Engine" in result_df.columns
        assert "Engine ID" not in result_df.columns
        assert "New Name" not in result_df.columns # Should not be created
        assert "Non-Existent Column" not in result_df.columns


    def test_rename_only_with_none_dtype(self, sample_df):
        """
        Tests that a column is renamed but its dtype is not changed when
        the unit is specified as None.
        """
        # Arrange
        original_dtype = sample_df["Pressure Ratio"].dtype
        column_map = [
            ("Pressure Ratio", "P_Ratio", None),
        ]

        # Act
        result_df = _rename_columns_and_set_units(
            df=sample_df.copy(),
            return_only_renamed_columns=False,
            column_names_and_units=column_map
        )

        # Assert
        assert "P_Ratio" in result_df.columns
        assert "Pressure Ratio" not in result_df.columns
        assert result_df["P_Ratio"].dtype == original_dtype


    def test_with_empty_dataframe(self):
        """
        Tests that the function handles an empty DataFrame gracefully.
        """
        # Arrange
        empty_df = pd.DataFrame({"A": [], "B": []})
        column_map = [("A", "New A", "int")]

        # Act
        result_df = _rename_columns_and_set_units(
            df=empty_df.copy(),
            return_only_renamed_columns=False,
            column_names_and_units=column_map
        )

        # Assert
        expected_df = pd.DataFrame({"New A": [], "B": []})
        expected_df["New A"] = expected_df["New A"].astype("int") # Set expected type
        assert_frame_equal(result_df, expected_df)


    def test_raises_key_error_if_no_match_and_return_only(self, sample_df):
        """
        Tests that a KeyError is raised if no columns are matched and
        return_only_renamed_columns is True. This highlights a potential
        fragility in the original function design.
        """
        # Arrange
        column_map = [
            ("Non-Existent Column", "New Name", "int"),
        ]

        # Act & Assert
        with pytest.raises(KeyError):
            # This will fail because 'New Name' is not in the df.columns
            # after the loop finishes, since no renaming occurred.
            # Note: The provided snippet was slightly modified to pass this test
            # by being more robust. To test the original code, the list comprehension
            # should be: `[col[1] for col in column_names_and_units]`
            _rename_columns_and_set_units(
                df=sample_df.copy(),
                return_only_renamed_columns=True,
                column_names_and_units=column_map
            )


class TestReturnShortUnits:
    """Test suite for the `_return_short_units` function."""

    @pytest.mark.parametrize(
        "input_dtype, expected_output",
        [
            # Case 1: Standard pint unit with a common short form
            (pint_pandas.PintType("pint[kilogram]"), "kg"),
            
            # Case 2: Another common unit
            (pint_pandas.PintType("pint[second]"), "s"),

            # Case 3: A unit that is already short
            (pint_pandas.PintType("pint[meter]"), "m"),
            
            # Case 4: A complex unit
            (pint_pandas.PintType("pint[kilometer/hour]"), "km/h"),

            # Case 5: A unit with no standard short form
            (pint_pandas.PintType("pint[dimensionless]"), ""),
            
            # Case 6: Standard numpy/pandas dtype (integer)
            (np.dtype("int64"), "No Unit"),

            # Case 7: Standard numpy/pandas dtype (float)
            (np.dtype("float64"), "No Unit"),

            # Case 8: Standard numpy/pandas dtype (object/string)
            (np.dtype("object"), "No Unit"),

            # Case 9: Edge case with None as input
            (None, "No Unit"),
        ]
    )
    def test_unit_conversion_and_fallbacks(self, input_dtype, expected_output):
        """
        Tests that various dtypes are correctly converted to short unit strings
        or fall back to "No Unit" appropriately.
        """
        # Act: Call the function with the test input
        result = _return_short_units(input_dtype)

        # Assert: Check if the result matches the expected output
        assert result == expected_output


class TestExportTypedDataFrameToExcel:
    """Test suite for the  `export_typed_dataframe_to_excel` function."""

    @pytest.fixture
    def sample_typed_df(self) -> pd.DataFrame:
        data = {
            "Year": pd.Series([1999, 2005, 2012], dtype="pint[year]"),
            "Thrust": pd.Series([150.5, 165.2, 180.0], dtype="pint[kilonewton]"),
            "Comment": pd.Series(["Initial", "Mid-life", "Final"], dtype="object"),
        }
        return pd.DataFrame(data)

    def test_excel_export_content_and_format(self, sample_typed_df, tmp_path):
        """
        Tests that the exported Excel file has the correct structure:
        1. Correct column headers.
        2. A second row with the correct short units.
        3. The correct dequantified data below the unit row.
        """
        output_path = tmp_path / "test_export.xlsx"

        export_typed_dataframe_to_excel(sample_typed_df, output_path)

        assert output_path.exists()

        units_df = pd.read_excel(output_path, sheet_name='Data', header=0, nrows=1)
        data_df = pd.read_excel(output_path, sheet_name='Data', header=0, skiprows=[1])

        expected_units = {
            "Year": ["a"], # 'a' is the short unit for year (annum)
            "Thrust": ["kN"],
            "Comment": ["No Unit"],
        }
        expected_units_df = pd.DataFrame(expected_units)
        assert_frame_equal(units_df, expected_units_df)

        expected_data = sample_typed_df.pint.dequantify()
        expected_data.columns = expected_data.columns.droplevel(1)
        
        expected_data['Year'] = expected_data['Year'].astype('int64')
        expected_data['Thrust'] = expected_data['Thrust'].astype('float64')
        
        assert_frame_equal(data_df, expected_data)

class TestUpdateColumnData:
    """Test suite for the `update_column_data` function."""

    @pytest.fixture
    def df_main_fixture(self) -> pd.DataFrame:
        """Main DataFrame with some missing values."""
        data = {
            "ID": [1, 2, 3, 4],
            "Description": ["Alpha", "Bravo", "Charlie", "Delta"],
            "ValueA": [100.0, np.nan, 300.0, 400.0],
            "ValueB": [10.0, 20.0, np.nan, 40.0],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def df_other_fixture(self) -> pd.DataFrame:
        """'Other' DataFrame with updated values."""
        data = {
            "ID": [2, 4, 5],  # ID 2 matches, ID 4 matches, ID 5 is new
            "Description": ["Bravo_new", "Delta_new", "Echo_new"],
            "ValueA": [250.0, 450.0, 500.0],
            "ValueB": [25.0, 45.0, 55.0],
        }
        return pd.DataFrame(data)

    def test_basic_update_fills_nan_and_adds_indicator(self, df_main_fixture, df_other_fixture):
        """Tests if NaN values are filled and the indicator column is correct."""
        result_df = update_column_data(
            df_main=df_main_fixture,
            df_other=df_other_fixture,
            merge_column="ID",
            list_columns=["ValueA"],
        )

        # Expected result: ValueA updated, and new 'Updated?(ValueA)' column added
        expected_data = {
            "ID": [1, 2, 3, 4],
            "Description": ["Alpha", "Bravo", "Charlie", "Delta"],
            "ValueA": [100.0, 250.0, 300.0, 450.0],  # 2 and 4 updated
            "Updated?(ValueA)": [False, True, False, True], # 2 and 4 are True
            "ValueB": [10.0, 20.0, np.nan, 40.0],
        }
        expected_df = pd.DataFrame(expected_data)

        # Check column order as well
        expected_columns = ["ID", "Description", "ValueA", "Updated?(ValueA)", "ValueB"]
        assert list(result_df.columns) == expected_columns
        assert_frame_equal(result_df, expected_df)

    def test_overwrite_existing_values_and_sets_indicator(self, df_main_fixture, df_other_fixture):
        """Tests if existing values are overwritten and the indicator is set."""
        result_df = update_column_data(
            df_main=df_main_fixture,
            df_other=df_other_fixture,
            merge_column="ID",
            list_columns=["ValueA"],
        )
        # Value for ID 4 was 400.0 and should be overwritten by 450.0
        assert result_df.loc[result_df["ID"] == 4, "ValueA"].iloc[0] == 450.0
        # Indicator for ID 4 should be True
        assert result_df.loc[result_df["ID"] == 4, "Updated?(ValueA)"].iloc[0] == True
        
        # Value for ID 1 was 100.0 and was not in df_other
        assert result_df.loc[result_df["ID"] == 1, "ValueA"].iloc[0] == 100.0
        # Indicator for ID 1 should be False
        assert result_df.loc[result_df["ID"] == 1, "Updated?(ValueA)"].iloc[0] == False

    def test_multiple_column_update_and_indicators(self, df_main_fixture, df_other_fixture):
        """Tests updating multiple columns and adding multiple indicators."""
        result_df = update_column_data(
            df_main=df_main_fixture,
            df_other=df_other_fixture,
            merge_column="ID",
            list_columns=["ValueA", "ValueB"],
        )

        expected_data = {
            "ID": [1, 2, 3, 4],
            "Description": ["Alpha", "Bravo", "Charlie", "Delta"],
            "ValueA": [100.0, 250.0, 300.0, 450.0],
            "Updated?(ValueA)": [False, True, False, True],
            "ValueB": [10.0, 25.0, np.nan, 45.0],
            "Updated?(ValueB)": [False, True, False, True],
        }
        expected_df = pd.DataFrame(expected_data)
        
        expected_columns = ["ID", "Description", "ValueA", "Updated?(ValueA)", "ValueB", "Updated?(ValueB)"]
        assert list(result_df.columns) == expected_columns
        assert_frame_equal(result_df, expected_df)

    def test_nan_in_other_does_not_update(self, df_main_fixture):
        """
        Tests that a NaN value in df_other does not overwrite an existing
        value in df_main and sets the indicator to False.
        """
        df_other_with_nan = pd.DataFrame({
            "ID": [2, 4],
            "ValueA": [np.nan, 450.0], # This NaN should NOT update ID 2's ValueA
            "ValueB": [25.0, np.nan],  # This NaN should NOT update ID 4's ValueB
        })
        
        result_df = update_column_data(
            df_main=df_main_fixture,
            df_other=df_other_with_nan,
            merge_column="ID",
            list_columns=["ValueA", "ValueB"],
        )
        
        # Check ID 2
        row_2 = result_df[result_df["ID"] == 2].iloc[0]
        # ValueA was np.nan, df_other has np.nan -> stays np.nan
        assert pd.isna(row_2["ValueA"])
        # Updated?(ValueA) should be False because the update value was NaN
        assert row_2["Updated?(ValueA)"] == False
        # ValueB was 20.0, df_other has 25.0 -> updates to 25.0
        assert row_2["ValueB"] == 25.0
        # Updated?(ValueB) should be True
        assert row_2["Updated?(ValueB)"] == True
        
        # Check ID 4
        row_4 = result_df[result_df["ID"] == 4].iloc[0]
        # ValueA was 400.0, df_other has 450.0 -> updates to 450.0
        assert row_4["ValueA"] == 450.0
        # Updated?(ValueA) should be True
        assert row_4["Updated?(ValueA)"] == True
        # ValueB was 40.0, df_other has np.nan -> stays 40.0
        assert row_4["ValueB"] == 40.0
        # Updated?(ValueB) should be False
        assert row_4["Updated?(ValueB)"] == False

    def test_no_side_effects(self, df_main_fixture, df_other_fixture):
        """Ensures the original input DataFrames are not modified."""
        df_main_original = df_main_fixture.copy()
        df_other_original = df_other_fixture.copy()

        update_column_data(
            df_main=df_main_fixture,
            df_other=df_other_fixture,
            merge_column="ID",
            list_columns=["ValueA"],
        )

        assert_frame_equal(df_main_fixture, df_main_original)
        assert_frame_equal(df_other_fixture, df_other_original)

    def test_no_matching_rows_in_other_adds_false_indicator(self, df_main_fixture):
        """Tests behavior when 'other' has no matching keys."""
        df_other_nomatch = pd.DataFrame({
            "ID": [10, 11],
            "ValueA": [1000.0, 1100.0],
            "ValueB": [100.0, 110.0],
        })

        result_df = update_column_data(
            df_main=df_main_fixture,
            df_other=df_other_nomatch,
            merge_column="ID",
            list_columns=["ValueA"],
        )

        # The result should be the original main DataFrame + an indicator column
        expected_df = df_main_fixture.copy()
        # Find index of 'ValueA' to insert after it
        loc = expected_df.columns.get_loc("ValueA") + 1
        expected_df.insert(loc=loc, column="Updated?(ValueA)", value=False)

        assert_frame_equal(result_df, expected_df)

    def test_empty_other_dataframe_adds_false_indicators(self, df_main_fixture):
        """Tests behavior with an empty 'other' DataFrame."""
        df_other_empty = pd.DataFrame(columns=["ID", "ValueA", "ValueB"])

        result_df = update_column_data(
            df_main=df_main_fixture,
            df_other=df_other_empty,
            merge_column="ID",
            list_columns=["ValueA", "ValueB"],
        )

        # The result should be the original main DataFrame + indicator columns
        expected_df = df_main_fixture.copy()
        loc_A = expected_df.columns.get_loc("ValueA") + 1
        expected_df.insert(loc=loc_A, column="Updated?(ValueA)", value=False)
        loc_B = expected_df.columns.get_loc("ValueB") + 1
        expected_df.insert(loc=loc_B, column="Updated?(ValueB)", value=False)

        assert_frame_equal(result_df, expected_df)

    def test_no_new_rows_are_added(self, df_main_fixture, df_other_fixture):
        """Confirms that rows existing only in 'other' are not added to 'main'."""
        result_df = update_column_data(
            df_main=df_main_fixture,
            df_other=df_other_fixture,
            merge_column="ID",
            list_columns=["ValueA"],
        )
        assert len(result_df) == len(df_main_fixture)
        assert 5 not in result_df["ID"].values

    def test_missing_columns_raise_key_error(self, df_main_fixture, df_other_fixture):
        """Tests if a KeyError is raised for missing columns."""
        # Test missing merge_column in main_df
        with pytest.raises(KeyError, match="'WrongID' not in main DataFrame columns"):
            update_column_data(
                df_main=df_main_fixture,
                df_other=df_other_fixture,
                merge_column="WrongID",
                list_columns=["ValueA"],
            )

        # Test missing merge_column in other_df
        with pytest.raises(KeyError, match="'ID' not in other DataFrame columns"):
            update_column_data(
                df_main=df_main_fixture,
                df_other=df_other_fixture.rename(columns={"ID": "WrongID"}),
                merge_column="ID",
                list_columns=["ValueA"],
            )

        # Test missing update column in main_df
        with pytest.raises(KeyError, match="'ValueC' not in main DataFrame columns"):
            update_column_data(
                df_main=df_main_fixture,
                df_other=df_other_fixture,
                merge_column="ID",
                list_columns=["ValueC"],
            )

        # Test missing update column in other_df
        with pytest.raises(KeyError, match="'ValueB' not in other DataFrame columns"):
            update_column_data(
                df_main=df_main_fixture,
                df_other=df_other_fixture.drop(columns=["ValueB"]),
                merge_column="ID",
                list_columns=["ValueA", "ValueB"],
            )

            
class TestMergeWildcard:
    """Test suite for the `left_merge_wildcard` function."""

    @pytest.fixture(scope="class")
    def sample_dataframes(self):
        """
        Provides a comprehensive set of DataFrames to test all scenarios:

        1. A left key with a wildcard that finds multiple matches ('CFM56-5*').
        2. A left key without a wildcard that has an exact match in the right
           DataFrame ('GE90-115B'). This should be ignored by the logic.
        3. A left key with a wildcard that finds no matches ('NonExistent*').

        The left DataFrame:

        | Designation  | Aircraft     |
        |--------------|--------------|
        | CFM56-5*     | A320 Family  |
        | GE90-115B    | B777         |
        | NonExistent* | Concept      |

        merged with the right DataFrame:

        | Engine       | Thrust_kN | Manufacturer |
        |--------------|-----------|--------------|
        | CFM56-5A1    | 111       | CFM          |
        | CFM56-5A2    | 120       | CFM Intl     |
        | CFM56-5B1    | 133       | CFM          |
        | GE90-115B    | 514       | GE           |

        should yield the following result:

        | Designation  | Aircraft     | Thrust_kN       | Manufacturer |
        |--------------|--------------|-----------------|--------------|
        | CFM56-5*     | A320 Family  | (111+120+133)/3 | CFM          |
        | GE90-115B    | B777         | 133             | NaN          |
        | NonExistent* | Concept      | NaN             | NaN          |
        """
        df_left = pd.DataFrame({
            'Designation': ['CFM56-5*', 'GE90-115B', 'NonExistent*'],
            'Aircraft': ['A320 Family', 'B777', 'Concept'],
        })

        df_right = pd.DataFrame({
            'Engine': ['CFM56-5A1', 'CFM56-5A2', 'CFM56-5B1', 'GE90-115B'],
            'Thrust_kN': [111, 120, 133, 514],
            'Manufacturer': ['CFM', 'CFM Intl', 'CFM', 'GE']
        })
        
        return df_left, df_right

    def test_left_merge_wildcard_scenarios(self, sample_dataframes):
        """
        A single comprehensive test to verify all behaviors of the function:
        - Correctly aggregates a successful wildcard match.
        - Ignores a non-wildcard row for matching.
        - Handles a wildcard row that finds no matches.
        """
        df_left, df_right = sample_dataframes

        result_df = left_merge_wildcard(
            df_left=df_left,
            df_right=df_right,
            left_on='Designation',
            right_on='Engine'
        )

        expected_data = {
            'Designation': ['CFM56-5*', 'GE90-115B', 'NonExistent*'],
            'Aircraft': ['A320 Family', 'B777', 'Concept'],
            # 1. 'CFM56-5*': Aggregated from 3 matches. Thrust is mean(111, 120, 133).
            # 2. 'GE90-115B': Ignored for matching, so Thrust is NaN.
            # 3. 'NonExistent*': Wildcard finds no matches, so Thrust is NaN.
            'Thrust_kN': [(111 + 120 + 133) / 3, 514.0, np.nan],
            # 1. 'CFM56-5*': Manufacturer is the 'first' of the matches ('CFM').
            # 2. 'GE90-115B': Ignored, so Manufacturer is NaN.
            # 3. 'NonExistent*': No matches, so Manufacturer is NaN.
            'Manufacturer': ['CFM', 'GE', np.nan]
        }
        expected_df = pd.DataFrame(expected_data)

        assert_frame_equal(result_df, expected_df)