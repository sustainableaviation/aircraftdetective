import pytest
import pandas as pd
import numpy as np
import pint_pandas
from pathlib import Path
from pandas.testing import assert_frame_equal
from aircraftdetective.utility.tabular import (
    rename_columns_and_set_units,
    _return_short_units,
    export_typed_dataframe_to_excel
)

@pytest.fixture
def sample_df() -> pd.DataFrame:
    data = {
        "Engine ID": ["A1", "B2"],
        "Fuel Flow (kg/s)": [25.1, 26.2],
        "Pressure Ratio": [1.5, 1.6],
        "Extra Data": [100, 200]
    }
    return pd.DataFrame(data)


class TestRenameColumnsAndSetUnits:
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
        result_df = rename_columns_and_set_units(
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
        result_df = rename_columns_and_set_units(
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
        result_df = rename_columns_and_set_units(
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
        result_df = rename_columns_and_set_units(
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
        result_df = rename_columns_and_set_units(
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
            rename_columns_and_set_units(
                df=sample_df.copy(),
                return_only_renamed_columns=True,
                column_names_and_units=column_map
            )


class TestReturnShortUnits:
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


@pytest.fixture
def sample_typed_df() -> pd.DataFrame:
    data = {
        "Year": pd.Series([1999, 2005, 2012], dtype="pint[year]"),
        "Thrust": pd.Series([150.5, 165.2, 180.0], dtype="pint[kilonewton]"),
        "Comment": pd.Series(["Initial", "Mid-life", "Final"], dtype="object"),
    }
    return pd.DataFrame(data)


class TestExportTypedDataFrameToExcel:
    """
    Groups all tests for the export_typed_dataframe_to_excel function.
    """

    def test_excel_export_content_and_format(self, sample_typed_df, tmp_path):
        """
        Tests that the exported Excel file has the correct structure:
        1. Correct column headers.
        2. A second row with the correct short units.
        3. The correct dequantified data below the unit row.
        """
        # --- Arrange ---
        # Define the path for the output file inside pytest's temp directory
        output_path = tmp_path / "test_export.xlsx"

        # --- Act ---
        # Call the function to create the Excel file
        export_typed_dataframe_to_excel(sample_typed_df, output_path)

        # --- Assert ---
        # 1. Check that the file was actually created
        assert output_path.exists()

        # 2. Read the file back in to verify its contents.
        # We need to read the headers and data separately.
        # Read the unit row (the first data row in the file, index 0)
        units_df = pd.read_excel(output_path, sheet_name='Data', header=0, nrows=1)
        # Read the actual data, skipping the unit row
        data_df = pd.read_excel(output_path, sheet_name='Data', header=0, skiprows=[1])

        # 3. Verify the units row
        expected_units = {
            "Year": ["a"], # 'a' is the short unit for year (annum)
            "Thrust": ["kN"],
            "Comment": ["No Unit"],
        }
        expected_units_df = pd.DataFrame(expected_units)
        assert_frame_equal(units_df, expected_units_df)

        # 4. Verify the data portion
        # Create the expected dequantified data frame
        expected_data = sample_typed_df.pint.dequantify()
        expected_data.columns = expected_data.columns.droplevel(1)
        
        # FIX: Cast columns to match how pandas reads them from Excel
        expected_data['Year'] = expected_data['Year'].astype('int64')
        expected_data['Thrust'] = expected_data['Thrust'].astype('float64')
        
        assert_frame_equal(data_df, expected_data)