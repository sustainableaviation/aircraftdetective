import pytest
import math
import pandas as pd
import pandas.testing as pd_testing
import numpy as np

# Updated imports to include all functions
from aircraftdetective.calculations.decomposition import (
    compute_efficiency_improvement_metrics,
    compute_lmdi_factor_contributions,
    _compute_lmdi_factor_contributions_vectorized,
    compute_efficiency_disaggregation
)


@pytest.fixture
def sample_aircraft_data() -> pd.DataFrame:
    """Provides a standard, well-formed DataFrame for testing.
    
    Now includes Year, EI, and SLF.
    """
    data = {
        'Year':             [2005, 2000, 2010, 2015, 2008],
        'Type':             ['Wide', 'Narrow', 'Narrow', 'Wide', 'Wide'],
        'EU':               [100,    120,    108,    80,     90],
        'EI':               [125,    150,    130,    90,     110], # New
        'TSFC':             [0.8,    1.0,    0.9,    0.6,    0.7],
        'OEW/Exit Limit':   [320,    250,    240,    300,    310],
        'L/D':              [15,     12,     13,     18,     16],
        'SLF':              [0.8,    0.8,    0.83,   0.89,   0.82], # New
        'Other_Data':       ['A', 'B', 'C', 'D', 'E'] # To ensure other columns are preserved
    }
    return pd.DataFrame(data)


@pytest.fixture
def prepared_data(sample_aircraft_data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw sample_aircraft_data and renames columns
    to match the function's requirements.
    """
    return sample_aircraft_data.rename(
        columns={
            'EU': 'Energy Use (per ASK)',
            'EI': 'Energy Intensity (per RPK)',
            'TSFC': 'TSFC (cruise)',
        }
    )


class TestComputeImprovementMetrics:
    """
    Test suite for the compute_efficiency_improvement_metrics function.
    """

    def test_calculations_with_sample_data(self, prepared_data):
        """
        Tests the core calculation logic using the prepared sample_aircraft_data.
        The function sorts by Year, so the output order is Year-based.
        
        Baselines:
        - Narrow (Year 2000): EU=120, EI=150, TSFC=1.0, OEW=250, L/D=12, SLF=0.8
        - Wide (Year 2005):   EU=100, EI=125, TSFC=0.8, OEW=320, L/D=15, SLF=0.8
        """
        result_df = compute_efficiency_improvement_metrics(prepared_data)

        # Build the expected DataFrame, sorted by Year (as the function does)
        expected_data = {
            # Original Data (Year-sorted)
            'Year':                     [2000, 2005, 2008, 2010, 2015],
            'Type':                     ['Narrow', 'Wide', 'Wide', 'Narrow', 'Wide'],
            'Energy Use (per ASK)':     [120, 100, 90, 108, 80],
            'Energy Intensity (per RPK)':[150, 125, 110, 130, 90],
            'TSFC (cruise)':            [1.0, 0.8, 0.7, 0.9, 0.6],
            'OEW/Exit Limit':           [250, 320, 310, 240, 300],
            'L/D':                      [12, 15, 16, 13, 18],
            'SLF':                      [0.8, 0.8, 0.82, 0.83, 0.89],
            'Other_Data':               ['B', 'A', 'E', 'C', 'D'],

            # Calculated Data - EU (lower is better)
            'Index(EU)': [1.0, 1.0, 100/90, 120/108, 100/80],
            'Percent(EU)': [0.0, 0.0, (100/90 - 1) * 100, (120/108 - 1) * 100, (100/80 - 1) * 100],
            
            # Calculated Data - EI (lower is better)
            'Index(EI)': [1.0, 1.0, 125/110, 150/130, 125/90],
            'Percent(EI)': [0.0, 0.0, (125/110 - 1) * 100, (150/130 - 1) * 100, (125/90 - 1) * 100],

            # Calculated Data - Engines (TSFC, lower is better)
            'Index(Engines)': [1.0, 1.0, 0.8/0.7, 1.0/0.9, 0.8/0.6],
            'Percent(Engines)': [0.0, 0.0, (0.8/0.7 - 1) * 100, (1.0/0.9 - 1) * 100, (0.8/0.6 - 1) * 100],
            
            # Calculated Data - Weight (OEW, lower is better)
            'Index(Weight)': [1.0, 1.0, 320/310, 250/240, 320/300],
            'Percent(Weight)': [0.0, 0.0, (320/310 - 1) * 100, (250/240 - 1) * 100, (320/300 - 1) * 100],

            # Calculated Data - Aerodynamics (L/D, higher is better)
            'Index(Aerodynamics)': [1.0, 1.0, 16/15, 13/12, 18/15],
            'Percent(Aerodynamics)': [0.0, 0.0, (16/15 - 1) * 100, (13/12 - 1) * 100, (18/15 - 1) * 100],

            # Calculated Data - Operations (SLF, higher is better)
            'Index(Operations)': [1.0, 1.0, 0.82/0.8, 0.83/0.8, 0.89/0.8],
            'Percent(Operations)': [0.0, 0.0, (0.82/0.8 - 1) * 100, (0.83/0.8 - 1) * 100, (0.89/0.8 - 1) * 100],
        }
        
        expected_df = pd.DataFrame(expected_data).reset_index(drop=True)
        result_df_sorted = result_df.sort_values(by='Year').reset_index(drop=True)

        pd_testing.assert_frame_equal(result_df_sorted, expected_df, atol=1e-5)

    def test_original_dataframe_not_modified(self, prepared_data):
        """
        Ensures the function does not modify the input DataFrame in place.
        """
        original_copy = prepared_data.copy()
        compute_efficiency_improvement_metrics(prepared_data)
        pd_testing.assert_frame_equal(prepared_data, original_copy)

    def test_preserves_other_columns(self, prepared_data):
        """
        Ensures that columns not used in the calculation are preserved.
        """
        result_df = compute_efficiency_improvement_metrics(prepared_data)
        
        assert 'Other_Data' in result_df.columns
        assert set(result_df['Other_Data']) == set(prepared_data['Other_Data'])
        assert len(result_df) == len(prepared_data)

    def test_raises_on_missing_required_columns(self, sample_aircraft_data):
        """
        Tests that the function fails if the required column names are not present.
        This test *intentionally* uses the *un-prepared* fixture.
        """
        # 'Energy Use (per ASK)' is one of the renamed columns
        with pytest.raises(ValueError, match="Required column 'Energy Use \(per ASK\)' not found"):
            compute_efficiency_improvement_metrics(sample_aircraft_data)

    @pytest.mark.parametrize(
        "invalid_df, match_message",
        [
            pytest.param(
                None,
                "must be a non-empty Pandas DataFrame",
                id="not_a_dataframe"
            ),
            pytest.param(
                pd.DataFrame(),
                "must be a non-empty Pandas DataFrame",
                id="empty_dataframe"
            ),
            pytest.param(
                pd.DataFrame({
                    'Year': [2000, 2005], 'Type': ['A', 'A'],
                    'Energy Use (per ASK)': [100.0, 80.0],
                    'Energy Intensity (per RPK)': [100.0, 80.0],
                    'TSFC (cruise)': [10.0, 8.0],
                    'OEW/Exit Limit': [50.0, 45.0],
                    'L/D': [20.0, 22.0],
                    'SLF': [np.nan, np.nan] # All NaN
                }),
                "Column 'SLF' cannot be all NaN",
                id="all_nan_column"
            ),
            pytest.param(
                pd.DataFrame({
                    'Year': [2000, 2005], 'Type': ['A', 'A'],
                    'Energy Use (per ASK)': [100.0, 80.0],
                    'Energy Intensity (per RPK)': [100.0, 80.0],
                    'TSFC (cruise)': [10.0, 8.0],
                    'OEW/Exit Limit': [50.0, 45.0],
                    'L/D': [20.0, "twenty-two"], # Non-numeric
                    'SLF': [0.8, 0.9]
                }),
                "Column 'L/D' must be of a numeric type",
                id="non_numeric_metric"
            ),
        ]
    )
    def test_raises_value_error_for_invalid_input(self, invalid_df, match_message):
        """
        Tests all validation checks that should raise a ValueError.
        """
        with pytest.raises(ValueError, match=match_message):
            compute_efficiency_improvement_metrics(invalid_df)


class TestLmdiFactorContributions:
    """
    Test suite for the `compute_lmdi_factor_contributions` (scalar) function.
    No changes needed here, as the function logic is stable.
    """

    @pytest.mark.parametrize(
        "aggregate_t1, aggregate_t2, factor_t1, factor_t2, expected_contribution",
        [
            pytest.param(20.0, 50.0, 2.0, 5.0, 30.0, id="increase_with_changing_factor"),
            pytest.param(20.0, 50.0, 10.0, 10.0, 0.0, id="increase_with_constant_factor"),
            pytest.param(50.0, 20.0, 5.0, 2.0, -30.0, id="decrease_with_changing_factor"),
            pytest.param(100.0, 150.0, 2.0, 4.0, (50 / math.log(1.5)) * math.log(2.0), id="general_increase_case"),
            pytest.param(100.0, 100.0, 2.0, 4.0, 100.0 * math.log(2.0), id="no_aggregate_change_case"),
        ]
    )
    def test_valid_scenarios(
        self, aggregate_t1, aggregate_t2, factor_t1, factor_t2, expected_contribution
    ):
        """
        Tests the LMDI factor contribution calculation for various valid scenarios.
        """
        result = compute_lmdi_factor_contributions(
            aggregate_t1, aggregate_t2, factor_t1, factor_t2
        )
        assert result == pytest.approx(expected_contribution)


class TestLmdiFactorContributionsVectorized:
    """
    Test suite for the `_compute_lmdi_factor_contributions_vectorized` function.
    """
    
    def test_vectorized_calculations(self):
        """
        Tests the vectorized LMDI calculation logic.
        """
        # Scenarios from the scalar test
        agg_t1 = pd.Series([20.0, 20.0, 50.0, 100.0, 100.0])
        agg_t2 = pd.Series([50.0, 50.0, 20.0, 150.0, 100.0])
        fact_t1 = pd.Series([2.0, 10.0, 5.0, 2.0, 2.0])
        fact_t2 = pd.Series([5.0, 10.0, 2.0, 4.0, 4.0])
        
        expected = pd.Series([
            30.0,
            0.0,
            -30.0,
            (50 / math.log(1.5)) * math.log(2.0),
            100.0 * math.log(2.0)
        ])
        
        result = _compute_lmdi_factor_contributions_vectorized(
            agg_t1, agg_t2, fact_t1, fact_t2
        )
        pd_testing.assert_series_equal(result, expected, atol=1e-5)

    def test_vectorized_handles_scalar_and_series(self):
        """
        Tests that the function works when t1 is scalar and t2 is a Series.
        """
        agg_t1 = 100.0 # Scalar
        agg_t2 = pd.Series([150.0, 100.0])
        fact_t1 = 2.0   # Scalar
        fact_t2 = pd.Series([4.0, 4.0])
        
        expected = pd.Series([
            (50 / math.log(1.5)) * math.log(2.0),
            100.0 * math.log(2.0)
        ])
        
        result = _compute_lmdi_factor_contributions_vectorized(
            agg_t1, agg_t2, fact_t1, fact_t2
        )
        pd_testing.assert_series_equal(result, expected, atol=1e-5)

    def test_vectorized_handles_zeros(self):
        """
        Tests that rows where t1=t2 (e.g., baseline) correctly return 0.0.
        """
        agg_t1 = pd.Series([100.0, 50.0])
        agg_t2 = pd.Series([100.0, 75.0]) # First row has no change
        fact_t1 = pd.Series([10.0, 2.0])
        fact_t2 = pd.Series([10.0, 3.0]) # First row has no change
        
        expected_contribution = (75 - 50) / math.log(75/50) * math.log(3/2)
        expected = pd.Series([0.0, expected_contribution])
        
        result = _compute_lmdi_factor_contributions_vectorized(
            agg_t1, agg_t2, fact_t1, fact_t2
        )
        pd_testing.assert_series_equal(result, expected, atol=1e-5)


@pytest.fixture
def sample_index_data() -> pd.DataFrame:
    """
    Provides a DataFrame as it would be *after*
    compute_efficiency_improvement_metrics has run.
    This is the input for compute_efficiency_disaggregation.
    
    Baseline (2010): EU=1.0, EI=1.0, Eng=1.0, Wgt=1.0, Aero=1.0, Ops=1.0
    2015: EU=1.25, EI=1.5, Eng=1.1, Wgt=1.05, Aero=1.0897, Ops=1.1
    """
    # 1.1 * 1.05 * 1.0897... ~= 1.25 (for EU)
    # 1.1 * 1.05 * 1.0897 * 1.1... ~= 1.5 (for EI)
    # This data is manually constructed to be (approximately) consistent
    e_2015 = 1.1
    w_2015 = 1.05
    a_2015 = 1.25 / (e_2015 * w_2015) # ~1.0897
    o_2015 = 1.1
    ei_2015 = e_2015 * w_2015 * a_2015 * o_2015 # Should be 1.25 * 1.1 = 1.375
    
    data = {
        'Year': [2010, 2015],
        'Index(EU)': [1.0, 1.25],
        'Index(EI)': [1.0, ei_2015],
        'Index(Engines)': [1.0, e_2015],
        'Index(Weight)': [1.0, w_2015],
        'Index(Aerodynamics)': [1.0, a_2015],
        'Index(Operations)': [1.0, o_2015]
    }
    return pd.DataFrame(data)


class TestComputeEfficiencyDisaggregation:
    """
    Test suite for the compute_efficiency_disaggregation function.
    """
    
    def test_disaggregation_calculations_and_columns(self, sample_index_data):
        """
        Tests that the correct output columns are created.
        """
        result_df = compute_efficiency_disaggregation(sample_index_data)
        
        expected_cols = [
            'ContributionEU(Engines)',
            'ContributionEU(Weight)',
            'ContributionEU(Aerodynamics)',
            'ContributionEI(Engines)',
            'ContributionEI(Weight)',
            'ContributionEI(Aerodynamics)',
            'ContributionEI(Operations)'
        ]
        
        for col in expected_cols:
            assert col in result_df.columns
            assert pd.api.types.is_numeric_dtype(result_df[col])
            
        # The first row (baseline) must have 0.0 contribution
        for col in expected_cols:
            assert result_df.iloc[0][col] == pytest.approx(0.0)

    def test_disaggregation_sum_equals_total_change(self, sample_index_data):
        """
        Tests the core LMDI property: Sum(contributions) == Total Change
        Total Change = Index(t2) - Index(t1)
        """
        result_df = compute_efficiency_disaggregation(sample_index_data)
        
        # Get the non-baseline row (Year 2015)
        row = result_df.iloc[1]
        
        # --- Check EU ---
        total_change_eu = row['Index(EU)'] - sample_index_data.iloc[0]['Index(EU)']
        sum_contributions_eu = (
            row['ContributionEU(Engines)'] +
            row['ContributionEU(Weight)'] +
            row['ContributionEU(Aerodynamics)']
        )
        assert total_change_eu == pytest.approx(sum_contributions_eu)
        assert total_change_eu == pytest.approx(1.25 - 1.0) # Sanity check

        # --- Check EI ---
        total_change_ei = row['Index(EI)'] - sample_index_data.iloc[0]['Index(EI)']
        sum_contributions_ei = (
            row['ContributionEI(Engines)'] +
            row['ContributionEI(Weight)'] +
            row['ContributionEI(Aerodynamics)'] +
            row['ContributionEI(Operations)']
        )
        assert total_change_ei == pytest.approx(sum_contributions_ei)
        assert total_change_ei == pytest.approx(1.375 - 1.0) # Sanity check

    def test_disaggregation_validation(self, sample_index_data):
        """
        Tests the validation logic for the disaggregation function.
        """
        # Test missing column
        with pytest.raises(ValueError, match="Required column 'Index\(EU\)' not found"):
            compute_efficiency_disaggregation(
                sample_index_data.drop(columns=['Index(EU)'])
            )
            
        # Test non-positive (zero) value
        sample_index_data.loc[1, 'Index(Engines)'] = 0.0
        with pytest.raises(ValueError, match="All Index columns must contain positive values"):
            compute_efficiency_disaggregation(sample_index_data)

        # Test non-numeric Year
        sample_index_data.loc[1, 'Year'] = "2015_str"
        with pytest.raises(ValueError, match="Column 'Year' must be of a numeric type"):
            compute_efficiency_disaggregation(sample_index_data)