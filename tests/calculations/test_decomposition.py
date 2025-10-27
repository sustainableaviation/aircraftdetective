import pytest
import math
import pandas as pd
import pandas.testing as pd_testing
import numpy as np

from aircraftdetective.calculations.decomposition import (
    _compute_lmdi_factor_contributions,
    _compute_efficiency_improvement_metrics
)


@pytest.fixture
def sample_aircraft_data() -> pd.DataFrame:
    """Provides a standard, well-formed DataFrame for testing.
    
    | YOI | Type   | EU  | TSFC | OEW/Exit Limit | L/D | Other_Data |
    |-----|--------|-----|------|----------------|-----|------------|
    | 2005| Wide   | 100 | 0.8  | 320            | 15  | A          |
    | 2000| Narrow | 120 | 1.0  | 250            | 12  | B          |
    | 2010| Narrow | 108 | 0.9  | 240            | 13  | C          |
    | 2015| Wide   | 80  | 0.6  | 300            | 18  | D          |
    | 2008| Wide   | 90  | 0.7  | 310            | 16  | E          |
    """
    data = {
        'YOI':              [2005, 2000, 2010, 2015, 2008],
        'Type':             ['Wide', 'Narrow', 'Narrow', 'Wide', 'Wide'],
        'EU':               [100,   120,    108,    80,     90],
        'TSFC':             [0.8,   1.0,    0.9,    0.6,    0.7],
        'OEW/Exit Limit':   [320,   250,    240,    300,    310],
        'L/D':              [15,    12,     13,     18,     16],
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
            'TSFC': 'TSFC (cruise)',
        }
    )


class TestComputeImprovementMetrics:
    """
    Test suite for the _compute_efficiency_improvement_metrics function.
    """

    def test_calculations_with_sample_data(self, prepared_data):
        """
        Tests the core calculation logic using the prepared sample_aircraft_data.
        The function sorts by YOI, so the output order is YOI-based.
        
        Baselines:
        - Narrow (YOI 2000): EU=120, TSFC=1.0, OEW=250, L/D=12
        - Wide (YOI 2005):   EU=100, TSFC=0.8, OEW=320, L/D=15
        """
        result_df = _compute_efficiency_improvement_metrics(prepared_data)

        # Build the expected DataFrame, sorted by YOI (as the function does)
        expected_data = {
            # Original Data (YOI-sorted)
            'YOI':              [2000, 2005, 2008, 2010, 2015],
            'Type':             ['Narrow', 'Wide', 'Wide', 'Narrow', 'Wide'],
            'Energy Use (per ASK)': [120, 100, 90, 108, 80],
            'TSFC (cruise)':    [1.0, 0.8, 0.7, 0.9, 0.6],
            'OEW/Exit Limit':   [250, 320, 310, 240, 300],
            'L/D':              [12, 15, 16, 13, 18],
            'Other_Data':       ['B', 'A', 'E', 'C', 'D'],

            # Calculated Data
            'Index(Energy Use (per ASK))': [
                1.0,        # N-2000 (Base)
                1.0,        # W-2005 (Base)
                100/90,     # W-2008
                120/108,    # N-2010
                100/80      # W-2015
            ],
            'Percent(Energy Use (per ASK))': [
                0.0,
                0.0,
                (100/90 - 1) * 100,
                (120/108 - 1) * 100,
                (100/80 - 1) * 100
            ],
            
            'Index(TSFC (cruise))': [
                1.0,        # N-2000 (Base)
                1.0,        # W-2005 (Base)
                0.8/0.7,    # W-2008
                1.0/0.9,    # N-2010
                0.8/0.6     # W-2015
            ],
            'Percent(TSFC (cruise))': [
                0.0,
                0.0,
                (0.8/0.7 - 1) * 100,
                (1.0/0.9 - 1) * 100,
                (0.8/0.6 - 1) * 100
            ],
            
            'Index(OEW/Exit Limit)': [
                1.0,        # N-2000 (Base)
                1.0,        # W-2005 (Base)
                320/310,    # W-2008
                250/240,    # N-2010
                320/300     # W-2015
            ],
            'Percent(OEW/Exit Limit)': [
                0.0,
                0.0,
                (320/310 - 1) * 100,
                (250/240 - 1) * 100,
                (320/300 - 1) * 100
            ],
            
            'Index(L/D)': [
                1.0,        # N-2000 (Base)
                1.0,        # W-2005 (Base)
                16/15,      # W-2008
                13/12,      # N-2010
                18/15       # W-2015
            ],
            'Percent(L/D)': [
                0.0,
                0.0,
                (16/15 - 1) * 100,
                (13/12 - 1) * 100,
                (18/15 - 1) * 100
            ],
        }
        
        expected_df = pd.DataFrame(expected_data).reset_index(drop=True)
        result_df_sorted = result_df.sort_values(by='YOI').reset_index(drop=True)

        pd_testing.assert_frame_equal(result_df_sorted, expected_df, atol=1e-5)

    def test_original_dataframe_not_modified(self, prepared_data):
        """
        Ensures the function does not modify the input DataFrame in place.
        """
        original_copy = prepared_data.copy()
        _compute_efficiency_improvement_metrics(prepared_data)
        pd_testing.assert_frame_equal(prepared_data, original_copy)

    def test_preserves_other_columns(self, prepared_data):
        """
        Ensures that columns not used in the calculation are preserved.
        """
        result_df = _compute_efficiency_improvement_metrics(prepared_data)
        
        assert 'Other_Data' in result_df.columns
        # Check that the set of values is preserved, even if order changes
        assert set(result_df['Other_Data']) == set(prepared_data['Other_Data'])
        assert len(result_df) == len(prepared_data)

    def test_raises_on_missing_required_columns(self, sample_aircraft_data):
        """
        Tests that the function fails if the required column names are not present.
        This test *intentionally* uses the *un-prepared* fixture.
        """
        with pytest.raises(ValueError, match="Required column 'Energy Use \(per ASK\)' not found"):
            _compute_efficiency_improvement_metrics(sample_aircraft_data)

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
                    'YOI': [2000, 2005],
                    'Type': ['A', 'A'],
                    'Energy Use (per ASK)': [100.0, 80.0],
                    'TSFC (cruise)': [10.0, 8.0],
                    'OEW/Exit Limit': [50.0, 45.0],
                    'L/D': [np.nan, np.nan] # All NaN
                }),
                "Column 'L/D' cannot be all NaN",
                id="all_nan_column"
            ),
            pytest.param(
                pd.DataFrame({
                    'YOI': [2000, 2005],
                    'Type': ['A', 'A'],
                    'Energy Use (per ASK)': [100.0, 80.0],
                    'TSFC (cruise)': [10.0, 8.0],
                    'OEW/Exit Limit': [50.0, 45.0],
                    'L/D': [20.0, "twenty-two"] # Non-numeric
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
            _compute_efficiency_improvement_metrics(invalid_df)


class TestLmdiFactorContributions:
    """
    Test suite for the `_compute_lmdi_factor_contributions` function.
    """

    @pytest.mark.parametrize(
        "aggregate_t1, aggregate_t2, factor_t1, factor_t2, expected_contribution",
        [
            # Based on numerical example from Ang & Zhang (2000), Table 5.
            # The total change in the aggregate (50 - 20 = 30) is fully explained
            # by the change in the first factor.
            pytest.param(20.0, 50.0, 2.0, 5.0, 30.0, id="increase_with_changing_factor"),
            
            # When the factor itself does not change, its contribution must be zero.
            pytest.param(20.0, 50.0, 10.0, 10.0, 0.0, id="increase_with_constant_factor"),
            
            # This is the reverse of the first case. The total change is -30,
            # which should be fully explained by the factor's decrease.
            pytest.param(50.0, 20.0, 5.0, 2.0, -30.0, id="decrease_with_changing_factor"),
            
            # A more general case without round numbers.
            # L(150, 100) = (150-100) / ln(150/100) ≈ 123.25
            # Contribution = L * ln(4/2) ≈ 123.25 * ln(2) ≈ 85.45
            pytest.param(100.0, 150.0, 2.0, 4.0, (50 / math.log(1.5)) * math.log(2.0), id="general_increase_case"),
        ]
    )
    def test_valid_scenarios(
        self, aggregate_t1, aggregate_t2, factor_t1, factor_t2, expected_contribution
    ):
        """
        Tests the LMDI factor contribution calculation for various valid scenarios.
        """
        result = _compute_lmdi_factor_contributions(
            aggregate_t1, aggregate_t2, factor_t1, factor_t2
        )
        # Use pytest.approx for safe floating-point comparison
        assert result == pytest.approx(expected_contribution)