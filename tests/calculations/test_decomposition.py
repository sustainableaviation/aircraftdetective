import pytest
import math

# Import the function from its source module
from aircraftdetective.calculations.decomposition import (
    _compute_lmdi_factor_contributions
)


class TestLmdiFactorContributions:
    """
    Groups tests for the `_compute_lmdi_factor_contributions` function.
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

    def test_no_change_in_aggregate(self):
        """
        Tests the specific edge case where the aggregate value is unchanged.
        The function should return 0.0 immediately.
        """
        result = _compute_lmdi_factor_contributions(
            aggregate_t1=100.0, aggregate_t2=100.0, factor_t1=5.0, factor_t2=10.0
        )
        assert result == 0.0

    @pytest.mark.parametrize(
        "aggregate_t1, aggregate_t2, factor_t1, factor_t2, expected_error",
        [
            # Zero values that lead to undefined mathematical operations.
            pytest.param(10.0, 20.0, 0.0, 5.0, ZeroDivisionError, id="zero_factor_t1_division"),
            pytest.param(0.0, 20.0, 2.0, 5.0, ValueError, id="zero_aggregate_t1_log"),
            pytest.param(10.0, 0.0, 2.0, 5.0, ValueError, id="zero_aggregate_t2_log"),
            pytest.param(10.0, 20.0, 2.0, 0.0, ValueError, id="zero_factor_t2_log"),
            
            # Negative values are not permissible in logarithms.
            pytest.param(-10.0, 20.0, 2.0, 5.0, ValueError, id="negative_aggregate_t1"),
            pytest.param(10.0, -20.0, 2.0, 5.0, ValueError, id="negative_aggregate_t2"),
            pytest.param(10.0, 20.0, -2.0, 5.0, ValueError, id="negative_factor_ratio"),
        ]
    )
    def test_invalid_inputs(
        self, aggregate_t1, aggregate_t2, factor_t1, factor_t2, expected_error
    ):
        """
        Tests that the function raises appropriate errors for invalid inputs like zeros
        or negative numbers, which are mathematically undefined for this formula.
        """
        with pytest.raises(expected_error):
            _compute_lmdi_factor_contributions(
                aggregate_t1, aggregate_t2, factor_t1, factor_t2
            )