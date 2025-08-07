import pytest
import numpy as np
import numpy.testing as npt
from aircraftdetective.calculations.engines import (
    determine_takeoff_to_cruise_tsfc_ratio,
    scale_engine_data_from_icao_emissions_database
)
from typing import cast



def test_determine_takeoff_to_cruise_tsfc_ratio():
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


def test_scale_engine_data_from_icao_emissions_database():
    calibration = determine_takeoff_to_cruise_tsfc_ratio(
        path_excel_engine_data_for_calibration='tests/fixtures/fixture_tsfc_data.xlsx'
    )
    output_linear = scale_engine_data_from_icao_emissions_database(
        path_excel_engine_data_icao_in='tests/fixtures/fixture_tsfc_no_data.xlsx',
        path_excel_engine_data_icao_out=#tmppath,
        scaling_polynomial=calibration['pol_linear_fit']
    )
    pandas assert equal 
    