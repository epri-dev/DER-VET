"""
Copyright (c) 2024, Electric Power Research Institute

 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
     * Neither the name of DER-VET nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
"""
This file tests analysis cases that ONLY contain a SINGLE BATTERY. It is
organized by value stream combination and tests a bariety of optimization
horizons, time scale sizes, and other scenario options. All tests should pass.

The tests in this file can be run with DERVET and StorageVET, so make sure to
update TEST_PROGRAM with the lower case string name of the program that you
would like the tests to run on.

"""
import pytest
from pathlib import Path
from storagevet.ErrorHandling import *
from test.TestingLib import *


def setup_default_case(test_file):
    case = check_initialization(f'{test_file}{CSV}')

def infeasible_error(test_file):
    with pytest.raises(SolverInfeasibleError):
        results_instance = assert_ran(f'{test_file}{CSV}')

def timeseries_missing_error(test_file):
    with pytest.raises(TimeseriesMissingError):
        results_instance = assert_ran(f'{test_file}{CSV}')

def timeseries_data_error(test_file):
    with pytest.raises(TimeseriesDataError):
        results_instance = assert_ran(f'{test_file}{CSV}')

def run_default_case(test_file):
    results_instance = assert_ran(f'{test_file}{CSV}')


def test_default_asis():
    setup_default_case(DEFAULT_MP)
    run_default_case(DEFAULT_MP)

def test_default_ice_active():
    temp_mp = modify_mp('ICE')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_pv_active():
    temp_mp = modify_mp('PV')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_controllableload_active():
    temp_mp = modify_mp('ControllableLoad')
    temp_mp = modify_mp('Scenario', key='time_series_filename', value='./test/datasets/controllableload_001.csv', column='Optimization Value', mp_in=temp_mp, mp_out_tag='ControllableLoad')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_caes_active():
    temp_mp = modify_mp('CAES')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_ev1_active():
    temp_mp = modify_mp('ElectricVehicle1')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_ev2_active():
    temp_mp = modify_mp('ElectricVehicle2')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_chp_active():
    temp_mp = modify_mp('CHP')
    temp_mp = modify_mp('Scenario', key='time_series_filename', value='./test/datasets/thermal_001_nocoolingload.csv', column='Optimization Value', mp_in=temp_mp, mp_out_tag='CHP')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_chiller_active():
    temp_mp = modify_mp('Chiller')
    temp_mp = modify_mp('Scenario', key='time_series_filename', value='./test/datasets/thermal_001_noheatingloads.csv', column='Optimization Value', mp_in=temp_mp, mp_out_tag='Chiller')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_boiler_active():
    temp_mp = modify_mp('Boiler')
    temp_mp = modify_mp('Scenario', key='time_series_filename', value='./test/datasets/thermal_001_nocoolingload.csv', column='Optimization Value', mp_in=temp_mp, mp_out_tag='Boiler')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_electrolyzer_active():
    temp_mp = modify_mp('ElectrolyzerSystem')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_incl_ts_discharge_limits():
    # turn on boolean to include ts limits
    temp_mp = modify_mp('Battery', key='incl_ts_discharge_limits', value=1, column='Optimization Value')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_incl_ts_charge_limits():
    # turn on boolean to include ts limits
    temp_mp = modify_mp('Battery', key='incl_ts_charge_limits', value=1, column='Optimization Value')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_incl_ts_energy_limits():
    # turn on boolean to include ts limits
    temp_mp = modify_mp('Battery', key='incl_ts_energy_limits', value=1, column='Optimization Value')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_incl_ts_limits_off():
    # with incl_ts_*_limits all off, the max and min columns in the time series will be ignored, and this will run just fine
    temp_mp = modify_mp('Scenario', key='time_series_filename', value='./test/datasets/default_incl_ts_limits_infeasible.csv', column='Optimization Value')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_incl_ts_discharge_limits_infeasible():
    # should fail with SolverInfeasibleError
    temp_mp = modify_mp('Battery', key='incl_ts_discharge_limits', value=1, column='Optimization Value', mp_out_tag='discharge')
    temp_mp = modify_mp('Scenario', key='time_series_filename', value='./test/datasets/default_incl_ts_limits_infeasible.csv', column='Optimization Value', mp_in=temp_mp, mp_out_tag='discharge')
    setup_default_case(temp_mp)
    infeasible_error(temp_mp)
    remove_temp_files(temp_mp)

def test_default_incl_ts_charge_limits_infeasible():
    # should fail with SolverInfeasibleError
    temp_mp = modify_mp('Battery', key='incl_ts_charge_limits', value=1, column='Optimization Value', mp_out_tag='charge')
    temp_mp = modify_mp('Scenario', key='time_series_filename', value='./test/datasets/default_incl_ts_limits_infeasible.csv', column='Optimization Value', mp_in=temp_mp, mp_out_tag='charge')
    setup_default_case(temp_mp)
    infeasible_error(temp_mp)
    remove_temp_files(temp_mp)

def test_default_incl_ts_energy_limits_infeasible():
    # should fail with SolverInfeasibleError
    temp_mp = modify_mp('Battery', key='incl_ts_energy_limits', value=1, column='Optimization Value', mp_out_tag='energy')
    temp_mp = modify_mp('Scenario', key='time_series_filename', value='./test/datasets/default_incl_ts_limits_infeasible.csv', column='Optimization Value', mp_in=temp_mp, mp_out_tag='energy')
    setup_default_case(temp_mp)
    infeasible_error(temp_mp)
    remove_temp_files(temp_mp)

def test_default_incl_ts_energy_limits_missing_ts_columns():
    # should fail with TimeseriesMissingError
    temp_mp = modify_mp('Battery', key='incl_ts_energy_limits', value=1, column='Optimization Value', mp_out_tag='energy')
    temp_mp = modify_mp('Scenario', key='time_series_filename', value='./test/datasets/default_incl_ts_energy_limits_missing_ts_columns.csv', column='Optimization Value', mp_in=temp_mp, mp_out_tag='energy')
    timeseries_missing_error(temp_mp)
    remove_temp_files(temp_mp)

def test_default_incl_ts_energy_limits_bad_ts_data():
    # should fail with TimeseriesDataError
    temp_mp = modify_mp('Battery', key='incl_ts_energy_limits', value=1, column='Optimization Value', mp_out_tag='energy')
    temp_mp = modify_mp('Scenario', key='time_series_filename', value='./test/datasets/default_incl_ts_energy_limits_bad_ts_data.csv', column='Optimization Value', mp_in=temp_mp, mp_out_tag='energy')
    timeseries_data_error(temp_mp)
    remove_temp_files(temp_mp)

def test_default_incl_ts_charge_limits_bad_ts_data():
    # should fail with TimeseriesDataError
    temp_mp = modify_mp('Battery', key='incl_ts_charge_limits', value=1, column='Optimization Value', mp_out_tag='charge')
    temp_mp = modify_mp('Scenario', key='time_series_filename', value='./test/datasets/default_incl_ts_charge_limits_bad_ts_data.csv', column='Optimization Value', mp_in=temp_mp, mp_out_tag='charge')
    timeseries_data_error(temp_mp)
    remove_temp_files(temp_mp)
