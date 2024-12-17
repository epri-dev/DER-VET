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
import numpy as np
import numpy.testing as npt
from storagevet.ErrorHandling import *
from test.TestingLib import *
#import ipdb
#ipdb.set_trace()


DIR = Path("./test/test_electrolyzer/model_params")
JSON = '.json'
CSV = '.csv'
MAX_PERCENT_ERROR = 1e-7

def assert_om_costs_result(pro_forma, npv, ts, name, npv_discount_rate, rated_power, fixed_om_cost, variable_om_cost, efficiency, variable_efficiency=False):
    # NOTE: these tests assume a single optimization year
    name = name.lower()

    # fixed_om_cost
    actual_result = fixed_om_cost * rated_power * -1
    reported_result = pro_forma.loc[:,f'ELECTROLYZERSYSTEM: {name} Fixed O&M Cost'].iloc[1]
    error_msg = f'ValueError in pro_forma file: ELECTROLYZERSYSTEM: {name} Fixed O&M Cost'
    assert_within_error_bound(actual_result, reported_result, MAX_PERCENT_ERROR, error_msg)
    # npv
    # in the npv, the pro_forma value is reduced by npv_discount_rate: value / 1.07 ** (year - opt_year)
    npv_val = 0.0
    for i, val in enumerate(pro_forma.loc[:,f'ELECTROLYZERSYSTEM: {name} Fixed O&M Cost']):
        npv_val += (val / (1 + npv_discount_rate * 1e-2) ** i)
    actual_result = npv_val
    reported_result = npv.loc[:,f'ELECTROLYZERSYSTEM: {name} Fixed O&M Cost'].iloc[0]
    error_msg = f'ValueError in npv file: ELECTROLYZERSYSTEM: {name} Fixed O&M Cost'
    assert_within_error_bound(actual_result, reported_result, MAX_PERCENT_ERROR, error_msg)

    # variable_om_cost
    if variable_efficiency:
        power_times_efficiency_sum = (ts.loc[:,f'ELECTROLYZERSYSTEM: {name} Power (kW)'] * ts.loc[:,f'ELECTROLYZERSYSTEM: {name} Efficiency Actual (kg/kWh)']).sum()
        actual_result = variable_om_cost * power_times_efficiency_sum * -1
        error_msg = f'ValueError in pro_forma file: ELECTROLYZERSYSTEM: {name} Variable O&M Costs (with variable efficiency)'
    else:
        power_sum = ts.loc[:,f'ELECTROLYZERSYSTEM: {name} Power (kW)'].sum()
        actual_result = variable_om_cost * efficiency * power_sum * -1
        error_msg = f'ValueError in pro_forma file: ELECTROLYZERSYSTEM: {name} Variable O&M Costs'
    reported_result = pro_forma.loc[:,f'ELECTROLYZERSYSTEM: {name} Variable O&M Costs'].iloc[1]
    assert_within_error_bound(actual_result, reported_result, MAX_PERCENT_ERROR, error_msg)
    # npv
    # in the npv, the pro_forma value is reduced by npv_discount_rate: value / 1.07 ** (year - opt_year)
    npv_val = 0.0
    for i, val in enumerate(pro_forma.loc[:,f'ELECTROLYZERSYSTEM: {name} Variable O&M Costs']):
        npv_val += (val / (1 + npv_discount_rate * 1e-2) ** i)
    actual_result = npv_val
    reported_result = npv.loc[:,f'ELECTROLYZERSYSTEM: {name} Variable O&M Costs'].iloc[0]
    error_msg = f'ValueError in npv file: ELECTROLYZERSYSTEM: {name} Variable O&M Costs'
    assert_within_error_bound(actual_result, reported_result, MAX_PERCENT_ERROR, error_msg)

def assert_hydrogen_value_result(pro_forma, npv, ts, h2_unit_value, name, npv_discount_rate, variable_efficiency=False):
    # pro_forma
    # ElectrolyzerSystem H2 value should equal H2 production (from TS) * H2 unit value
    name = name.lower()
    if variable_efficiency:
        actual_result = ts.loc[:,f'ELECTROLYZERSYSTEM: {name} Hydrogen Production Actual (kg)'].sum() * h2_unit_value
    else:
        actual_result = ts.loc[:,f'ELECTROLYZERSYSTEM: {name} Hydrogen Production (kg)'].sum() * h2_unit_value
    reported_result = pro_forma.loc[:,f'ELECTROLYZERSYSTEM: {name} Hydrogen Value'].iloc[1]
    error_msg = f'ValueError in pro_forma file: ELECTROLYZERSYSTEM: {name} Hydrogen Value'
    assert_within_error_bound(actual_result, reported_result, MAX_PERCENT_ERROR, error_msg)
    # npv
    # in the npv, the pro_forma value is reduced by npv_discount_rate: value / 1.07 ** (year - opt_year)
    npv_val = 0.0
    for i, val in enumerate(pro_forma.loc[:,f'ELECTROLYZERSYSTEM: {name} Hydrogen Value']):
        npv_val += (val / (1 + npv_discount_rate * 1e-2) ** i)
    actual_result = npv_val
    reported_result = npv.loc[:,f'ELECTROLYZERSYSTEM: {name} Hydrogen Value'].iloc[0]
    error_msg = f'ValueError in npv file: ELECTROLYZERSYSTEM: {name} Hydrogen Value'
    assert_within_error_bound(actual_result, reported_result, MAX_PERCENT_ERROR, error_msg)

def assert_electrolyzer_matches_schedule(ts, name, input_ts_df):
    name = name.lower()
    power = ts.loc[:, f'ELECTROLYZERSYSTEM: {name} Power (kW)']
    schedule = input_ts_df.loc[:,'ElectrolyzerSystem Schedule (kW)']
    col1 = 'output power'
    col2 = 'input schedule'
    mod_ts = ts.copy()
    mod_ts.drop(columns=ts.columns, inplace=True)
    mod_ts[col1] = power
    mod_ts[col2] = schedule
    assert_timeseries_columns_equal(mod_ts, col1, col2)

def assert_electrolyzer_economic_feasibility(ts, h2_unit_value, name, efficiency):
    # test instances when the ElectrolyzerSystem is not scheduled
    # ElectrolyzerSystem only runs when value of hydrogen exceeds the value of electricity
    # ElectrolyzerSystem will not run when electricity is too expensive
    h2_value = h2_unit_value * efficiency
    name = name.lower()
    power = ts.loc[:, f'ELECTROLYZERSYSTEM: {name} Power (kW)']
    cheap_electricity = ts.loc[:, 'Energy Price ($/kWh)'] < h2_value
    expensive_electricity = ts.loc[:, 'Energy Price ($/kWh)'] >= h2_value
    if expensive_electricity.sum():
        assert 0.0 == power[expensive_electricity].max()

def assert_electrolyzer_power_is_within_limit(ts, name):
    # at all times, the ElectrolyzerSystem output power cannot be greater than
    #   the available power (from a battery, and/or from PV)
    #   but only when the POI constraint for no import is set (max_import=0)
    #   NOTE: Total Storage Power (kW) is always in the output TS
    name = name.lower()
    power = ts.loc[:, f'ELECTROLYZERSYSTEM: {name} Power (kW)']
    error_msg = f'The ELECTROLYZERSYSTEM: {name} Power (kW) cannot exceed the available power'
    available_power = ts.loc[:,'Total Generation (kW)'] + ts.loc[:,'Total Storage Power (kW)']
    col1 = 'ElectrolyzerSystem output power'
    col2 = 'total available power'
    mod_ts = ts.copy()
    mod_ts.drop(columns=ts.columns, inplace=True)
    mod_ts[col1] = available_power
    mod_ts[col2] = power
    assert_timeseries_col1_ge_col2(mod_ts, col1, col2)

def assert_hydrogen_production_quota_met(ts, name, h2_production_quota, opt_horizon):
    name = name.lower()
    h2_production = ts.loc[:, f'ELECTROLYZERSYSTEM: {name} Hydrogen Production (kg)']
    error_msg = f'The ELECTROLYZERSYSTEM: {name} Hydrogen Production must meet or exceed the hydrogen_production_quota within each optimization window: {opt_horizon}'
    ts_beg = h2_production.index[0]
    ts_end = h2_production.index[-1]
    ts_test = ts_beg
    if opt_horizon == 'month':
        # test monthly opt periods
        while ts_test < ts_end:
            indexes = (h2_production.index.year == ts_test.year) & (h2_production.index.month == ts_test.month)
            #i_beg = np.where(indexes)[0][0]
            i_end = np.where(indexes)[0][-1]
            if sum(indexes) > 0:
                assert sum(h2_production.loc[indexes]) >= h2_production_quota, error_msg
                #print(i_end, sum(h2_production.loc[indexes]))
            # move to start of next opt period, ending when the last index is reached
            if i_end+1 == len(indexes):
                ts_test = ts_end
            else:
                ts_test = h2_production.index[i_end+1]
    elif opt_horizon == 'year':
        # test yearly opt periods
        assert 1 == 0, f'the test with n={opt_horizon} has not yet been implemented'
    else:
        # test hourly opt periods
        assert 1 == 0, f'the test with n={opt_horizon} has not yet been implemented'

def assert_lcoh_in_payback_result(payback):
    # test that lcoh exists once in the payback result file
    # retrun the lcoh value
    lcoh_row = payback['Metric'] == 'Levelized Cost of Hydrogen'
    assert 1 == lcoh_row.sum()
    lcoh =  payback[lcoh_row]['Value'].values[0]
    return lcoh

def multiple_ez_case(test_file, number_of_ezs):
    results_instance = assert_ran(test_file)
    results = results_instance.instances[0]
    ts = results.time_series_data
    npv = results.cost_benefit_analysis.npv
    pro_forma = results.proforma_df()
    input_ts_df = get_input_ts_df(test_file)

    names = []
    for ez in range(number_of_ezs):
        names.append(get_value_mp('ElectrolyzerSystem', 'name', test_file, id_col=ez+1).lower())
    assert len(names) == number_of_ezs

    # lcoh (levelized cost of hydrogen) should exist in the payback results csv
    lcoh = assert_lcoh_in_payback_result(results.cost_benefit_analysis.payback)

    # check that there are number_of_ezs capacity factor values
    capacity_factors = []
    for ez in range(number_of_ezs):
        electrolyzer_row = results.capacity_factor_summary.Name.where(results.capacity_factor_summary.Name == names[ez].lower()).dropna().index
        capacity_factors.append(results.capacity_factor_summary.loc[electrolyzer_row[0], 'Capacity Factor'])
    assert len(capacity_factors) == number_of_ezs

    return lcoh, npv, names

def ez_case(test_file, schedule=0, variable_efficiency=False):
    results_instance = assert_ran(test_file)
    results = results_instance.instances[0]
    ts = results.time_series_data
    npv = results.cost_benefit_analysis.npv
    pro_forma = results.proforma_df()
    input_ts_df = get_input_ts_df(test_file)
    # params
    name = get_value_mp('ElectrolyzerSystem', 'name', test_file)
    h2_unit_value = float(get_value_mp('ElectrolyzerSystem', 'hydrogen_unit_value', test_file))
    efficiency = float(get_value_mp('ElectrolyzerSystem', 'efficiency', test_file))
    rated_power = float(get_value_mp('ElectrolyzerSystem', 'rated_power', test_file))
    min_power = float(get_value_mp('ElectrolyzerSystem', 'min_power', test_file))
    h2_schedule_mode = int(get_value_mp('ElectrolyzerSystem', 'hydrogen_schedule_mode', test_file))
    h2_production_quota = float(get_value_mp('ElectrolyzerSystem', 'hydrogen_production_quota', test_file))
    fixed_om_cost = float(get_value_mp('ElectrolyzerSystem', 'fixed_om_cost', test_file))
    variable_om_cost = float(get_value_mp('ElectrolyzerSystem', 'variable_om_cost', test_file))
    npv_discount_rate = float(get_value_mp('Finance', 'npv_discount_rate', test_file))
    opt_horizon = str(get_value_mp('Scenario', 'n', test_file))

    # these assertions are common across all ElectrolyzerSystem DER-VET cases
    assert_usecase_considered_services(results, ['DA'])

    # NOTE: the following test can fail if there is no CAPEX Year in the proforma
    #   that can happen if the construction year is on or after the start year.
    #   for all technologies present
    assert_om_costs_result(pro_forma, npv, ts, name, npv_discount_rate, rated_power, fixed_om_cost, variable_om_cost, efficiency, variable_efficiency=variable_efficiency)
    assert_hydrogen_value_result(pro_forma, npv, ts, h2_unit_value, name, npv_discount_rate, variable_efficiency=variable_efficiency)
    assert h2_schedule_mode == schedule
    # when the ElectrolyzerSystem is scheduled, the power values come directly from the input TS file
    if schedule == 1:
        # ensure that the ElectrolyzerSystem power matches the input scheduled power, at each timestep
        assert_electrolyzer_matches_schedule(ts, name, input_ts_df)
    elif schedule == 2:
        # ensure that the hydrogen production in each opt window meets or exceeds
        #   the value of the input parameter hydrogen_production_quota
        # NOTE: with variable_efficiency, these quotas may not be met,
        #       so we only test here with a constant efficiency
        if not variable_efficiency:
            assert_hydrogen_production_quota_met(ts, name, h2_production_quota, opt_horizon)
    else:   # schedule == 0
        # when the ElectrolyzerSystem is not scheduled
        #   the energy prices are considered, at each timestep
        assert_electrolyzer_economic_feasibility(ts, h2_unit_value, name, efficiency)

    # test for new columns in TS
    assert_timeseries_columns_exist(ts, [f'ELECTROLYZERSYSTEM: {name.lower()} Efficiency (kg/kWh)'])
    assert_timeseries_columns_exist(ts, [f'ELECTROLYZERSYSTEM: {name.lower()} Hydrogen Production (kg)'])
    assert_timeseries_columns_exist(ts, [f'ELECTROLYZERSYSTEM: {name.lower()} Original Load (kW)'])
    assert_timeseries_columns_exist(ts, [f'ELECTROLYZERSYSTEM: {name.lower()} Power (kW)'])
    assert_timeseries_columns_exist(ts, [f'ELECTROLYZERSYSTEM: {name.lower()} On (y/n)'])

    # at all times, the ElectrolyzerSystem output power cannot be greater than the rated_power
    assert round_to_n(ts.loc[:,f'ELECTROLYZERSYSTEM: {name.lower()} Power (kW)'].max(), 8) <= round_to_n(rated_power, 8)
    # when the ElectrolyzerSystem is ON, its output power must be at least min_power
    # do not test if the ElectrolyzerSystem never operates
    electrolyzer_on = ts.loc[:,f'ELECTROLYZERSYSTEM: {name.lower()} Power (kW)'] > 0
    if electrolyzer_on.sum() > 0:
        assert ts.loc[electrolyzer_on,f'ELECTROLYZERSYSTEM: {name.lower()} Power (kW)'].min() >= min_power

    # lcoh (levelized cost of hydrogen) should exist in the payback results csv
    lcoh = assert_lcoh_in_payback_result(results.cost_benefit_analysis.payback)

    # capacity factor
    electrolyzer_row = results.capacity_factor_summary.Name.where(results.capacity_factor_summary.Name == name.lower()).dropna().index
    capacity_factor = results.capacity_factor_summary.loc[electrolyzer_row[0], 'Capacity Factor']

    return ts, npv, pro_forma, capacity_factor, name, h2_unit_value, lcoh, efficiency

def test_ez_001_a():
    # NOTE: this test actually runs 3 DER-VET cases
    # POI constraints are OFF
    test_file = DIR / f'001-EZ-PV-BESS'
    ts, npv, pro_forma, capacity_factor, name, h2_unit_value, lcoh, efficiency = ez_case(f'{test_file}{CSV}')
    cf = capacity_factor

    # turn on POI constraints, no import
    temp_mp = modify_mp('Scenario', key='apply_interconnection_constraints', value=1, column='Optimization Value', mp_in=test_file, mp_out_tag='PoiOnNoImport')
    temp_mp = modify_mp('Scenario', key='max_import', value=0, column='Optimization Value', mp_in=temp_mp, mp_out_tag='PoiOnNoImport')
    temp_mp = modify_mp('Scenario', key='max_export', value=9e9, column='Optimization Value', mp_in=temp_mp, mp_out_tag='PoiOnNoImport')
    ts, npv, pro_forma, capacity_factor, name, h2_unit_value, lcoh, efficiency = ez_case(f'{temp_mp}{CSV}')
    assert_electrolyzer_power_is_within_limit(ts, name)
    cf_noImport = capacity_factor
    remove_temp_files(temp_mp)

    # compare the capacity factor for EZ in each of these 2 cases above
    # the ElectrolyzerSystem should operate more often without the no import POI constraint
    assert cf > cf_noImport, f'The ElectrolyzerSystem should operate more often (with a higher capacity factor) without the no Import POI constraint.'

    # turn on POI constraints, no import, no Battery
    temp_mp = modify_mp('Scenario', key='apply_interconnection_constraints', value=1, column='Optimization Value', mp_in=test_file, mp_out_tag='PoiOnNoImportNoBESS')
    temp_mp = modify_mp('Scenario', key='max_import', value=0, column='Optimization Value', mp_in=temp_mp, mp_out_tag='PoiOnNoImportNoBESS')
    temp_mp = modify_mp('Scenario', key='max_export', value=9e9, column='Optimization Value', mp_in=temp_mp, mp_out_tag='PoiOnNoImportNoBESS')
    temp_mp = modify_mp('Battery', key='name', value='no', column='Active', mp_in=temp_mp, mp_out_tag='PoiOnNoImportNoBESS')
    ts, npv, pro_forma, capacity_factor, name, h2_unit_value, lcoh, efficiency = ez_case(f'{temp_mp}{CSV}')
    cf_noImport_noBESS = capacity_factor
    assert_electrolyzer_power_is_within_limit(ts, name)
    remove_temp_files(temp_mp)

    # compare the capacity factor for EZ in each of these 2 noImport cases above
    # the ElectrolyzerSystem should operate less often when there is no BESS
    assert cf_noImport > cf_noImport_noBESS, f'The ElectrolyzerSystem should operate less often (with a lower capacity factor) without a BESS.'

def test_ez_001_b():
    test_file = DIR / f'001-EZ-PV-BESS'

    # turn on ElectrolyzerSystem schedule so that the EZ power is fixed
    temp_mp = modify_mp('ElectrolyzerSystem', key='hydrogen_schedule_mode', value=1, column='Optimization Value', mp_in=test_file, mp_out_tag='scheduled')
    ts, npv, pro_forma, capacity_factor, name, h2_unit_value, lcoh, efficiency = ez_case(f'{temp_mp}{CSV}', schedule=1)

    # extend this case by also turning on variable efficiency
    temp_mp = modify_mp('ElectrolyzerSystem', key='incl_variable_efficiency', value=1, column='Optimization Value', mp_in=temp_mp, mp_out_tag='scheduled')
    ts, npv, pro_forma, capacity_factor, name, h2_unit_value, lcoh, efficiency = ez_case(f'{temp_mp}{CSV}', schedule=1, variable_efficiency=True)
    assert_timeseries_columns_exist(ts, [f'ELECTROLYZERSYSTEM: {name.lower()} Efficiency Actual (kg/kWh)'])
    assert_timeseries_columns_exist(ts, [f'ELECTROLYZERSYSTEM: {name.lower()} Hydrogen Production Actual (kg)'])
    remove_temp_files(temp_mp)

def test_ez_001_c():
    test_file = DIR / f'001-EZ-PV-BESS'

    # set a hydrogen production quota for each optimization window
    # the sum of the hydrogen produced in each opt window must meet or exceed
    # the value of the input parameter hydrogen_production_quota (in kg)
    temp_mp = modify_mp('ElectrolyzerSystem', key='hydrogen_schedule_mode', value=2, column='Optimization Value', mp_in=test_file, mp_out_tag='quota1')
    ts, npv, pro_forma, capacity_factor, name, h2_unit_value, lcoh, efficiency = ez_case(f'{temp_mp}{CSV}', schedule=2)

    # now set the hydrogen_prodution_quota value too high, so it will cause an infeasibility
    # the code should catch this inut error as a ModelParameterError
    # for a non-leap year, with n=month, the min_opt_period_hours will be 28*24=672
    temp_mp = modify_mp('ElectrolyzerSystem', key='hydrogen_production_quota', value=6800, column='Optimization Value', mp_in=temp_mp, mp_out_tag='quota1')
    with pytest.raises(ModelParameterError):
        check_initialization(f'{temp_mp}{CSV}')
    remove_temp_files(temp_mp)

def test_ez_001_d():
    # POI constraints are OFF
    test_file = DIR / f'001-EZ-PV-BESS'

    # turn on POI constraints, no import, no BESS, set a non-zero min_power, binary ON
    temp_mp = modify_mp('Scenario', key='apply_interconnection_constraints', value=1, column='Optimization Value', mp_in=test_file, mp_out_tag='MinPowerConstraint')
    temp_mp = modify_mp('Scenario', key='max_import', value=0, column='Optimization Value', mp_in=temp_mp, mp_out_tag='MinPowerConstraint')
    temp_mp = modify_mp('Scenario', key='max_export', value=9e9, column='Optimization Value', mp_in=temp_mp, mp_out_tag='MinPowerConstraint')
    temp_mp = modify_mp('Battery', key='name', value='no', column='Active', mp_in=temp_mp, mp_out_tag='MinPowerConstraint')
    temp_mp = modify_mp('ElectrolyzerSystem', key='min_power', value=200, column='Optimization Value', mp_in=temp_mp, mp_out_tag='MinPowerConstraint')
    temp_mp = modify_mp('Scenario', key='binary', value=1, column='Optimization Value', mp_in=temp_mp, mp_out_tag='MinPowerConstraint')
    ts, npv, pro_forma, capacity_factor, name, h2_unit_value, lcoh, efficiency = ez_case(f'{temp_mp}{CSV}')
    # test min_power constraint (when the electrolyzersystem is ON, it must operate at a power of
    #     at least min_power, and at most rated_power)
    # also test to make sure that the electrolyzersystem is OFF when PV is less than min_power
    # NOTE: these tests are a part of every ez_case() run
    assert_electrolyzer_power_is_within_limit(ts, name)

    remove_temp_files(temp_mp)

def test_ez_002():
    test_file = DIR / f'002-EZ-PV-BESS-NoTAX'

    # tests that change the hydrogen_unit_value

    # set hydrogen_unit_value to zero
    # test that the ElectrolyzerSystem does not operate
    temp_mp = modify_mp('ElectrolyzerSystem', key='hydrogen_unit_value', value=0, column='Optimization Value', mp_in=test_file, mp_out_tag='lcoh0')
    ts, npv, pro_forma, capacity_factor, name, h2_unit_value, lcoh, efficiency = ez_case(f'{temp_mp}{CSV}')
    name = name.lower()
    power = ts.loc[:, f'ELECTROLYZERSYSTEM: {name} Power (kW)']
    assert 0 == power.sum()
    assert 0 == npv[f'ELECTROLYZERSYSTEM: {name} Variable O&M Costs']['NPV']
    remove_temp_files(temp_mp)

    # test that changing the hydrogen_unit_value has no affect on the LCOH result
    # (it's taken out of the equation in the code)
    # use a fixed h2 schedule for these tests

    # the default test case uses a hydrogen_unit_value of 3
    temp_mp = modify_mp('ElectrolyzerSystem', key='hydrogen_schedule_mode', value=1, column='Optimization Value', mp_in=test_file, mp_out_tag='lcoh1')
    ts, npv, pro_forma, capacity_factor, name, h2_unit_value, lcoh, efficiency = ez_case(f'{temp_mp}{CSV}', schedule=1)
    lcoh_default = lcoh
    remove_temp_files(temp_mp)

    # now set it to 0
    temp_mp = modify_mp('ElectrolyzerSystem', key='hydrogen_schedule_mode', value=1, column='Optimization Value', mp_in=test_file, mp_out_tag='lcoh2')
    temp_mp = modify_mp('ElectrolyzerSystem', key='hydrogen_unit_value', value=0, column='Optimization Value', mp_in=temp_mp, mp_out_tag='lcoh2')
    ts, npv, pro_forma, capacity_factor, name, h2_unit_value, lcoh, efficiency = ez_case(f'{temp_mp}{CSV}', schedule=1)
    lcoh_zero = lcoh
    remove_temp_files(temp_mp)

    # now set it to 9e4
    temp_mp = modify_mp('ElectrolyzerSystem', key='hydrogen_schedule_mode', value=1, column='Optimization Value', mp_in=test_file, mp_out_tag='lcoh3')
    temp_mp = modify_mp('ElectrolyzerSystem', key='hydrogen_unit_value', value=9e4, column='Optimization Value', mp_in=temp_mp, mp_out_tag='lcoh3')
    ts, npv, pro_forma, capacity_factor, name, h2_unit_value, lcoh, efficiency = ez_case(f'{temp_mp}{CSV}', schedule=1)
    lcoh_high = lcoh
    remove_temp_files(temp_mp)

    # the lcoh should be the same in all 3 of these runs, for a EZ+PV+BESS-NoTAX case
    error_msg = f'Changing the value of hydrogen (hydrogen_unit_value) should not affect the LCOH value'
    assert_within_error_bound(lcoh_default, lcoh_zero, MAX_PERCENT_ERROR, error_msg)
    assert_within_error_bound(lcoh_default, lcoh_high, MAX_PERCENT_ERROR, error_msg)

def test_ez_003_a():
    test_file = DIR / f'003-2EZs-PV-NoTAX'

    # tests that change the hydrogen_unit_value, with 2 EZs
    # the default test case uses a hydrogen_unit_value of 3, and 4
    lcoh, npv, names = multiple_ez_case(f'{test_file}{CSV}', 2)
    lcoh_default = lcoh
    # EZ1 operates at a lower power level over the year, compared to EZ2 (for this case)
    #   this means that the variable_om_costs for EZ2 will be a larger negative number
    assert npv[f'ELECTROLYZERSYSTEM: {names[0]} Variable O&M Costs']['NPV'] > npv[f'ELECTROLYZERSYSTEM: {names[1]} Variable O&M Costs']['NPV'], 'ez1 should have a larger Variable O&M cost than ez2'

    # now set the hydrogen unit values to 44 and 0
    temp_mp = modify_mp('ElectrolyzerSystem', key='hydrogen_unit_value', id_col=1, value=44, column='Optimization Value', mp_in=test_file, mp_out_tag='lcoh1')
    temp_mp = modify_mp('ElectrolyzerSystem', key='hydrogen_unit_value', id_col=2, value=0, column='Optimization Value', mp_in=temp_mp, mp_out_tag='lcoh1')
    lcoh, npv, names = multiple_ez_case(f'{temp_mp}{CSV}', 2)
    lcoh_1 = lcoh
    # EZ1 operates at a lower power level over the year, compared to EZ2 (for this case)
    #   this means that the variable_om_costs for EZ2 will be a larger negative number
    assert npv[f'ELECTROLYZERSYSTEM: {names[0]} Variable O&M Costs']['NPV'] > npv[f'ELECTROLYZERSYSTEM: {names[1]} Variable O&M Costs']['NPV'], 'ez1 should have a larger Variable O&M cost than ez2'
    remove_temp_files(temp_mp)

    # now set the hydrogen unit values to 8 and 4e7
    temp_mp = modify_mp('ElectrolyzerSystem', key='hydrogen_unit_value', id_col=1, value=8, column='Optimization Value', mp_in=test_file, mp_out_tag='lcoh2')
    temp_mp = modify_mp('ElectrolyzerSystem', key='hydrogen_unit_value', id_col=2, value=4e7, column='Optimization Value', mp_in=temp_mp, mp_out_tag='lcoh2')
    lcoh, npv, names = multiple_ez_case(f'{temp_mp}{CSV}', 2)
    lcoh_2 = lcoh
    # EZ1 operates at a lower power level over the year, compared to EZ2 (for this case)
    #   this means that the variable_om_costs for EZ2 will be a larger negative number
    assert npv[f'ELECTROLYZERSYSTEM: {names[0]} Variable O&M Costs']['NPV'] > npv[f'ELECTROLYZERSYSTEM: {names[1]} Variable O&M Costs']['NPV'], 'ez1 should have a larger Variable O&M cost than ez2'
    remove_temp_files(temp_mp)

    # the lcoh should be the same in all 3 of these runs, for a EZ+PV-NoTAX case
    error_msg = f'Changing the value of hydrogen (hydrogen_unit_value) should not affect the LCOH value'
    assert_within_error_bound(lcoh_default, lcoh_1, MAX_PERCENT_ERROR, error_msg)
    assert_within_error_bound(lcoh_default, lcoh_2, MAX_PERCENT_ERROR, error_msg)

def test_ez_003_b():
    test_file = DIR / f'003-2EZs-PV-NoTAX'

    # error should occur because each ElectrolyzerSystem should have a separate schedule input column in the time series
    temp_mp = modify_mp('Scenario', key='time_series_filename', value='.\\data\\hourly_timeseries_hydrogen.csv', column='Optimization Value', mp_in=test_file, mp_out_tag='missing-ts-column')
    with pytest.raises(TimeseriesMissingError):
        check_initialization(f'{temp_mp}{CSV}')
    remove_temp_files(temp_mp)
