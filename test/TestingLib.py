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
from dervet.DERVET import DERVET
import os
import pandas as pd
import numpy as np
import warnings
import math
from pathlib import Path

DIR = Path("./")
JSON = '.json'
CSV = '.csv'

DEFAULT_MP = DIR / f'Model_Parameters_Template_DER'
TEMP_MP = DIR / f'temp_model_parameters'

def round_to_n(x, n):
    # source: https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    round_to_n = lambda x, n: x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
    return round_to_n(x, n)

def _checkThatFileExists(f, name='Unlabeled', raise_exception_on_fail=True, write_msg_to_terminal=True):
    path_file = Path(f)
    if write_msg_to_terminal:
        msg = f'\n{name} file:\n  {path_file.resolve()}'
        print(msg)
    if not path_file.is_file():
        if raise_exception_on_fail:
            raise FileNotFoundError(f'\n\nFAIL: Your specified {name} file does not exist:\n{path_file.resolve()}\n')
        else:
            print(f'\n\nFAIL: Your specified {name} file does not exist:\n{path_file.resolve()}\n')
            return None
    return path_file

def run_case(model_param_location: str):
    print(f"Testing {model_param_location}...")
    # first make sure the model_param file exists
    model_param_file = _checkThatFileExists(Path(model_param_location), 'Model Parameter Input File')
    case = DERVET(model_param_file)
    results = case.solve()
    print(results.dir_abs_path)
    return results


def check_initialization(model_param_location: str):
    print(f"Testing {model_param_location}...")
    # first make sure the model_param file exists
    model_param_file = _checkThatFileExists(Path(model_param_location), 'Model Parameter Input File')
    case = DERVET(model_param_file)
    return case


def assert_file_exists(model_results, results_file_name='timeseries_results'):
    if model_results.sensitivity_df.empty:
        check_for_file = model_results.dir_abs_path / f'{results_file_name}{model_results.csv_label}.csv'
        print(check_for_file)
        assert os.path.isfile(check_for_file), f'No {results_file_name} found at {check_for_file}'
    else:
        for index in model_results.instances.keys():
            check_for_file = model_results.dir_abs_path / str(index) / f'{results_file_name}{model_results.csv_label}.csv'
            assert os.path.isfile(check_for_file), f'No {results_file_name} found at {check_for_file}'

def assert_file_does_not_exist(model_results, results_file_name: str):
    if model_results.sensitivity_df.empty:
        check_for_file = model_results.dir_abs_path / f'{results_file_name}{model_results.csv_label}.csv'
        assert os.path.isfile(check_for_file) == False, f'{results_file_name} found at {check_for_file}, but no file was expected'
    else:
        for index in model_results.instances.keys():
            check_for_file = model_results.dir_abs_path / str(index) / f'{results_file_name}{model_results.csv_label}.csv'
            assert os.path.isfile(check_for_file) == False, f'{results_file_name} found at {check_for_file} but no file was expected'

def assert_within_error_bound(expected_value: float, actual_value: float, error_bound: float, error_message: str):
    if error_bound < 0:
        raise Exception(f'Testing Fail: the specified error_bound ({error_bound}) must be a positive number')
    diff = abs(expected_value-actual_value)
    #print(*[expected_value, actual_value, diff, '----'], sep='\n')
    if not diff:
        # the difference in values is zero
        return
    assert diff/abs(expected_value) <= error_bound/100, f'Test value: {actual_value}   Should be in range: ({expected_value+(expected_value*(error_bound/100))},{expected_value-(expected_value*(error_bound/100))}). {error_message}'


##########################################


def assert_ran(model_param_location: str):
    results = run_case(model_param_location)
    assert_file_exists(results) # assert that timeseries_results.csv file exists
    return results


def assert_ran_with_services(model_param_location: str, services: list):
    results = run_case(model_param_location)
    assert_file_exists(results) # assert that timeseries_results.csv file exists
    value_stream_keys = results.instances[0].service_agg.value_streams.keys()
    print(set(value_stream_keys))
    assert set(services) == set(value_stream_keys)


def assert_usecase_considered_services(results, services: list):
    value_stream_keys = results.instances[0].service_agg.value_streams.keys()
    print(set(value_stream_keys))
    assert set(services) == set(value_stream_keys)

def assert_timeseries_columns_exist(ts, items: list):
    timeseries_columns = ts.columns
    assert set(items).issubset(timeseries_columns)

def assert_timeseries_columns_do_not_exist(ts, items: list):
    timeseries_columns = ts.columns
    assert not set(items).intersection(timeseries_columns)

def assert_timeseries_columns_are_all_zeroes(ts, items: list):
    ts_subset = ts[items]
    max_tf = ts_subset.max().between(-1e-5, 1e-5)
    min_tf = ts_subset.min().between(-1e-5, 1e-5)
    print(max_tf, min_tf)
    assert all(max_tf) & all(min_tf)

def assert_timeseries_columns_have_positive_values(ts, items: list):
    ts_subset = ts[items]
    max_tf = ts_subset.max().gt(1e-5)
    assert all(max_tf)

def assert_timeseries_columns_have_negative_values(ts, items: list):
    ts_subset = ts[items]
    min_tf = ts_subset.min().gt(1e-5)
    assert all(min_tf)

def assert_timeseries_columns_equal(ts, col1: str, col2: str):
    # across all times
    ts_diff = ts[col1] - ts[col2]
    assert ts_diff.abs().max() < 1e-5

def assert_timeseries_columns_not_equal(ts, col1: str, col2: str):
    # across all times
    ts_diff = ts[col1] - ts[col2]
    assert ts_diff.abs().max() > 1e-5

def assert_timeseries_col1_ge_col2(ts, col1: str, col2: str):
    # across all times
    ts_diff = ts[col1] - ts[col2]
    #print(ts_diff.describe())
    assert all(ts_diff.ge(-1e-5))

def assert_expected_number_of_timeseries_columns(ts, number_of_expected_timeseries_columns: int):
    number_of_actual_timeseries_columns = len(ts.columns)
    assert number_of_actual_timeseries_columns == number_of_expected_timeseries_columns, f'{number_of_actual_timeseries_columns} columns found in the timeseries results file, but expecting {number_of_expected_timeseries_columns}'

def assert_npv_columns_are_zero(npv, items: list):
    # npv only has one row
    npv_subset = npv[items]
    tf = npv_subset.max().between(-1e-5, 1e-5)
    assert all(tf)

def assert_npv_columns_are_negative(npv, items: list):
    # npv only has one row; negative represents a cost
    npv_subset = npv[items]
    tf = npv_subset.max().lt(0)
    assert all(tf)

def assert_pro_forma_columns_are_all_zeroes(pf, items: list):
    pf_subset = pf[items]
    max_tf = pf_subset.max().between(-1e-5, 1e-5)
    min_tf = pf_subset.min().between(-1e-5, 1e-5)
    assert all(max_tf) & all(min_tf)

def assert_pro_forma_columns_are_all_negative(pf, items: list):
    pf_subset = pf[items]
    max_tf = pf_subset.max().lt(0)
    min_tf = pf_subset.min().lt(0)
    assert all(max_tf) & all(min_tf)

def assert_expected_number_of_results_files(results, number_of_expected_results_files: int):
    # NOTE: with a .csv model parameters input, there are two model parameter files output to results
    #       with a .json model parameters input, there is one model parameter file output to results
    results_files_generator_object = results.dir_abs_path.glob('*')
    number_of_actual_results_files = len([k for k in results_files_generator_object])
    assert number_of_actual_results_files == number_of_expected_results_files, f'{number_of_actual_results_files} files found in results folder, but expecting {number_of_expected_results_files}'

def compare_proforma_results(results, frozen_proforma_location: str, error_bound: float, opt_years=None):
    if isinstance(results, pd.DataFrame):
        actual_proforma_df = frozen_proforma_location
    else:
        assert_file_exists(results, 'pro_forma') # assert that pro_forma.csv file exists
        actual_proforma_df = results.proforma_df()
    if isinstance(frozen_proforma_location, pd.DataFrame):
        expected_df = frozen_proforma_location
    else:
        try:
            expected_df = pd.read_csv(frozen_proforma_location, index_col='Unnamed: 0')
        except ValueError:
            expected_df = pd.read_csv(frozen_proforma_location, index_col='Year')
    for yr_indx, values_series in expected_df.iterrows():
        print(f'\nPROFORMA YEAR: {yr_indx}\n')
        try:
            actual_indx = pd.Period(yr_indx)
            if opt_years is not None and actual_indx.year not in opt_years:
                continue
        except ValueError:
            actual_indx = yr_indx
        assert actual_indx in actual_proforma_df.index, f'{actual_indx} not in test proforma index'
        for col_indx in values_series.index:
            # NOTE: this loops through expected columns (extra columns appearing in the actual
            #       proforma file are ignored)
            print(col_indx)
            assert col_indx in actual_proforma_df.columns, f'{col_indx} not in test proforma columns'
            error_message = f'ValueError in Proforma [{yr_indx}, {col_indx}]\n'
            print(expected_df.loc[yr_indx, col_indx], actual_proforma_df.loc[actual_indx, col_indx])
            assert_within_error_bound(expected_df.loc[yr_indx, col_indx], actual_proforma_df.loc[actual_indx, col_indx], error_bound, error_message)

def compare_npv_results(results, frozen_npv_location: str, error_bound: float, opt_years=None):
    if isinstance(results, pd.DataFrame):
        actual_npv_df = results
    else:
        assert_file_exists(results, 'npv')  # assert that npv.csv file exists
        actual_npv_df = results.instances[0].cost_benefit_analysis.npv
    if isinstance(frozen_npv_location, pd.DataFrame):
        expected_df = frozen_npv_location
    else:
        try:
            expected_df = pd.read_csv(frozen_npv_location, index_col='Unnamed: 0')
        except ValueError:
            expected_df = pd.read_csv(frozen_npv_location, index_col='Year')
    for yr_indx, values_series in expected_df.iterrows():
        print(f'\n{yr_indx}:\n')
        try:
            actual_indx = pd.Period(yr_indx)
            if opt_years is not None and actual_indx.year not in opt_years:
                continue
        except ValueError:
            actual_indx = yr_indx
        assert actual_indx in actual_npv_df.index, f'{actual_indx} not in test npv index'
        for col_indx in values_series.index:
            # NOTE: this loops through expected columns (extra columns appearing in the actual
            #       npv file are ignored)
            print(col_indx)
            assert col_indx in actual_npv_df.columns, f'{col_indx} not in test npv columns'
            error_message = f'ValueError in NPV [{yr_indx}, {col_indx}]\n'
            print(expected_df.loc[yr_indx, col_indx], actual_npv_df.loc[actual_indx, col_indx])
            assert_within_error_bound(expected_df.loc[yr_indx, col_indx], actual_npv_df.loc[actual_indx, col_indx], error_bound, error_message)

def check_lcpc(results, test_model_param_location, expected_target_hours=None):
    # this test compares the covered hours from the internal results data (drill_down)
    #   with what is expected given the target hours from the model parameters file
    actual_lcpc_pd = results.instances[0].drill_down_dict['load_coverage_prob']
    # get max # of hours that value of lcpc is 100
    actual_covered_hrs = sum(actual_lcpc_pd['Load Coverage Probability (%)'] == 100)
    # get target hours
    if expected_target_hours is None:
        # when expected_target_hours is None, get the value from model parameters csv
        case_mp_pd = pd.read_csv(test_model_param_location)
        try:
            target_covered_hours = case_mp_pd.loc[(case_mp_pd['Tag'] == 'Reliability') & (case_mp_pd['Key'] == 'target'), 'Value']
        except KeyError:
            target_covered_hours = case_mp_pd.loc[(case_mp_pd['Tag'] == 'Reliability') & (case_mp_pd['Key'] == 'target'), 'Optimization Value']
        target_covered_hours = int(target_covered_hours.values[0])
    else:
        target_covered_hours = int(expected_target_hours)
    print(actual_covered_hrs, target_covered_hours)
    assert target_covered_hours <= actual_covered_hrs, f'Hours covered: {actual_covered_hrs}\nExpected: {target_covered_hours}'


def compare_size_results(results, frozen_size_location: str, error_bound: float):
    assert_file_exists(results, 'size')  # assert that size.csv file exists
    actual_df = results.instances[0].sizing_df
    try:
        actual_df.set_index("DER", inplace=True)
    except KeyError:
        pass
    expected_df = pd.read_csv(frozen_size_location, index_col='DER')
    for der_name in expected_df.index:
        print(f'\nDER SIZING: {der_name}\n')
        for col in expected_df.columns:
            # NOTE: this loops through expected columns (extra columns appearing in the actual
            #       size results file are ignored)
            print(col)
            actual_value = actual_df.loc[der_name, col]
            expected_value = expected_df.loc[der_name, col]
            print(actual_value, expected_value)
            if str(actual_value) != 'nan' and str(expected_value) != 'nan':
                error_message = f'ValueError in [{der_name}, {col}]\nExpected: {expected_value}\nGot: {actual_value}'

                assert_within_error_bound(expected_value, actual_value, error_bound, error_message)


def compare_lcpc_results(results, frozen_lcpc_location: str, error_bound: float):
    # this test compares each row of the load coverage probability data
    #   (from the internal results data (drill_down) with the load_coverage_prob
    #   file in results (frozen)
    actual_df = results.instances[0].drill_down_dict.get('load_coverage_prob')
    assert actual_df is not None
    expected_df = pd.read_csv(frozen_lcpc_location, index_col='Outage Length (hrs)')
    for time_step in expected_df.index:
        # loop through keys in expected dataframe
        actual_value = actual_df.loc[time_step]['Load Coverage Probability (%)']
        expected_value = expected_df.loc[time_step]['Load Coverage Probability (%)']
        if actual_value != 'nan' and expected_value != 'nan':
            error_message = f'ValueError in [{time_step}]\nExpected: {expected_value}\nGot: {actual_value}'

            assert_within_error_bound(expected_value, actual_value, error_bound, error_message)


def modify_mp(tag, key='name', value='yes', column='Active', mp_in=DEFAULT_MP, mp_out_tag=None, id_col=None):
    # read in default MP, modify it, write it to a temp file
    mp = pd.read_csv(f'{mp_in}{CSV}')
    indexes = (mp.Tag == tag) & (mp.Key == key)
    if id_col is not None:
        indexes = (indexes) & (mp.ID == str(id_col))
    indexes = indexes[indexes].index.values
    if len(indexes) != 1:
        raise Exception(f'a unique row from the default model parameters cannot be determined (tag: {tag}, key: {key}')
    mp_cell = (indexes[0], column)
    mp.loc[mp_cell] = value
    if mp_out_tag is None:
        tempfile_name = f'{TEMP_MP}--{tag}'
    else:
        tempfile_name = f'{TEMP_MP}--{mp_out_tag}'
    mp.to_csv(f'{tempfile_name}{CSV}', index=False)
    return tempfile_name


def get_value_mp(tag, key, mp_in, id_col=None):
    # returns the optimization value (as a String) from one row in the input model parameters file
    column = 'Optimization Value'
    mp = pd.read_csv(mp_in)
    indexes = (mp.Tag == tag) & (mp.Key == key)
    if id_col is not None:
        indexes = (indexes) & (mp.ID == str(id_col))
    indexes = indexes[indexes].index.values
    if len(indexes) != 1:
        raise Exception(f'a unique row from the default model parameters cannot be determined (tag: {tag}, key: {key}')
    mp_cell = (indexes[0], column)
    return mp.loc[mp_cell]


def get_input_ts_df(model_params_file):
    # returns a dataframe with the input timeseries, from the CSV described in the model parameters file
    input_ts_file = get_value_mp('Scenario', 'time_series_filename', model_params_file)
    # replace any backslashes with forward slash
    input_ts_file = input_ts_file.replace('\\', '/')
    # suppress the UserWarning about date format parsing here
    # we want a flexible date parser, and do not need to warn
    # message='Could not infer format, so each element will be parsed individually, falling back
    #   to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format'
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore',
            category=UserWarning,
        )
        input_ts_df = pd.read_csv(input_ts_file, parse_dates=True, index_col='Datetime (he)')
    # account for hour-beginning vs hour-ending difference
    #   the input ts to dervet is he (starts at hour 01),
    #   while the output ts from dervet is hb (starts at hour 00)
    # shift the input ts data back an hour so that it begins with hour 00
    input_ts_df.index -= pd.Timedelta('1h')
    return input_ts_df


def remove_temp_files(temp_mp):
    Path(f'{temp_mp}{CSV}').unlink()
    Path(f'{temp_mp}{JSON}').unlink()
