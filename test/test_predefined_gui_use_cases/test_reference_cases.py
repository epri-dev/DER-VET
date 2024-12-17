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
Various tests for the 4 pre-defined use cases that are defaults in the GUI
EXPECTED_DIR contains the results folder from each pre-defined case.
These frozen expected results are compared against actual results which are
run when this test file is initiated with pytest.

"""
from pathlib import Path
from test.TestingLib import *
import pytest

MAX_PERCENT_ERROR = 1e-4
EXPECTED_DIR = Path("./test/test_predefined_gui_use_cases")

USECASE_FOLDER = {
    1: Path("./der_for_bill_reduction"),
    2: Path("./der_for_reliability"),
    3: Path("./caiso_market_case"),
    4: Path("./ev_battery_sizing"),
}

USECASE_RESULTS_FILES_COUNT = {
    1: 24,
    2: 25,
    3: 18,
    4: 22,
}

USECASE_TIMESERIES_COLUMNS_COUNT = {
    1: 19,
    2: 23,
    3: 26,
    4: 20,
}


class TestUseCase1:
    # DER_for_Bill_Reduction
    def setup_class(self):
        use_case_number = 1
        self.actual_results = run_case(EXPECTED_DIR / USECASE_FOLDER[use_case_number] / 'inputs' / 'model_parameters.json')
        self.expected_results_folder = EXPECTED_DIR / USECASE_FOLDER[use_case_number] / 'results'
        self.ts = self.actual_results.instances[0].time_series_data
        self.results_files_count = USECASE_RESULTS_FILES_COUNT[use_case_number]
        self.timeseries_columns_count = USECASE_TIMESERIES_COLUMNS_COUNT[use_case_number]

    def test_number_of_results_files_is_expected(self):
        assert_expected_number_of_results_files(self.actual_results, self.results_files_count)

    def test_number_of_timeseries_columns_is_expected(self):
        assert_expected_number_of_timeseries_columns(self.ts, self.timeseries_columns_count)

    def test_npv_results_are_expected(self):
        compare_npv_results(self.actual_results, self.expected_results_folder / "npv.csv",
                                 MAX_PERCENT_ERROR)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.actual_results, self.expected_results_folder / "pro_forma.csv",
                                 MAX_PERCENT_ERROR)

    def test_size_results_are_expected(self):
        compare_size_results(self.actual_results, self.expected_results_folder / "size.csv",
                             MAX_PERCENT_ERROR)

    def test_lcpc_exists(self):
        # this output file is created only when we optimize for reliability
        assert_file_exists(self.actual_results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.actual_results, None, expected_target_hours=0)


class TestUseCase2:
    # DER_for_Reliability
    def setup_class(self):
        use_case_number = 2
        self.actual_results = run_case(EXPECTED_DIR / USECASE_FOLDER[use_case_number] / 'inputs' / 'model_parameters.json')
        self.expected_results_folder = EXPECTED_DIR / USECASE_FOLDER[use_case_number] / 'results'
        self.ts = self.actual_results.instances[0].time_series_data
        self.results_files_count = USECASE_RESULTS_FILES_COUNT[use_case_number]
        self.timeseries_columns_count = USECASE_TIMESERIES_COLUMNS_COUNT[use_case_number]

    def test_number_of_results_files_is_expected(self):
        assert_expected_number_of_results_files(self.actual_results, self.results_files_count)

    def test_number_of_timeseries_columns_is_expected(self):
        assert_expected_number_of_timeseries_columns(self.ts, self.timeseries_columns_count)

    def test_npv_results_are_expected(self):
        compare_npv_results(self.actual_results, self.expected_results_folder / "npv.csv",
                                 MAX_PERCENT_ERROR)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.actual_results, self.expected_results_folder / "pro_forma.csv",
                                 MAX_PERCENT_ERROR)

    def test_size_results_are_expected(self):
        compare_size_results(self.actual_results, self.expected_results_folder / "size.csv",
                             MAX_PERCENT_ERROR)

    def test_lcpc_exists(self):
        # this output file is created only when we optimize for reliability
        assert_file_exists(self.actual_results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.actual_results, None, expected_target_hours=4)


class TestUseCase3:
    # CAISO_Market_Case
    def setup_class(self):
        use_case_number = 3
        self.actual_results = run_case(EXPECTED_DIR / USECASE_FOLDER[use_case_number] / 'inputs' / 'model_parameters.json')
        self.expected_results_folder = EXPECTED_DIR / USECASE_FOLDER[use_case_number] / 'results'
        self.ts = self.actual_results.instances[0].time_series_data
        self.results_files_count = USECASE_RESULTS_FILES_COUNT[use_case_number]
        self.timeseries_columns_count = USECASE_TIMESERIES_COLUMNS_COUNT[use_case_number]

    def test_number_of_results_files_is_expected(self):
        assert_expected_number_of_results_files(self.actual_results, self.results_files_count)

    def test_number_of_timeseries_columns_is_expected(self):
        assert_expected_number_of_timeseries_columns(self.ts, self.timeseries_columns_count)

    def test_npv_results_are_expected(self):
        compare_npv_results(self.actual_results, self.expected_results_folder / "npv.csv",
                                 MAX_PERCENT_ERROR)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.actual_results, self.expected_results_folder / "pro_forma.csv",
                                 MAX_PERCENT_ERROR)

    def test_size_results_are_expected(self):
        compare_size_results(self.actual_results, self.expected_results_folder / "size.csv",
                             MAX_PERCENT_ERROR)

    def test_lcpc_does_not_exist(self):
        # this output file is created only when we optimize for reliability
        assert_file_does_not_exist(self.actual_results, 'load_coverage_prob')


class TestUseCase4:
    # EV_Battery_Sizing
    def setup_class(self):
        use_case_number = 4
        self.actual_results = run_case(EXPECTED_DIR / USECASE_FOLDER[use_case_number] / 'inputs' / 'model_parameters.json')
        self.expected_results_folder = EXPECTED_DIR / USECASE_FOLDER[use_case_number] / 'results'
        self.ts = self.actual_results.instances[0].time_series_data
        self.results_files_count = USECASE_RESULTS_FILES_COUNT[use_case_number]
        self.timeseries_columns_count = USECASE_TIMESERIES_COLUMNS_COUNT[use_case_number]

    def test_number_of_results_files_is_expected(self):
        assert_expected_number_of_results_files(self.actual_results, self.results_files_count)

    def test_number_of_timeseries_columns_is_expected(self):
        assert_expected_number_of_timeseries_columns(self.ts, self.timeseries_columns_count)

    def test_npv_results_are_expected(self):
        compare_npv_results(self.actual_results, self.expected_results_folder / "npv.csv",
                                 MAX_PERCENT_ERROR)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.actual_results, self.expected_results_folder / "pro_forma.csv",
                                 MAX_PERCENT_ERROR)

    def test_size_results_are_expected(self):
        compare_size_results(self.actual_results, self.expected_results_folder / "size.csv",
                             MAX_PERCENT_ERROR)

    def test_lcpc_does_not_exist(self):
        # this output file is created only when we optimize for reliability
        assert_file_does_not_exist(self.actual_results, 'load_coverage_prob')
