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
This file tests End to end tests passed on the validation report on September
1st. All tests should pass.

The tests in this file can be run with DERVET.

"""
from pathlib import Path
from test.TestingLib import *
import pytest

PCT_ERROR_PROFORMA = 1e-4
PCT_ERROR_SIZE = 1e-5
TEST_DIR = Path("./test/test_validation_report_sept1")

USECASE1 = Path("./Model_params/Usecase1")


class TestUseCase1EssSizing4Btm:
    """ 1ESS sizing - BTM with PF reliability calculations"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE1 / "Model_Parameters_Template_Usecase1_UnPlanned_ES.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase1/es")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def xtest_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3.csv",
                                 PCT_ERROR_PROFORMA)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             PCT_ERROR_SIZE)


class TestUseCase1EssSizingPv4Btm:
    """ 1ESS sizing, 1PV fixed - BTM with PF reliability calculations"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE1 / "Model_Parameters_Template_Usecase1_UnPlanned_ES+PV.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase1/es+pv")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def xtest_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3.csv",
                                 PCT_ERROR_PROFORMA)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             PCT_ERROR_SIZE)


class TestUseCase1EssPvIce4UserConstraints:
    """ Part 2 of Usecase 1C - 1ESS fixed, 1PV fixed, 1ICE fixed - user constraints with PF
    reliability calculations"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE1 / "Model_Parameters_Template_Usecase1_UnPlanned_ES+PV+DG_step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase1/es+pv+dg_step2")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3.csv",
                                 PCT_ERROR_PROFORMA)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             PCT_ERROR_SIZE)


USECASE2 = Path("./Model_params/Usecase2")


class TestUseCase2EssSizing4Reliability:
    """ Part 1 of Usecase 2A - 1ESS - size for just reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / "Model_Parameters_Template_Usecase3_Planned_ES.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es/step1")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3_es_step1.csv",
                                 PCT_ERROR_PROFORMA)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es_step1.csv", .1)


class TestUseCase2EssSizing4ReliabilityRTS:
    """ Part 1b of Usecase 2A - 1ESS - size for just reliability with RTS"""
    # this includes an optimization service (RTS) to properly test that the minimum energy requirement
    # is getting applied as a constraint (check_lcpc does this)
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / "Model_Parameters_Template_Usecase3_Planned_ES_RTS.csv"
        self.results = run_case(self.mp_name)

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)


class TestUseCase2Ess4BtmUserConstraints:
    """ Part 2 of Usecase 2A - 1 ESS - given size , bill reduction and user constraint with PF
    reliability"""

    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / \
                       "Model_Parameters_Template_Usecase3_Planned_ES_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es/step2")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3_es_step2.csv",
                                 PCT_ERROR_PROFORMA, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es_step2.csv",
                             PCT_ERROR_SIZE)


class TestUsecase2EssSizingPv4Reliability:
    """ Part 1 of Usecase 2B - BAT sized for reliability with fixed size PV"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es+pv/step1")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results,
                                 self.validated_folder / "pro_formauc3_es+pv_step1.csv",
                                 PCT_ERROR_PROFORMA)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es+pv_step1.csv",
                             PCT_ERROR_SIZE)


class TestUsecase2EssPv4BtmUserConstraints:
    """ Part 2 of Usecase 2B - 1ESS, 1PV - fixed size, BTM with user constraint and PF
    reliabilty calculations"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es+pv/step2")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results,
                                 self.validated_folder / "pro_formauc3_es+pv_step2.csv",
                                 PCT_ERROR_PROFORMA, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es+pv_step2.csv",
                             PCT_ERROR_SIZE)


class TestUsecase2EssSizingPvIce4Reliability:
    """ Part 1 of Usecase 2C - BAT, PV (fixed size), ICE fixed sized - sized for reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV+DG_Step1.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es+pv+dg/step1")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results,
                                 self.validated_folder / "pro_formauc3_es+pv+dg_step1.csv",
                                 PCT_ERROR_PROFORMA)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es+pv+dg_step1.csv",
                             PCT_ERROR_SIZE)


class TestUsecase2EssPvIce4BtmUserConstraints:
    """ Part 2 of Usecase 2C - fixed size BAT + PV, DCM and retailTimeShift with User constraints
    and PF reliability calculations"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV+DG_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es+pv+dg/step2")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results,
                                 self.validated_folder / "pro_formauc3_es+pv+dg_step2.csv",
                                 PCT_ERROR_PROFORMA)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es+pv+dg_step2.csv",
                             PCT_ERROR_SIZE)


USECASE3PLANNED = Path("./Model_params/Usecase3/planned")


class TestUseCase3EssSizing4PlannedOutage:
    """ Part 1 of Usecase 3A - BAT sizing for a planned outage on one day"""

    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3PLANNED / \
                       "Model_Parameters_Template_Usecase3_Planned_ES.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/planned/es")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             PCT_ERROR_SIZE)


class TestUseCase3Ess4DaFrUserConstraintsPlannedOutage:
    """ Part 2 of Usecase 3A - Sized BAT, DA + FR + User constraints"""

    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3PLANNED / \
                       "Model_Parameters_Template_Usecase3_Planned_ES_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/planned/step2/es")

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3.csv",
                                 PCT_ERROR_PROFORMA, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             PCT_ERROR_SIZE)


class TestUseCase3EssSizingPv4PlannedOutage:
    """ Part 1 of Usecase 3A - BAT sizing for planned outage with fixed PV"""

    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3PLANNED / \
                       "Model_Parameters_Template_Usecase3_Planned_ES+PV.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/planned/es+pv")

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             PCT_ERROR_SIZE)


class TestUseCase3EssPv4DaFrUserConstraintsPlannedOutage:
    """ Part 2 of Usecase 3A - User constraints + FR + DA with fixed size PV and Battery"""

    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3PLANNED / \
                       "Model_Parameters_Template_Usecase3_Planned_ES+PV_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/planned/step2/es+pv")

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3.csv",
                                 PCT_ERROR_PROFORMA, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             PCT_ERROR_SIZE)


class TestUseCase3EssSizingPvIce4PlannedOutage:
    """ Part 1 of Usecase 3 Planned C - BAT sizing + fixed PV + fixed ICE  for reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3PLANNED / \
                       "Model_Parameters_Template_Usecase3_Planned_ES+PV+DG.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/planned/es+pv+dg")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             PCT_ERROR_SIZE)


class TestUseCase3EssPvIce4DaFrUserConstraintsPlannedOutage:
    """ Part 2 of Usecase 3 Planned C - fixed sized BAT, PV, ICE for FR + DA"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3PLANNED / \
                       "Model_Parameters_Template_Usecase3_Planned_ES+PV+DG_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/planned/step2/es+pv+dg")

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results,
                                 self.validated_folder / "pro_formauc3.csv",
                                 PCT_ERROR_PROFORMA, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             PCT_ERROR_SIZE)


USECASE3UNPLANNED = Path("./Model_params/Usecase3/unplanned")


class TestUseCase3EssSizing4Reliability:
    """ Part 1 of Usecase 3 Unplanned A - BAT sizing for reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3UNPLANNED / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/unplanned/es")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             PCT_ERROR_SIZE)


#FIXME
class xTestUseCase3EssSizing4DaFrUserConstraintsUnplannedOutage_slow:
    """ Part 2 of Usecase 3 Unplanned A - FR + DA + UserConstraints, BAT fixed size with PF
    reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3UNPLANNED / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/unplanned/step2_ws/es")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3_es_step2.csv",
                                 PCT_ERROR_PROFORMA, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es_step2.csv",
                             PCT_ERROR_SIZE)


class TestUseCase3EssPvSizing4Reliability:
    """ Part 1 of Usecase 3 Unplanned B - BAT + PV sizing for reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3UNPLANNED / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/unplanned/es+pv")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             PCT_ERROR_SIZE)


#FIXME
class xTestUseCase3EssPv4DaFrUserConstraintsUnplannedOutage_slow:
    """ Part 2 of Usecase 3 Unplanned B - FR + DA + UserConstraints, BAT + PV fixed size with PF
    reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3UNPLANNED / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/unplanned/step2_ws/es+pv1")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results, self.validated_folder /
                                 "pro_formauc3_es+pv_step2.csv",
                                 PCT_ERROR_PROFORMA, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es+pv_step2.csv",
                             PCT_ERROR_SIZE)


class TestUseCase3EssPvIceSizing4Reliability:
    """ Part 1 of Usecase 3 Unplanned B - BAT + PV + ICE sizing for reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3UNPLANNED / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV+DG.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/unplanned/es+pv+dg")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             PCT_ERROR_SIZE)


class xTestUseCase3EssPvIce4DaFrUserConstraintsUnplannedOutage_slow:
    """ Part 2 of Usecase 3 Unplanned C - FR + DA + UserConstraints, BAT + PV fixed size with PF
    reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3UNPLANNED / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV+DG_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/unplanned/step2_ws/es+pv+dg")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results, self.validated_folder /
                                 "pro_formauc3_es+pv_step2.csv",
                                 PCT_ERROR_PROFORMA, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es+pv_step2.csv",
                             PCT_ERROR_SIZE)
