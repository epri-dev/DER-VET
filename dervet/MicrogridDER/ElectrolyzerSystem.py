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
ElectrolyzerSystem

This Python class contains methods and attributes specific for an ElectrolyzerSystem Technology
"""
import logging
import cvxpy as cvx
import numpy as np
import pandas as pd
import rainflow
import storagevet.Library as Lib
from storagevet.Technology.DistributedEnergyResource import DER
from dervet.MicrogridDER.DERExtension import DERExtension
from dervet.MicrogridDER.ContinuousSizing import ContinuousSizing
from storagevet.ErrorHandling import *
from storagevet.Library import truncate_float, is_leap_yr


class ElectrolyzerSystem(DER, ContinuousSizing, DERExtension):
    """
    """

    def __init__(self, params):
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters
        """
        TellUser.debug(f"Initializing ElectrolyzerSystem")
        self.tag = 'ElectrolyzerSystem'
        # create generic technology object
        DER.__init__(self, params)
        ContinuousSizing.__init__(self, params)
        DERExtension.__init__(self, params)

        self.technology_type = 'Load'
        #self.technology_type = 'ElectrolyzerSystem'
        self.tag = 'ElectrolyzerSystem'

        self.efficiency = params['efficiency'] # kg/kWh
        self.variable_efficiency_data = params.get('variable_efficiency_data', None)

        self.h2_unit_value = params['hydrogen_unit_value']  # $/kg
        self.ts_hydrogen_schedule = params.get('ts_hydrogen_schedule', False)

        self.hydrogen_production_quota = params['hydrogen_production_quota']

        self.annual_hydrogen_production_kg = pd.DataFrame()

        self.capital_cost_function = [params['ccost'],
                                      params['ccost_kW']]

        self.fixed_om = params['fixed_om_cost']  # $/kW-yr
        self.variable_om = params['variable_om_cost'] # $/kg
        self.hydrogen_schedule_mode = params['hydrogen_schedule_mode']

        #self.variable_names = {'electrolyzer_power', 'electrolyzer_schedule'}

        self.incl_binary = params['binary'] # not used

        # initialize degradation attributes
        self.cycle_life = params.get('cycle_life_data', None)
        self.degrade_perc = 0.0
        self.soh_initial = 1 #Initial SOC at the start of the project
        self.soh=1 #Initial SOC at the start of the project
        self.yearly_degrade = params['yearly_degrade'] / 100
        self.incl_degradation = bool(params['incl_degradation'])
        self.degrade_data = None
        self.counted_cycles = []
        self.eol_condition = 0  # this is not a collected input
        self.state_of_health = 0  # this is not a collected input
        self.years_system_degraded = set()
        self.yearly_degradation_report = pd.DataFrame()
        self.actual_time_to_replacement = None  # set by degredation module

        self.rated_power = params['rated_power']
        self.p_min = params['min_power']
        self.max_rated_power = params['max_rated_power']
        self.min_rated_power = params['min_rated_power']
        if not self.rated_power:
            self.rated_power = cvx.Variable(integer=True, name=f'{self.name} rating')
            self.size_constraints += [cvx.NonPos(-self.rated_power)]
            if self.min_rated_power:
                self.size_constraints += [cvx.NonPos(self.min_rated_power - self.rated_power)]
            if self.max_rated_power:
                self.size_constraints += [cvx.NonPos(self.rated_power - self.max_rated_power)]

        self.unset_rated_power = None
        self.unset_size_constraints = None
        self.was_sized = False

        # do not allow the ElectrolyzerSystem to participate in market services
        # it has no ability to do so
        self.can_participate_in_market_services = False

        # checks when not sizing
        if not self.being_sized():
            # ensure that scheduled power values do not exceed rated_power
            if self.hydrogen_schedule_mode == 1:
                if self.ts_hydrogen_schedule.max() > self.rated_power:
                    TellUser.error(f'{self.tag} {self.name}: There is at least one value in the input ElectrolyzerSystem Schedule (kW) column of the input time series that exceeds the rated_power value ({self.rated_power} kW)')
                    TellUser.close_log()
                    raise InputDataError()

    def initialize_degradation_module(self, opt_agg):
        """
        Notes: Should be called once, after optimization levels are assigned, but before
        optimization loop gets called

        Args:
            opt_agg (DataFrame):

        Returns: None
        """
        if self.incl_degradation:
            # initialize degradation dataframe
            self.degrade_data = pd.DataFrame(index=['Optimization Start']+list(opt_agg.control.unique()))
            self.degrade_data['degradation progress %'] = self.degrade_perc
            self.degrade_data['state of health %'] = self.soh * 1e2
            # NOTE: we do not model performance degradation here; self.degraded_energy_capacity() is constant
            #self.degrade_data['effective energy capacity (kWh)'] = self.degraded_energy_capacity()
            self.calc_degradation('Optimization Start', None, None)
            # calculate current degrade_perc since installation
            step_before_optimziation_problems = opt_agg.sort_index().index[0] - pd.Timedelta(self.dt, unit='h')
            self.calc_degradation('Optimization Start', self.operation_year.to_timestamp(), step_before_optimziation_problems)

    def degraded_energy_capacity(self):
        """ Updates rated_power and control constraints based on degradation percent
        Applies degrade percent to rated power capacity

        Returns:
            Degraded power capacity
        """

        ## reduce the performance based on the SOH
        #soh_change = self.degrade_perc
        #new_ene_max = max(self.rated_power * (1 - soh_change), 0)
        #return new_ene_max

        # do not reduce the performance based on the SOH
        return self.rated_power

    def calc_degradation(self, opt_period, start_dttm, last_dttm):
        """ calculate degradation percent based on yearly degradation and cycle degradation

        Args:
            opt_period: the index of the optimization that occurred before calling this function, None if
                no optimization problem has been solved yet
            start_dttm (DateTime): Start timestamp to calculate degradation. ie. the first datetime in the optimization
                problem
            last_dttm (DateTime): End timestamp to calculate degradation. ie. the last datetime in the optimization
                problem

        A percent that represented the power degradation
        """
        # time difference between time stamps converted into years multiplied by yearly degrate rate
        if self.incl_degradation:
            cycle_degrade = 0
            yearly_degradation = 0

            if not isinstance(opt_period, str):
                # calculate degradation due to cycling iff energy values are given
                energy_series = self.variables_df.loc[start_dttm:last_dttm, 'electrolyzer_power']
                # Find the effective energy capacity
                eff_e_cap = self.degraded_energy_capacity()

                #If using rainflow counting package uncomment following few lines
                # use rainflow counting algorithm to get cycle counts
                # cycle_counts = rainflow.count_cycles(energy_series, ndigits=4)
                #
                # aux_df = pd.DataFrame(cycle_counts, columns=['Depth', 'N_cycles'])
                # aux_df['Opt window'] = opt_period
                #
                # # sort cycle counts into user inputed cycle life bins
                # digitized_cycles = np.searchsorted(self.cycle_life['Power Fluctuation (%)'],[min(i[0]/eff_e_cap, 1) for i in cycle_counts], side='left')

                # use rainflow extract function to get information on each cycle
                cycle_extract=list(rainflow.extract_cycles(energy_series))
                aux_df = pd.DataFrame(cycle_extract, columns=['rng', 'mean','count','i_start','i_end'])
                aux_df['Opt window'] = opt_period

                # sort cycle counts into user inputed cycle life bins
                # NOTE: digitized_cycles can be out of range here if cycle depth > 1
                digitized_cycles = np.searchsorted(self.cycle_life['Power Fluctuation (%)'],[min(i[0] / eff_e_cap, 1) for i in cycle_extract], side='left')
                aux_df['Input_cycle_Depth_mapping'] = np.array(self.cycle_life['Power Fluctuation (%)'][digitized_cycles]*eff_e_cap)
                aux_df['Cycle Life Value (cycles)'] = np.array(self.cycle_life['Cycle Life Value (cycles)'][digitized_cycles] )

                self.counted_cycles.append(aux_df.copy())
                # sum up number of cycles for all cycle counts in each bin
                cycle_sum = self.cycle_life.loc[:, :]
                cycle_sum.loc[:, 'cycles'] = 0.0
                for i in range(len(cycle_extract)):
                    cycle_sum.loc[digitized_cycles[i], 'cycles'] += cycle_extract[i][2]

                # sum across bins to get total degrade percent
                # 1/cycle life value is degrade percent for each cycle
                cycle_degrade = np.dot(1/cycle_sum['Cycle Life Value (cycles)'], cycle_sum.cycles)* (1 - self.eol_condition)

            if start_dttm is not None and last_dttm is not None:
                # add the yearly degradation linearly to the # of years from START_DTTM to (END_DTTM + dt)
                days_in_year = 366 if is_leap_yr(start_dttm.year) else 365
                portion_of_year = (last_dttm + pd.Timedelta(self.dt, unit='h') - start_dttm) / pd.Timedelta(days_in_year, unit='d')
                yearly_degradation = self.yearly_degrade * portion_of_year

            # add the degradation due to time passing and cycling for total degradation
            degrade_percent = cycle_degrade + yearly_degradation

            # record the degradation
            # the total degradation after optimization OPT_PERIOD must also take into account the
            # degradation that occurred before the ElectrolyzerSystem was in operation (which we saved as SELF.DEGRADE_PERC)
            self.degrade_data.loc[opt_period, 'degradation progress %'] = degrade_percent + self.degrade_perc
            self.degrade_perc += degrade_percent

            soh_new = self.soh_initial - self.degrade_perc
            self.soh = self.degrade_data.loc[opt_period, 'state of health %'] = soh_new

            # apply degradation to technology (affects physical_constraints['electrolyzer_power'] and control constraints)
            # NOTE: we do not model performance degradation here; self.degraded_energy_capacity() is constant
            eff_e_cap = self.degraded_energy_capacity()
            #TellUser.info(f"ELETROLYZERSYSTEM - {self.name}: effective energy capacity is now {truncate_float(eff_e_cap)} kWh " +
            #              f"({truncate_float(100*(1 - (self.rated_power-eff_e_cap)/self.rated_power), 7)}% of original)")
            #self.degrade_data.loc[opt_period, 'effective energy capacity (kWh)'] = eff_e_cap
            #self.effective_soe_max = eff_e_cap * 1  #self.ulsoc
            #self.effective_soe_min = eff_e_cap * 0  #self.llsoc

            if self.degraded_energy_capacity() <= self.rated_power * self.state_of_health:
                # record the year that the energy capacity reaches the point of replacement
                self.years_system_degraded.add(start_dttm.year)

                # reset the energy capacity to its original nameplate if replaceable
                if self.replaceable:
                    self.degrade_perc = 0
                    #self.effective_soe_max = 1 * self.rated_power  #self.ulsoc * self.ene_max_rated
                    #self.effective_soe_min = 0 * self.rated_power  #self.llsoc * self.ene_max_rated

    def set_end_of_life_based_on_degradation_cycle(self, analysis_years, start_year, end_year, is_ecc):
        """If degradation occurred AND it it the end of the optimization loop call this method -->
                if state of health reaches 0 during optimization loop --> calculate expected lifetime
                ELSE
                calculate the average yearly degradation, then estimate the expected lifetime is 1 / yearly degradation
            Reports to user if the expected lifetime is not the same as the user inputted expected lifetime

        Args:
            analysis_years (list):
            start_year (pd.Period):
            end_year (pd.Period):
            is_ecc (bool): if CBA calculations will show the ECC (if the booklife changes, then we need to
                warn the user...see message below)

        Returns: the new last year of operation for the BAT after 1 lifetime

        """
        if self.incl_degradation:
            # ESTIMATE EXPECTED LIFETIME
            num_full_lifetimes = len(self.years_system_degraded)
            if num_full_lifetimes:
                # get number of years it took to be replaced (get average if replaced more than once)
                foo = max(self.years_system_degraded) + 1 - self.operation_year.year
                avg_lifetime = foo / num_full_lifetimes
                # set FAILURE_YEARS to be the years that the system degraded
                self.failure_preparation_years = list(self.years_system_degraded)
            else:
                # create a data frame with a row for every year in the project lifetime
                yr_index = pd.period_range(start=start_year, end=end_year, freq='Y')
                self.yearly_degradation_report = pd.Series(index=pd.Index(yr_index))

                # determine yearly degradation for the years that we counted cycles for
                no_years_solved = len(analysis_years)
                analysis_years = np.sort(analysis_years)  # sort the list of years, smallest to largest
                no_optimizations_per_year = len(self.degrade_data.iloc[1:].index) / no_years_solved
                for indx, year in enumerate(analysis_years):
                    first_degrad_inx = indx * no_optimizations_per_year
                    initial_degradation = self.degrade_data.iloc[int(first_degrad_inx)]['degradation progress %']
                    last_degrad_idx = first_degrad_inx + no_optimizations_per_year
                    final_degradation = self.degrade_data.iloc[int(last_degrad_idx)]['degradation progress %']
                    tot_yr_degradation = final_degradation - initial_degradation
                    self.yearly_degradation_report[pd.Period(year, freq='Y')] = tot_yr_degradation
                # fill in the remaining years (assume constant degradation)
                self.yearly_degradation_report.ffill(inplace=True)
                # estimate lifetime with average yearly degradation
                # NOTE: 1 minus state_of_health is the same as degradation progress %
                avg_lifetime = (1-self.state_of_health)/self.yearly_degradation_report.mean()

            # reset failure years
                self.failure_preparation_years = []

            # set EXPECTED_LIFETIME to be the actual EOL -- it should never be 0 years
            # NOTE: avg_lifetime gets rounded here (downward with int())
            self.actual_time_to_replacement = max(int(avg_lifetime), 1)
            # report actual EOL to user
            TellUser.warning(f"{self.unique_tech_id()} degradation is ON, and so we have estimated the EXPECTED_LIFETIME" +
                             f" to be {self.actual_time_to_replacement} years (inputted value: {self.expected_lifetime})")
            if is_ecc and (self.actual_time_to_replacement != self.expected_lifetime):
                general_msg = "CBA ECC: The USER-GIVEN expected lifetime is not the ACTUAL lifetime of the Electrolyzer System.\nThe ECC calculation costs " \
                              "will still be annualized for the USER-GIVEN lifetime, but replacements (if any) will incur with the smallest\n" \
                              "lifetime. Please update your expected lifetime and ecc% to match the ACTUAL lifetime of this DER and rerun for a " \
                              "more accurate Economic Carrying Cost analysis.\n" \
                              f"-- ElectrolyzerSystem name: {self.name} -- USER-GIVEN (expected lifetime: {self.expected_lifetime}, ecc%: " \
                              f"{self.ecc_perc * 100}) -- ACTUAL (expected lifetime: {self.actual_time_to_replacement}, ecc%: ?) --"
                TellUser.error(general_msg)

            # set FAILURE_YEARS to be the years that the system degraded to SOH=0
            failed_on = max(self.years_system_degraded) if num_full_lifetimes else None
            time_btw_replacement = min(self.actual_time_to_replacement, self.expected_lifetime)
            self.set_failure_years(end_year, equipment_last_year_operation=failed_on, time_btw_replacement=time_btw_replacement)

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        """
        if self.hydrogen_schedule_mode == 1:
            self.ts_hydrogen_schedule = Lib.fill_extra_data(self.ts_hydrogen_schedule, years, load_growth, frequency)
            self.ts_hydrogen_schedule = Lib.drop_extra_data(self.ts_hydrogen_schedule, years)

    def initialize_variables(self, size):
        """ Adds optimization variables to dictionary

        Variables added: (with self.unique_ess_id as a prefix to these)
            electrolyzer_power (Variable): A cvxpy variable for ElectrolyzerSystem Power at the previous time step (kW)

        Notes:
            CVX Parameters turn into Variable when the condition to include them is active

        Args:
            size (Int): Length of optimization variables to create

        """
        self.variables_dict = {
            'electrolyzer_power': cvx.Variable(shape=size, name=f'{self.name}-power', nonneg=True),
            'electrolyzer_on': cvx.Variable(shape=size, boolean=True, name=f'{self.name}-on'),
        }

    def get_fixed_om(self):
        """ Returns the fixed om of a given technology
        """
        return self.fixed_om * self.rated_power

    def get_charge(self, mask):
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the charge as a function of time for the

        """

        return self.variables_dict['electrolyzer_power']

    def charge_capacity(self, solution=False):
        """

        Returns: the maximum charge that can be attained

        """
        if not solution:
            return self.rated_power
        else:
            try:
                rated_power = int(self.rated_power.value)
            except (AttributeError, TypeError):
                rated_power = self.rated_power
            return rated_power

    def name_plate_capacity(self):
        """ Returns the value of 1 electrolyzer in a set of electrolyzers

        Returns:

        """
        try:
            rated_power = self.rated_power.value
        except AttributeError:
            rated_power = self.rated_power
        return rated_power

    def set_size(self):
        """ Save value of size variables of DERs

        """
        # set these unset_ values to fall back on (if unset_size() is called)
        self.unset_rated_power = self.rated_power
        self.unset_size_constraints = self.size_constraints
        self.rated_power = self.name_plate_capacity()
        self.size_constraints = []
        self.was_sized = True

    def unset_size(self):
        """ Return size variables back to what they were before sizing
            Can only be used after set_size() is called
        """
        if self.was_sized:
            self.rated_power = self.unset_rated_power
            self.size_constraints = self.unset_size_constraints
            self.was_sized = False

    def get_capex(self, **kwargs):
        """ Returns the capex of a given technology
            NOTE: assumes a single unit
        """
        return np.dot(self.capital_cost_function, [1, self.rated_power])

    def objective_function(self, mask, annuity_scalar=1):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            self.costs (Dict): Dict of objective costs
        """
        costs = super().objective_function(mask, annuity_scalar)
        costs.update(self.sizing_objective())

        # NOTE: we use a constant efficiency here to avoid a non-linearity
        # add hydrogen value expression in $
        #   this is a negative sum here because it is not a cost (it's the opposite)
        #   (The optimization problem is a minimization problem, so negative values here are good)
        #   units: $/kg * kW * kg/kWh = $
        hydrogen_exp = cvx.sum(-1 * self.h2_unit_value * self.variables_dict['electrolyzer_power'] * self.efficiency * self.dt * annuity_scalar)
        # add variable o&m expression in $
        #   units: $/kg * kW * kg/kWh = $
        variable_om_exp = cvx.sum(self.variable_om * self.variables_dict['electrolyzer_power'] * self.efficiency * self.dt * annuity_scalar)
        costs.update({
            self.name + ' hydrogen_value': hydrogen_exp,
            self.name + ' fixed_om': self.get_fixed_om() * annuity_scalar,
            self.name + ' variable_om': variable_om_exp
        })

        return costs

    def constraints(self, mask):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns:
            A list of constraints that corresponds the EV requirement to collect the required energy to operate. It also allows
            flexibility to provide other grid services
        """

        constraint_list = []

        # optimization variables
        electrolyzer_power = self.variables_dict['electrolyzer_power']
        electrolyzer_on = self.variables_dict['electrolyzer_on']

        # constraints on hydrogen production
        if self.hydrogen_schedule_mode != 1:
            constraint_list += [cvx.NonPos((electrolyzer_on * self.p_min) - electrolyzer_power)]
            if not self.being_sized():
                # without a load timeseries input, DERVET controls the operation
                # limit the power of the ElectrolyzerSystem to at most its rated power, when on
                #   while considering the min_power, when on
                constraint_list += [cvx.NonPos(electrolyzer_power - (electrolyzer_on * self.rated_power))]
            else:
                constraint_list += [cvx.NonPos(electrolyzer_power - self.charge_capacity())]
                if self.max_rated_power:
                    constraint_list += [cvx.NonPos(electrolyzer_power - (electrolyzer_on * self.max_rated_power))]
                # add size constraints
                constraint_list += self.size_constraints

        else:
            # with a load timeseries input, the operation/power is fixed according to that timeseries
            # min_power is ignored
            TellUser.warning(f'For {self.unique_tech_id()}, since we are using a fixed schedule, we ignore the min_power parameter.')
            electrolyzer_schedule = cvx.Parameter(shape=sum(mask), name=self.name + '-schedule', value=self.ts_hydrogen_schedule[mask].values)
            constraint_list += [cvx.Zero(electrolyzer_power - electrolyzer_schedule)]
        if self.hydrogen_schedule_mode == 2:
            # ensure hydrogen production (power * efficiency) meets or exceeds the quota
            constraint_list += [cvx.NonPos(self.hydrogen_production_quota - sum(electrolyzer_power * self.efficiency))]
            # NOTE: if efficiency were no longer a constant value, this would become
            #   a nonlinear constraint and would break our optimization approach,
            #   so some handling is required when the incl_variable_efficiency parameter is set to 1
            #   In these case, we use the variable efficiency data in post-optimization
            if self.variable_efficiency_data is not None:
                TellUser.warning(f'For {self.unique_tech_id()}, since we are using a variable efficiency post-optimization, we cannot ensure that the hydrogen production quota will be met. The optimization does not know that it may be running the ElectrolyzerSystem inefficiently.')

        return constraint_list

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that
            summarize the results pertaining to this instance

        """
        results = pd.DataFrame(index=self.variables_df.index)
        # from Load Technology
        results[f"{self.unique_tech_id()} Original Load (kW)"] = \
            self.variables_df['electrolyzer_power']
        tech_id = self.unique_tech_id()
        solve_dispatch_opt = self.variables_df.get('electrolyzer_power')
        if solve_dispatch_opt is not None:
            results[tech_id + ' Power (kW)'] = self.variables_df['electrolyzer_power']
            results[tech_id + ' Hydrogen Production (kg)'] = self.variables_df['electrolyzer_power'] * self.efficiency
            results[tech_id + ' Efficiency (kg/kWh)'] = self.efficiency
            if self.hydrogen_schedule_mode == 1:
                # with a fixed schedule, electrolyzer_on is not in the optimization, so we create it here
                # set it to 0 when the electrolyzer_power is zero; otherwise set it to 1
                results[tech_id + ' On (y/n)'] = 1
                results[tech_id + ' On (y/n)'] = results[tech_id + ' On (y/n)'].mask(self.variables_df['electrolyzer_power'] == 0, 0)
            else:
                results[tech_id + ' On (y/n)'] = self.variables_df['electrolyzer_on']
            # handle variable efficiency (post-optimization step)
            if self.variable_efficiency_data is not None:
                results = self.use_variable_efficiency(results, tech_id)

        return results

    def use_variable_efficiency(self, results, tech_id):
        """ Adds 2 new columns to the time series results, using a variable efficiency input curve.
            This is a post-optimization step.

        Args:
            results: the time series results df for this technology instance
            tech_id: the name of this technology instance

        Returns: The Electrolyzer dataframe has the following 2 columns added:
            Hydrogen Production Actual (kg)
            Efficiency Actual (kg/kWh)

        """
        TellUser.info(f'{tech_id} has variable efficiency, so we calculate the actual hydrogen production now using uploaded variable efficiency curve data.')
        # calculate the % of full power the electrolyzer is operating in at each time step
        fractional_power = results[tech_id + ' Power (kW)'] / self.name_plate_capacity()
        # ensure that fractional_power never exceeds a value of 1
        if fractional_power.max() > 1:
            TellUser.error(f'{tech_id} fractional power is greater than 1 at times. This should not happen. Something went wrong when calculating Efficiency Actual.')

        # using the variable efficiency curve, determine the actual efficiency at each time step
        self.variable_efficiency = self.calc_variable_efficiency(fractional_power)
        results[tech_id + ' Efficiency Actual (kg/kWh)'] = self.variable_efficiency
        # calculate hydrogen production actual
        results[tech_id + ' Hydrogen Production Actual (kg)'] = \
            self.variables_df['electrolyzer_power'] * self.variable_efficiency

        return results

    def calc_variable_efficiency(self, fractional_power):
        """ Efficiency is determined from the input variable efficiency curve
            Given a time series of fractional_power, we use the efficiency curve
            to look up the index where that value occurs, and lookup the Efficiency value.
            NOTE: values are not interpolated, but they could be using idx_hi

        Args:
            fractional_power: a series of values between 0 and 1

        Returns:
            a series of Efficiency Actual (kg/kWh)
        """
        # first make sure that variable_efficiency_data is ordered by Fractional Power (low to hi)
        self.variable_efficiency_data.sort_values('Fractional Power (%)', inplace=True)
        self.variable_efficiency_data.reset_index(drop=True, inplace=True)
        # create the output series
        efficiency_actual = fractional_power.copy()
        efficiency_actual.name = 'Efficiency Actual'
        # FIXME: this is a slow way to do things
        for j, i in enumerate(fractional_power):
            idx = None
            if self.variable_efficiency_data['Fractional Power (%)'].eq(i).sum() == 1:
                idx = self.variable_efficiency_data['Fractional Power (%)'].eq(i).idxmax()
            if idx is None:
                idx_hi = self.variable_efficiency_data['Fractional Power (%)'].ge(i).idxmax()
                idx_lo = max(idx_hi, 1) - 1
                idx = idx_lo
            if idx is None:
                TellUser.error(f'Efficiency cannot be deternined from a fractional power of {i}')
            efficiency_actual.iloc[j] = self.variable_efficiency_data.iloc[idx]['Efficiency (kg/kWh)']
            #print(f'{i}, {idx}, {efficiency_actual}')

        return efficiency_actual

    def proforma_report(self, apply_inflation_rate_func, fill_forward_func, results):
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            apply_inflation_rate_func:
            fill_forward_func:
            results (pd.DataFrame):

        Returns: A DateFrame of with each year in opt_year as the index and
            the corresponding value this stream provided.

        """
        pro_forma = super().proforma_report(apply_inflation_rate_func, fill_forward_func, results)
        if self.variables_df.index.empty:
            return pro_forma
        optimization_years = self.variables_df.index.year.unique()
        tech_id = self.unique_tech_id()
        om_costs = pd.DataFrame()
        h2_value = pd.DataFrame()
        cumulative_energy_dispatch_kw = pd.DataFrame()
        power = self.variables_df['electrolyzer_power']
        power_column_name = tech_id + ' Cumulative Energy Dispatch (kW)'
        variable_column_name = tech_id + ' Variable O&M Costs'
        h2_value_column_name = tech_id + ' Hydrogen Value'
        h2_production_column_name = tech_id + ' Hydrogen Production (kg)'
        #h2_production_actual_column_name = tech_id + ' Hydrogen Production Actual (kg)'
        for year in optimization_years:
            index_yr = pd.Period(year=year, freq='Y')
            # add fixed o&m costs
            om_costs.loc[index_yr, self.fixed_column_name()] = -self.fixed_om
            # add variable_om costs
            om_costs.loc[index_yr, variable_column_name] = -self.variable_om
            power_sub = power.loc[power.index.year == year]
            if self.variable_efficiency_data is not None:
                # add variable power and variable efficiency value
                #   and then sum that to get cumulative energy dispatch
                efficiency_sub = self.variable_efficiency.loc[self.variable_efficiency.index.year == year]
                cumulative_energy_dispatch_kw.loc[index_yr, power_column_name] = np.sum(power_sub * efficiency_sub)
            else:
                # sum variable power value to get cumulative energy dispatch,
                #   and multiply by constant efficiency
                cumulative_energy_dispatch_kw.loc[index_yr, power_column_name] = np.sum(power_sub) * self.efficiency
            # hydrogen value = power (variable) * efficiency * unit_value
            h2_value.loc[index_yr, h2_value_column_name] = self.h2_unit_value
            # hydrogen production = power (variable) * efficiency
            self.annual_hydrogen_production_kg.loc[index_yr, h2_production_column_name] = 1.0

        # fill forward (escalate rates)
        om_costs = fill_forward_func(om_costs, None, is_om_cost=True)
        h2_value = fill_forward_func(h2_value, None)  # FIXME: is the h2 value subject to inflation?
        # NOTE: we use a 0 inflation rate here, because this is not a dollar value
        self.annual_hydrogen_production_kg = fill_forward_func(self.annual_hydrogen_production_kg, 0)

        # interpolate cumulative energy dispatch between analysis years
        #   be careful to not include years labeled as Strings (CAPEX)
        years_list = list(filter(lambda x: not(type(x) is str), om_costs.index))
        analysis_start_year = min(years_list).year
        analysis_end_year = max(years_list).year
        cumulative_energy_dispatch_kw = self.interpolate_energy_dispatch(
            cumulative_energy_dispatch_kw, analysis_start_year, analysis_end_year, None)
        # calculate om costs in dollars, as rate * power
        # fixed om
        om_costs.loc[:, self.fixed_column_name()] = om_costs.loc[:, self.fixed_column_name()] * self.rated_power
        # variable_om
        om_costs.loc[:, variable_column_name] = om_costs.loc[:, variable_column_name] * self.dt * cumulative_energy_dispatch_kw.loc[:, power_column_name]
        # calculate hydrogen production value in dollars (variable)
        #   hydrogen value = power (variable) * efficiency * unit_value
        #   NOTE: efficiency (which can be variable) is baked into cumulative_energy_dispatch
        h2_value.loc[:, h2_value_column_name] = h2_value.loc[:, h2_value_column_name] * self.dt * cumulative_energy_dispatch_kw.loc[:, power_column_name]
        # append with super class's proforma
        pro_forma = pd.concat([pro_forma, om_costs, h2_value], axis=1)

        # calc & save the annual hydrogen production in kg, for LCOH calc in CBA
        #   hydrogen production = power (variable) * efficiency
        #   NOTE: efficiency (which can be variable) is baked into cumulative_energy_dispatch
        self.annual_hydrogen_production_kg.loc[:, h2_production_column_name] = self.annual_hydrogen_production_kg.loc[:, h2_production_column_name] * self.dt * cumulative_energy_dispatch_kw.loc[:, power_column_name]
        self.annual_hydrogen_production_kg = self.annual_hydrogen_production_kg[h2_production_column_name]

        return pro_forma

    def sizing_summary(self):
        """ Creates the template for sizing df that each DER must fill to report their size.

        Returns: A dictionary describe this DER's size and captial costs.

        """
        sizing_results = {
            'DER': self.name,
            'Power (kW)': self.name_plate_capacity(),
            'Capital Cost ($)': self.capital_cost_function[0],
            'Capital Cost ($/kW)': self.capital_cost_function[1],
        }

        return sizing_results

    def update_for_evaluation(self, input_dict):
        """ Updates price related attributes with those specified in the input_dictionary

        Args:
            input_dict: hold input data, keys are the same as when initialized

        """
        super().update_for_evaluation(input_dict)

        variable_cost = input_dict.get('variable_om_cost')
        if variable_cost is not None:
            self.variable_om = variable_cost

        fixed_om_cost = input_dict.get('fixed_om_cost')
        if fixed_om_cost is not None:
            self.fixed_om = fixed_om_cost

    def sizing_error(self):
        """

        Returns: True if there is an input error

        """
        if self.min_rated_power < self.p_min:
            TellUser.error(f'{self.unique_tech_id()} would not be able to have a rating less than {self.p_min}. ' +
                           f'Please update min_rated_power to reflect this constraint.')
            return True
        return False

    def capacity_factor(self):
        """
        Creates the template for reporting the capacity factor of each DER

        The capacity factor (unitless) is calculated using the time series results from DER-VET
        CF = sum(the DER's power) / ( the number of timesteps * the DER's rated power)

        More details:
        The net capacity factor is the unitless ratio of actual electrical energy output over
        a given period of time to the theoretical maximum electrical energy output over that
        period. The theoretical maximum energy output of a given installation is defined as
        that due to its continuous operation at full nameplate capacity over the relevant period

        Returns: a dictionary with the name, technology_type (inherited) and the capacity factor
        """
        capacity_factor_dict = super().capacity_factor()

        if 'electrolyzer_power' in self.variables_df.columns:
            elec = self.variables_df['electrolyzer_power']
            # NOTE: this occurs post-opimization, so that if being_sized() is true,
            #       rated_power is already set to a numerical value
            if self.rated_power == 0:
                # non-operation
                capacity_factor_dict['Capacity Factor'] = 0
            else:
                capacity_factor_dict['Capacity Factor'] = sum(elec) / (elec.size * self.rated_power)

        return capacity_factor_dict

    def drill_down_reports(self, **kwargs):
        """Calculates any service related dataframe that is reported to the user.

        Args:
            monthly_data:
            time_series_data:
            technology_summary:
            sizing_df:

        Returns: dictionary of DataFrames of any reports that are value stream specific
            keys are the file name that the df will be saved with
        """
        dct = super().drill_down_reports(**kwargs)

        if self.incl_degradation:
            dct[f"{self.name.replace(' ', '_')}_degradation_data"] = self.degrade_data
            total_counted_cycles = pd.concat(self.counted_cycles)
            dct[f"{self.name.replace(' ', '_')}_cycle_counting"] = total_counted_cycles

        # FIXME: should this be in the incl_degradation loop?
        self.yearly_degradation_report.name = "Yearly Degradation"
        dct[f"{self.name.replace(' ', '_')}_yearly_degradation"] = self.yearly_degradation_report

        return dct
