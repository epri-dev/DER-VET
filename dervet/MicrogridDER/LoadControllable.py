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
Load

This Python class contains methods and attributes specific for technology analysis within DERVET.
"""
import cvxpy as cvx
import numpy as np
from storagevet.Technology.Load import Load
from dervet.MicrogridDER.ContinuousSizing import ContinuousSizing
from dervet.MicrogridDER.DERExtension import DERExtension
from storagevet.ErrorHandling import *


class ControllableLoad(Load, DERExtension, ContinuousSizing):
    """ An Load object. this object does not size.

    """

    def __init__(self, params):
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters for initialization
        """
        TellUser.debug(f"Initializing {__name__}")
        # create generic technology object
        Load.__init__(self, params)
        DERExtension.__init__(self, params)
        ContinuousSizing.__init__(self, params)

        # input params  UNITS ARE COMMENTED TO THE RIGHT
        self.rated_power = params['power_rating']  # kW
        self.duration = params['duration']  # hour
        self.variables_dict = {}
        if self.duration:  # if DURATION is not 0
            self.tag = 'ControllableLoad'
            self.variable_names = {'power', 'ene_load', 'uene', 'udis', 'uch'}

    def discharge_capacity(self):
        """

        Returns: the maximum discharge that can be attained

        """
        return self.rated_power

    def operational_max_energy(self):
        """

        Returns: the maximum energy that should stored in this DER based on user inputs

        """
        return self.rated_power * self.duration

    def qualifying_capacity(self, event_length):
        """ Describes how much power the DER can discharge to qualify for RA or DR. Used to determine
        the system's qualifying commitment.

        Args:
            event_length (int): the length of the RA or DR event, this is the
                total hours that a DER is expected to discharge for

        Returns: int/float

        """
        # this is not compatible with sizing
        if self.being_sized():
           error_msg = f'Since {self.tag}: {self.name} is being sized by DER-VET, the qualifying_capacity (for RA or DR) is unknown before optimization. Therefore, we cannot continue. Turn off sizing for this Technology, or disable RA and DR.'
           TellUser.error(error_msg)
           TellUser.close_log()
           raise ModelParameterError(error_msg)
        return min(self.discharge_capacity(), self.operational_max_energy()/event_length)

    def initialize_variables(self, size):
        """ Adds optimization variables to dictionary

        Variables added:
            power (Variable): A cvxpy variable equivalent to dis and ch in batteries/CAES

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """
        # self.variables_dict = {}
        if self.duration:
            self.variables_dict = {'power': cvx.Variable(shape=size, name='power'),  # p_t = charge_t - discharge_t
                                   'upower': cvx.Variable(shape=size, name='upower'),
                                   'ene_load': cvx.Variable(shape=size, name='ene', nonneg=True),
                                   'uene': cvx.Variable(shape=size, name=self.name + '-uene', nonneg=True),
                                   'udis': cvx.Variable(shape=size, name=self.name + '-udis'),
                                   'uch': cvx.Variable(shape=size, name=self.name + '-uch'),
                                   }
        return self.variables_dict

    def get_charge(self, mask):
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the charge as a function of time for the

        """
        # load + (charge - discharge)
        effective_charge = super().get_charge(mask)
        if self.duration:
            effective_charge += self.variables_dict['power']
        return effective_charge

    def get_charge_up_schedule(self, mask):
        """ the amount of charging power in the up direction (supplying power up into the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        if self.duration:
            return np.repeat(self.rated_power, sum(mask)) + self.variables_dict['power']
        else:
            return super().get_charge_up_schedule(mask)

    def get_charge_down_schedule(self, mask):
        """ the amount of charging power in the up direction (pulling power down from the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        if self.duration:
            return np.repeat(self.rated_power, sum(mask)) - self.variables_dict['power']
        else:
            return super().get_charge_up_schedule(mask)

    def get_delta_uenegy(self, mask):
        """ the amount of energy, from the current SOE level the DER's state of energy changes
        from subtimestep energy shifting

        Returns: the energy throughput in kWh for this technology

        """
        if self.duration:
            return self.variables_dict['uene']
        else:
            return super().get_delta_uenegy(mask)

    def get_uenergy_increase(self, mask):
        """ the amount of energy in a timestep that is provided to the distribution grid

        Returns: the energy throughput in kWh for this technology

        """
        if self.duration:
            return self.variables_dict['uch'] * self.dt
        else:
            return super().get_uenergy_increase(mask)

    def get_uenergy_decrease(self, mask):
        """ the amount of energy in a timestep that is taken from the distribution grid

        Returns: the energy throughput in kWh for this technology

        """
        if self.duration:
            return self.variables_dict['udis'] * self.dt
        else:
            return super().get_uenergy_decrease(mask)

    def get_state_of_energy(self, mask):
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the state of energy as a function of time for the

        """
        if self.duration:
            return self.variables_dict['ene_load']
        else:
            return super().get_state_of_energy(mask)

    def constraints(self, mask, **kwargs):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """
        constraint_list = []
        if self.duration:
            power = self.variables_dict['power']  # p_t = charge_t - discharge_t
            energy = self.variables_dict['ene_load']
            uene = self.variables_dict['uene']
            udis = self.variables_dict['udis']
            uch = self.variables_dict['uch']

            # constraints that keep the variables inside their limits
            constraint_list += [cvx.NonPos(power - self.rated_power)]
            constraint_list += [cvx.NonPos(-self.rated_power - power)]
            constraint_list += [cvx.NonPos(-energy)]
            constraint_list += [cvx.NonPos(energy - self.operational_max_energy())]
            # uene accounts for change in energy due to participating in sub timestep scale markets
            constraint_list += [cvx.Zero(uene + (self.dt * udis) - (self.dt * uch))]

            sub = mask.loc[mask]
            for day in sub.index.dayofyear.unique():
                day_mask = (day == sub.index.dayofyear)
                # general:  e_{t+1} = e_t + (charge_t - discharge_t) * dt = e_t + power_t * dt
                constraint_list += [cvx.Zero(energy[day_mask][:-1] + (power[day_mask][:-1] * self.dt) - energy[day_mask][1:])]
                # start of first timestep of the day
                constraint_list += [cvx.Zero(energy[day_mask][0] - self.operational_max_energy())]
                # end of the last timestep of the day
                constraint_list += [cvx.Zero(energy[day_mask][-1] + (power[day_mask][-1] * self.dt) - self.operational_max_energy())]

        return constraint_list

    def effective_load(self):
        """ Returns the load that is seen by the microgrid or point of interconnection

        """
        effective_load = super().effective_load()
        if self.duration:
            effective_load += self.variables_df.loc[:, 'power']
        return effective_load

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        results = super().timeseries_report()
        if self.duration:
            results[f"{self.unique_tech_id()} Load (kW)"] = self.effective_load()
            solve_dispatch_opt = self.variables_df.get('power')
            if solve_dispatch_opt is not None:
                results[f"{self.unique_tech_id()} Load Offset (kW)"] = \
                    self.variables_df.loc[:, 'power']
        return results

    def sizing_summary(self):
        """

        Returns: A dictionary describe this DER's size and captial costs.

        """
        sizing_results = {
            'DER': self.name,
            'Power Capacity (kW)': self.rated_power,
            'Duration (hours)': self.duration
        }
        return sizing_results

    def max_p_schedule_down(self):
        # ability to provide regulation down through discharging less
        if not self.duration:
            return super().max_p_schedule_down()
        return self.rated_power

    def decommissioning_report(self, last_year):
        """ Returns the cost of decommissioning a DER and the year the cost will be incurred

        Returns: dataframe index by year that the cost applies to. if the year

        """
        if not self.duration:
            return
        return super(ControllableLoad, self).decommissioning_report(last_year)

    def salvage_value_report(self, end_year):
        """ Returns the salvage value a DER and the year it will be incurred

        Args:
                end_year: last year of project

        Returns: dataframe index by year that the value applies to.

        """
        if not self.duration:
            return
        return super(ControllableLoad, self).salvage_value_report(end_year)
