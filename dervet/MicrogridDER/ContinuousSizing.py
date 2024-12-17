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
Continuous Sizing Module

"""
from storagevet.ErrorHandling import *
import numpy as np
import pandas as pd
import cvxpy as cvx


class ContinuousSizing:
    """ This class is to be inherited by DER classes that want to also define the ability
    to optimally size itself by kW of energy capacity

    """

    def __init__(self, params):
        TellUser.debug(f"Initializing {__name__}")
        self.size_constraints = []

    def being_sized(self):
        """ checks itself to see if this instance is being sized

        Returns: true if being sized, false if not being sized

        """
        return bool(len(self.size_constraints))

    def sizing_objective(self):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Returns:
            dict of objective costs
        """
        costs = {}
        if self.being_sized():
            costs[self.name + ' capex'] = self.get_capex()

        return costs

    def set_size(self):
        """ Save value of size variables of DERs

        """
        pass

    def sizing_summary(self):
        """

        Returns: A dictionary describe this DER's size and capital costs.

        """
        # template = pd.DataFrame(columns=)
        # sizing_dict = {
        #     'DER': np.nan,
        #     'Energy Rating (kWh)': np.nan,
        #     'Charge Rating (kW)': np.nan,
        #     'Discharge Rating (kW)': np.nan,
        #     'Round Trip Efficiency (%)': np.nan,
        #     'Lower Limit on SOC (%)': np.nan,
        #     'Upper Limit on SOC (%)': np.nan,
        #     'Duration (hours)': np.nan,
        #     'Capital Cost ($)': np.nan,
        #     'Capital Cost ($/kW)': np.nan,
        #     'Capital Cost ($/kWh)': np.nan,
        #     'Power Capacity (kW)': np.nan,
        #     'Quantity': 1,
        # }
        # return sizing_dict

    def sizing_error(self):
        """

        Returns: True if there is an input error

        """
        return False

    def max_p_schedule_down(self):
        return 0

    def max_p_schedule_up(self):
        return self.max_p_schedule_down()

    def is_discharge_sizing(self):
        return self.being_sized()

    def is_power_sizing(self):
        return self.being_sized()

    def max_power_defined(self):
        return True

