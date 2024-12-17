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
from storagevet.ServiceAggregator import ServiceAggregator
from storagevet.ErrorHandling import *
import cvxpy as cvx


class MicrogridServiceAggregator(ServiceAggregator):
    """ The entity that tracks the value streams and bids the Microgrid's capabilities
    into energy markets

    """

    def post_facto_reliability_only(self):
        """

        Returns: A boolean that is true if Reliability is doing post facto calculations only

        """
        return self.is_reliability_only() and self.post_facto_reliability()

    def post_facto_reliability(self):
        """

        Returns: A boolean that is true if Reliability is doing post facto calculations only when sizing

        """
        return 'Reliability' in self.value_streams.keys() and self.value_streams['Reliability'].post_facto_only

    def post_facto_reliability_only_and_user_defined_constraints(self):
        """

        Returns: A boolean that is true if Reliability is doing post facto and user-defined service is active

        """
        return len(self.value_streams.keys()) == 2 and 'User' in self.value_streams.keys() and self.post_facto_reliability()

    def is_reliability_only(self):
        """

        Returns: A boolean that is true if Reliability is doing post facto and user-defined constraint is true

        """
        return len(self.value_streams.keys()) == 1 and 'Reliability' in self.value_streams.keys()

    def is_whole_sale_market(self):
        """

        Returns: boolean, interect btw list of market services

        """
        return bool({'SR', 'NSR', 'FR', 'LF'} & set(self.value_streams.keys()))

    def set_size(self, der_lst, start_year):
        """ part of Deferral's sizing module: TODO REPLACE WITH set_size IN MICROGRID POI
        iterates over a list of DER+DERExtension objects and sets their minimum size
        based on the P and E requirements set by MIN_YEAR objective.

        Args:
            der_lst:
            start_year:

        Returns: der_list with size minimums

        """
        deferral = self.value_streams.get('Deferral')
        min_year = deferral.min_years
        last_year_to_defer = start_year.year + min_year - 1
        p_e_req = deferral.deferral_df.loc[last_year_to_defer, :]
        min_power = p_e_req.loc['Power Capacity Requirement (kW)']
        min_energy = p_e_req.loc['Energy Capacity Requirement (kWh)']
        if len(self.value_streams.keys()) > 1:
            der_lst[0].size_constraints += [cvx.NonPos(min_energy - der_lst[0].ene_max_rated)]
            der_lst[0].size_constraints += [cvx.NonPos(min_power - der_lst[0].ch_max_rated)]
            der_lst[0].size_constraints += [cvx.NonPos(min_power - der_lst[0].dis_max_rated)]
        else:
            der_lst[0].ch_max_rated = min_power
            der_lst[0].dis_max_rated = min_power
            der_lst[0].ene_max_rated = min_energy
        return der_lst

    def any_max_participation_constraints_not_included(self):
        """

        Returns: true if a max constraint for an active market participation service is not defined

        """
        return bool(sum([1 if not vs.max_participation_is_defined() and name in {'LF', 'SR', 'NSR', 'FR'} else 0 for name, vs in self.value_streams.items()]))
