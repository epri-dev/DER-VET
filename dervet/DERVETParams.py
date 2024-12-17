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
Params.py

"""

import numpy as np
from storagevet.Params import Params
import copy
from storagevet.ErrorHandling import *
from storagevet.Library import is_leap_yr
from pathlib import Path

# TODO -- declare these constants in one place for access across DER-VET
KW_PER_TON = 3.5168525  # unit conversion (1 ton in kW)
KW_PER_MMBTU_HR = 293.071107 # unit conversion (1 MMBtu/hr in kW)

class ParamsDER(Params):
    """
        Class attributes are made up of services, technology, and any other needed inputs. The attributes are filled
        by converting the .json or .csv file into a python object.

        Notes:
             Need to change the summary functions for pre-visualization every time the Params class is changed - TN
    """
    # set schema location based on the location of this file (this should override the global value within Params.py
    schema_location = Path(__file__).absolute().with_name('Schema.json')
    cba_input_error_raised = False
    cba_input_template = None
    # TODO add to this as needed --AE
    dervet_only_der_list = ['CT', 'CHP', 'DieselGenset', 'ControllableLoad', 'EV', 'Chiller', 'Boiler', 'ElectrolyzerSystem']

    @staticmethod
    def pandas_to_dict(model_parameter_pd):
        """converts csv to a json--which DERVET can then read directly

        Args:
            model_parameter_pd:

        Returns: dictionary that can be jumped as json in the data structure that DER-VET reads

        """
        json_tree = Params.pandas_to_dict(model_parameter_pd)
        # check if there was an ID column, if not then add one filled with '.'
        if 'ID' not in model_parameter_pd.columns:
            model_parameter_pd['ID'] = np.repeat('', len(model_parameter_pd))
        # check to see if Evaluation rows are included
        if 'Evaluation Value' in model_parameter_pd.columns and 'Evaluation Active' in model_parameter_pd.columns:
            # outer loop for each tag/object and active status, i.e. Scenario, Battery, DA, etc.
            for obj in model_parameter_pd.Tag.unique():
                # select all TAG rows
                tag_sub = model_parameter_pd.loc[model_parameter_pd.Tag == obj]
                # loop through each unique value in ID
                for id_str in tag_sub.ID.unique():
                    # select rows with given ID_STR
                    id_tag_sub = tag_sub.loc[tag_sub.ID == id_str]
                    # middle loop for each object's elements and is sensitivity is needed: max_ch_rated, ene_rated, price, etc.
                    for _, row in id_tag_sub.iterrows():
                        # skip adding to JSON if no value is given
                        if row['Key'] is np.nan or row['Evaluation Value'] == '.' or row['Evaluation Active'] == '.':
                            continue
                        else:
                            key_attrib = json_tree['tags'][obj][str(id_str)]['keys'][row['Key']]
                            key_attrib['evaluation'] = {
                                "active": str(row['Evaluation Active']),
                                "value": str(row['Evaluation Value'])
                            }
        return json_tree

    @classmethod
    def initialize(cls, filename, verbose):
        """ In addition to everything that initialize does in Params, this class will look at
        Evaluation Value to - 1) determine if cba value can be given and validate; 2) convert
        any referenced data into direct data 3) if sensitivity analysis, then make sure enough
        cba values are given 4) build a dictionary of CBA inputs that match with the instances
        that need to be run

            Args:
                filename (string): filename of JSON or CSV model parameter
                verbose (bool): whether or not to print to console for more feedback

            Returns dictionary of instances of Params, each key is a number
        """
        cls.instances = super().initialize(filename, verbose)  # everything that initialize does in Params (steps 1-4)
        # 1) INITIALIZE CLASS VARIABLES
        cls.sensitivity['cba_values'] = dict()
        cls.cba_input_error_raised = False

        # 5) load direct data and create input template
        # determine if cba value can be given and validate
        cls.cba_input_template = cls.cba_template_struct()

        # convert any referenced data into direct data (add referenced data to dict DATASETS)
        cls.read_evaluation_referenced_data()

        # report back any warning associated with the 'Evaulation' column
        if cls.cba_input_error_raised:
            TellUser.close_log()
            raise ModelParameterError("The model parameter has some errors associated to it in the CBA column. Please fix and rerun.")

        # 6) if SA, update case definitions to define which CBA values will apply for each case
        cls.add_evaluation_to_case_definitions()

        # 7) build a dictionary of CBA inputs that matches with the instance Params that the inputs should be paired with and
        # load up datasets that correspond with referenced data in respective cba_input_instance (as defined by CASE_DEFINITIONS)
        # distribute CBA dictionary of inputs to the corresponding Param instance (so its value can be passed on to Scenario)
        cls.cba_input_builder()
        return cls.instances

    def __init__(self):
        """ Initialize these following attributes of the empty Params class object.
        """
        super().__init__()
        self.Reliability = self.read_and_validate('Reliability')  # Value Stream
        self.ControllableLoad = self.read_and_validate('ControllableLoad')
        self.DieselGenset = self.read_and_validate('DieselGenset')
        self.CT = self.read_and_validate('CT')
        self.CHP = self.read_and_validate('CHP')
        self.ElectricVehicle1 = self.read_and_validate('ElectricVehicle1')
        self.ElectricVehicle2 = self.read_and_validate('ElectricVehicle2')
        self.Chiller = self.read_and_validate('Chiller')
        self.Boiler = self.read_and_validate('Boiler')
        self.ElectrolyzerSystem = self.read_and_validate('ElectrolyzerSystem')

    @classmethod
    def bad_active_combo(cls):
        """ Based on what the user has indicated as active (and what the user has not), predict whether or not
        the simulation(s) will have trouble solving.

        Returns (bool): True if there are errors found. False if there are no errors found in the errors log.

        """
        slf = cls.template
        other_ders = any([len(slf.CHP), len(slf.CT), len(slf.DieselGenset),
            len(slf.ElectricVehicle1), len(slf.ElectricVehicle2), len(slf.Chiller),
            len(slf.Boiler), len(slf.ElectrolyzerSystem)])
        return super().bad_active_combo(dervet=True, other_ders=other_ders)

    @classmethod
    def cba_template_struct(cls):
        """

        Returns: a template structure that summarizes the inputs for a CBA instance

        """
        template = dict()
        template['Scenario'] = cls.read_and_validate_evaluation('Scenario')
        template['Finance'] = cls.read_and_validate_evaluation('Finance')

        # create dictionary for CBA values for DERs
        template['ders_values'] = {
            'Battery': cls.read_and_validate_evaluation('Battery'),
            'CAES': cls.read_and_validate_evaluation('CAES'),
            'CT': cls.read_and_validate_evaluation('CT'),
            'CHP': cls.read_and_validate_evaluation('CHP'),
            'PV': cls.read_and_validate_evaluation('PV'),  # cost_per_kW (and then recalculate capex)
            'ICE': cls.read_and_validate_evaluation('ICE'),  # fuel_price,
            'DieselGenset': cls.read_and_validate_evaluation('DieselGenset'),  # fuel_price,
            'ElectricVehicle1': cls.read_and_validate_evaluation('ElectricVehicle1'),
            'ElectricVehicle2': cls.read_and_validate_evaluation('ElectricVehicle2'),
            'Chiller': cls.read_and_validate_evaluation('Chiller'),
            'Boiler': cls.read_and_validate_evaluation('Boiler'),
            'ControllableLoad': cls.read_and_validate_evaluation('ControllableLoad'),
            'ElectrolyzerSystem': cls.read_and_validate_evaluation('ElectrolyzerSystem'),
        }

        # create dictionary for CBA values for all services (from data files)
        template['valuestream_values'] = {'User': cls.read_and_validate_evaluation('User'),  # only have one entry in it (key = price)
                                          'Deferral': cls.read_and_validate_evaluation('Deferral')}
        return template

    @classmethod
    def read_and_validate_evaluation(cls, name):
        """ Read data from valuation JSON file

        Args:
            name (str): name of root element in json file

        Returns: A dictionary where keys are the ID value and the key is a dictionary
            filled with values provided by user that will be used by the CBA class
            or None if no values are active.

        """
        if '.json' == cls.filename.suffix:
            return cls.read_and_validate_evaluation_json(name)

    @classmethod
    def read_and_validate_evaluation_json(cls, name):
        """ Read data from valuation json file

        Args:
            name (str): name of root element in json file

        Returns: A dictionary where keys are the ID value and the key is a dictionary
            filled with values provided by user that will be used by the CBA class
            or None if no values are active.

        """
        schema_tag = cls.schema_dct.get("tags").get(name)
        # Check if tag is in schema (SANITY CHECK)
        if schema_tag is None:
            # cls.report_warning("missing tag", tag=name, raise_input_error=False)
            # warn user that the tag given is not in the schema
            return
        # check to see if user includes the tag within the provided json
        user_tag = cls.json_tree.get(name)
        if user_tag is None:
            return
        tag_data_struct = {}
        for tag_id, tag_attrib in user_tag.items():
            # This statement checks if the first character is 'y' or '1', if true it creates a dictionary.
            active_tag = tag_attrib.get('active')
            if active_tag is not None and (active_tag[0].lower() == "y" or active_tag[0] == "1"):
                dictionary = {}
                # grab the user given keys
                user_keys = tag_attrib.get('keys')
                # iterate through each key required by the schema
                schema_key_dict = schema_tag.get("keys")
                for schema_key_name, schema_key_attr in schema_key_dict.items():
                    key_attrib = user_keys.get(schema_key_name)
                    if key_attrib is None:
                        # look for optional key and use that here
                        try:
                            optional_key = cls.schema_dct['tags'][name]['keys'][schema_key_name]['optional']
                            key_attrib = user_keys.get(optional_key)
                        except (KeyError):
                            continue
                        if key_attrib is None:
                            continue
                    cba_value = key_attrib.get('evaluation')
                    # if we dont have a cba_value, skip to next key
                    if cba_value is None:
                        continue
                    # did the user mark cba input as active?
                    cba_active = cba_value.get('active')
                    if cba_active[0].lower() in ["y", "1"]:
                        # check if you are allowed to input Evaulation value for the give key
                        cba_allowed = schema_key_attr.get('cba')
                        if cba_allowed is None or cba_allowed[0].lower() in ['n', '0']:
                            cls.report_warning('cba not allowed', tag=name, key=schema_key_name, raise_input_error=False)
                            continue
                        else:
                            valuation_entry = None
                            intended_type = key_attrib.get('type')
                            key_sensitivity = key_attrib.get('sensitivity')
                            if key_sensitivity is not None and key_sensitivity.get('active', 'no')[0].lower() in ["y", "1"]:
                                # if analysis, then convert each value and save as list
                                tag_key = (name, schema_key_name, tag_id)
                                sensitivity_values = cls.extract_data(cba_value.get('value'), intended_type)
                                # validate each value
                                for values in sensitivity_values:
                                    cls.checks_for_validate(values, schema_key_attr, schema_key_name, f"{name}-{tag_id}")

                                #  check to make sure the length match with sensitivity analysis value set length
                                required_values = len(cls.sensitivity['attributes'][tag_key])
                                if required_values != len(sensitivity_values):
                                    cls.report_warning('cba sa length', tag=name, key=schema_key_name, required_num=required_values)
                                cls.sensitivity['cba_values'][tag_key] = sensitivity_values
                            else:
                                # convert to correct data type
                                valuation_entry = cls.convert_data_type(cba_value.get('value'), intended_type)
                                cls.checks_for_validate(valuation_entry, schema_key_attr, schema_key_name, f"{name}-{tag_id}")
                            # save evaluation value OR save a place for the sensitivity value to fill in the dictionary later w/ None
                            dictionary[schema_key_name] = valuation_entry
                # save set of KEYS (in the dictionary) to the TAG that it belongs to (multiple dictionaries if mutliple IDs)
                tag_data_struct[tag_id] = dictionary
        return tag_data_struct

    @classmethod
    def report_warning(cls, warning_type, raise_input_error=True, **kwargs):
        """ Print a warning to the user log. Warnings are reported, but do not result in exiting.

        Args:
            warning_type (str): the classification of the warning to be reported to the user
            raise_input_error (bool): raise this warning as an error instead back to the user and stop running
                the program
            kwargs: elements about the warning that need to be reported to the user (like the tag and key that
                caused the error

        """
        if warning_type == "too many tags":
            TellUser.error(f"INPUT: There are {kwargs['length']} {kwargs['tag']}'s, but only {kwargs['max']} can be defined")

        if warning_type == 'cba not allowed':
            TellUser.error(f"INPUT: {kwargs['tag']}-{kwargs['key']} is not be used within the " +
                           "CBA module of the program. Value is ignored.")
            cls.cba_input_error_raised = raise_input_error or cls.cba_input_error_raised
        if warning_type == "cba sa length":
            cls.cba_input_error_raised = raise_input_error or cls.cba_input_error_raised
            TellUser.error(f"INPUT: {kwargs['tag']}-{kwargs['key']} has not enough CBA evaluatino values to "
                           f"successfully complete sensitivity analysis. Please include {kwargs['required_num']} "
                           f"values, each corresponding to the Sensitivity Analysis value given")
        super().report_warning(warning_type, raise_input_error, **kwargs)

    @classmethod
    def read_evaluation_referenced_data(cls):
        """ This function makes a unique set of filename(s) based on grab_evaluation_lst and the data already read into REFERENCED_DATA.
            It applies for time series filename(s), monthly data filename(s), customer tariff filename(s).
            For each set, the corresponding class dataset variable (ts, md, ct) is loaded with the data.

            Preprocess monthly data files

        """

        ts_files = cls.grab_evaluation_lst('Scenario', 'time_series_filename') - set(cls.referenced_data['time_series'].keys())
        md_files = cls.grab_evaluation_lst('Scenario', 'monthly_data_filename') - set(cls.referenced_data['monthly_data'].keys())
        ct_files = cls.grab_evaluation_lst('Finance', 'customer_tariff_filename') - set(cls.referenced_data['customer_tariff'].keys())
        yr_files = cls.grab_evaluation_lst('Finance', 'yearly_data_filename') - set(cls.referenced_data['yearly_data'].keys())

        for ts_file in ts_files:
            cls.referenced_data['time_series'][ts_file] = cls.read_from_file('time_series', ts_file, 'Datetime (he)')
        for md_file in md_files:
            cls.referenced_data['monthly_data'][md_file] = cls.read_from_file('monthly_data', md_file, ['Year', 'Month'])
        for ct_file in ct_files:
            cls.referenced_data['customer_tariff'][ct_file] = cls.read_from_file('customer_tariff', ct_file, 'Billing Period')
        for yr_file in yr_files:
            cls.referenced_data['yearly_data'][yr_file] = cls.read_from_file('yearly_data', yr_file, 'Year')

    @classmethod
    def grab_evaluation_lst(cls, tag, key):
        """ Checks if the tag-key exists in cls.sensitivity, otherwise grabs the base case value
        from cls.template

        Args:
            tag (str):
            key (str):

        Returns: set of values

        """
        values = []

        tag_dict = cls.cba_input_template.get(tag)
        if tag_dict is not None:
            for id_str in tag_dict.keys():
                try:
                    values += list(cls.sensitivity['cba_values'][(tag, key, id_str)])
                except KeyError:
                    try:
                        values += [cls.cba_input_template[tag][id_str][key]]
                    except KeyError:
                        pass
        return set(values)

    @classmethod
    def add_evaluation_to_case_definitions(cls):
        """ Method that adds the 'Evaluation' values as a column to the dataframe that defines the differences in the cases
        being run.

        """
        cba_sensi = cls.sensitivity['cba_values']
        # for each tag-key cba value that sensitivity analysis applies to
        for tag_key, value_lst in cba_sensi.items():
            # initialize the new column with 'nan'
            cls.case_definitions[f"CBA {tag_key}"] = 'nan'
            # get the number of values that you will need to iterate through
            num_cba_values = len(value_lst)
            # for each index in VALUE_LST
            for index in range(num_cba_values):
                corresponding_opt_value = cls.sensitivity['attributes'][tag_key][index]
                # find the row(s) that contain the optimization value that was also the INDEX-th value in the Sensitivity Parameters entry
                cls.case_definitions.loc[cls.case_definitions[tag_key] == corresponding_opt_value, f"CBA {tag_key}"] = value_lst[index]

        # check for any entries w/ nan to make sure everything went fine
        if np.any(cls.case_definitions == 'nan'):
            TellUser.debug('There are some left over nans in the case definition. Something went wrong.')

    @classmethod
    def cba_input_builder(cls):
        """
            Function to create all the possible combinations of inputs to correspond to the
            sensitivity analysis case being run

        """
        # while case definitions is not an empty df (there is SA)
        # or if it is the last row in case definitions
        for index, case in cls.instances.items():
            cba_dict = copy.deepcopy(cls.cba_input_template)
            # check to see if there are any CBA values included in case definition
            # OTHERWISE just read in any referenced data
            for tag_key_id in cls.sensitivity['cba_values'].keys():
                row = cls.case_definitions.iloc[index]
                # modify the case dictionary
                if tag_key_id[0] in cls.cba_input_template['ders_values'].keys():
                    cba_dict['ders_values'][tag_key_id[0]][tag_key_id[2]][tag_key_id[1]] = row.loc[f"CBA {tag_key_id}"]
                elif tag_key_id[0] in cls.cba_input_template['valuestream_values'].keys():
                    cba_dict['valuestream_values'][tag_key_id[0]][tag_key_id[2]][tag_key_id[1]] = row.loc[f"CBA {tag_key_id}"]
                else:
                    cba_dict[tag_key_id[0]][tag_key_id[2]][tag_key_id[1]] = row.loc[f"CBA {tag_key_id}"]
            # flatten dictionaries for VS, Scenario, and Fiances & prepare referenced data
            cba_dict = case.load_values_evaluation_column(cba_dict)
            cls.instances[index].Finance['CBA'] = cba_dict

    def load_values_evaluation_column(self, cba_dict):
        """ Flattens each tag that the Schema has defined to only have 1 allowed. Loads data sets
         that are specified by the '_filename' parameters

        Returns a params class where the tag attributes that are not allowed to have more than one
        set of key inputs are just dictionaries of their key inputs (while the rest remain
        dictionaries of the sets of key inputs)
        """
        freq, dt, opt_years = \
            self.Scenario['frequency'], self.Scenario['dt'], self.Scenario['opt_years']
        cba_dict['Scenario'] = self.flatten_tag_id(cba_dict['Scenario'])
        cba_dict['Finance'] = self.flatten_tag_id(cba_dict['Finance'])
        cba_dict['valuestream_values']['User'] = \
            self.flatten_tag_id(cba_dict['valuestream_values']['User'])
        cba_dict['valuestream_values']['Deferral'] = \
            self.flatten_tag_id(cba_dict['valuestream_values']['Deferral'])

        scenario = cba_dict['Scenario']
        scenario['frequency'] = freq
        if 'time_series_filename' in scenario.keys():
            time_series = self.referenced_data['time_series'][scenario['time_series_filename']]
            scenario["time_series"] = \
                self.process_time_series(time_series, freq, dt, opt_years)
        if 'monthly_data_filename' in scenario.keys():
            raw_monthly_data = self.referenced_data["monthly_data"][scenario["monthly_data_filename"]]
            scenario["monthly_data"] = \
                self.process_monthly(raw_monthly_data, opt_years)

        finance = cba_dict['Finance']
        if 'yearly_data_filename' in finance.keys():
            finance["yearly_data"] = \
                self.referenced_data["yearly_data"][finance["yearly_data_filename"]]
        if 'customer_tariff_filename' in finance.keys():
            finance["customer_tariff"] = \
                self.referenced_data["customer_tariff"][finance["customer_tariff_filename"]]

        # FIXME: add similar treatment for Technology filenames here?
        return cba_dict

    def load_scenario(self):
        """ Interprets user given data and prepares it for Scenario initialization
            Adds load dump boolean to POI
        """
        super().load_scenario()
        self.POI.update({'active_load_dump': self.Scenario['activate_electricity_load_dump']})

    def load_finance(self):
        """ Interprets user given data and prepares it for Finance.

        """
        super().load_finance()
        self.Finance.update({'location': self.Scenario['location'],
                             'ownership': self.Scenario['ownership']})

    def load_technology(self, names_list=None):
        """ Interprets user given data and prepares it for each technology.

        """
        def get_min_opt_period_hours():
            dt = float(self.Scenario['dt'])
            n = self.Scenario['n']
            opt_years = self.Scenario['opt_years']
            opt_occurs_in_a_nonleap_year = False
            for y in np.array([opt_years]).flatten().tolist():
                if not is_leap_yr(int(y)):
                    opt_occurs_in_a_nonleap_year = True
            if n == 'year':
                if opt_occurs_in_a_nonleap_year:
                    min_hours_in_opt = 365 * 24
                else:
                    min_hours_in_opt = 366 * 24
            elif n == 'month':
                if opt_occurs_in_a_nonleap_year:
                    min_hours_in_opt = 28 * 24
                else:
                    min_hours_in_opt = 29 * 24
            else: # n is a number 0-8760
                 min_hours_in_opt = int(n)
            return min_hours_in_opt

        def non_zero_time_series_data_exists(col, ts):
            if col in ts.columns and not (ts[col] == 0).all():
                return True
            return False

        time_series = self.Scenario['time_series']
        time_series_nan_count = self.Scenario['time_series_nan_count']
        dt = self.Scenario['dt']
        binary = self.Scenario['binary']

        if names_list is None:
            # then no name_lst was inherited so initialize as list type
            names_list = []

        for id_str, pv_inputs in self.PV.items():
            if not pv_inputs['rated_capacity']:
                if pv_inputs['min_rated_capacity'] > pv_inputs['max_rated_capacity']:
                    self.record_input_error('Error: maximum rated power is less than the minimum rated power.' +
                                            f"PV {id_str}")
        for id_str, battery_inputs in self.Battery.items():
            if battery_inputs['state_of_health'] > battery_inputs['cycle_life_table_eol_condition']:
                self.record_input_error(f"Battery #{id_str} state_of_health > cycle_life_table_eol_condition. SOH input should be lesser than eol condition used to create cycle life table for accurate degradation calculation")

            if not battery_inputs['ch_max_rated'] or not battery_inputs['dis_max_rated']:
                if battery_inputs['incl_degradation']:
                    self.record_input_error(
                        f'Error: BATTERY {id_str}: Degradation with power sizing is still under ' +
                        f'development. Please choose to do one or the other.')
                if not battery_inputs['ch_max_rated']:
                    if battery_inputs['user_ch_rated_min'] > battery_inputs['user_ch_rated_max']:
                        self.record_input_error('Error: User battery min charge power requirement is greater than max charge power requirement.' +
                                                f"BATTERY {id_str}")
                if not battery_inputs['dis_max_rated']:
                    if battery_inputs['user_dis_rated_min'] > battery_inputs['user_dis_rated_max']:
                        self.record_input_error('User battery min discharge power requirement is greater than max discharge power requirement.')
            if not battery_inputs['ene_max_rated']:
                if battery_inputs['incl_degradation']:
                    self.record_input_error(
                        f'Error: BATTERY {id_str}: Degradation with energy sizing is still under' +
                        f' development. Please choose to do one or the other.')
                if battery_inputs['user_ene_rated_min'] > battery_inputs['user_ene_rated_max']:
                    self.record_input_error('Error: User battery min energy requirement is greater than max energy requirement.')
            # check if user wants to include timeseries constraints -> grab data
            if battery_inputs['incl_ts_energy_limits']:
                self.load_ts_limits(id_str, battery_inputs, 'Battery', 'Energy', 'kWh', time_series, time_series_nan_count)
            if battery_inputs['incl_ts_charge_limits']:
                self.load_ts_limits(id_str, battery_inputs, 'Battery', 'Charge', 'kW', time_series, time_series_nan_count)
            if battery_inputs['incl_ts_discharge_limits']:
                self.load_ts_limits(id_str, battery_inputs, 'Battery', 'Discharge', 'kW', time_series, time_series_nan_count)

        for id_str, caes_inputs in self.CAES.items():
            # check if user wants to include timeseries constraints -> grab data
            if caes_inputs['incl_ts_energy_limits']:
                self.load_ts_limits(id_str, caes_inputs, 'CAES', 'Energy', 'kWh', time_series, time_series_nan_count)
            if caes_inputs['incl_ts_charge_limits']:
                self.load_ts_limits(id_str, caes_inputs, 'CAES', 'Charge', 'kW', time_series, time_series_nan_count)
            if caes_inputs['incl_ts_discharge_limits']:
                self.load_ts_limits(id_str, caes_inputs, 'CAES', 'Discharge', 'kW', time_series, time_series_nan_count)

        for id_str, load_inputs in self.ControllableLoad.items():
            # check to make sure data was included
            col_name = "Site Load (kW)"
            load_inputs['site_load'] = self.get_single_series(time_series, col_name, time_series_nan_count, 'Site Load', id_str=id_str)
            load_inputs.update({'dt': dt,
                                'growth': self.Scenario['def_growth']})

        for id_str, ev1_input in self.ElectricVehicle1.items():
            # max ratings should not be greater than the min rating for power and energy
            if ev1_input['ch_min_rated'] > ev1_input['ch_max_rated']:
                self.record_input_error(f"EV1 #{id_str} ch_max_rated < ch_min_rated. ch_max_rated should be greater than ch_min_rated")
            ev1_input.update({'binary': binary,
                              'dt': dt})
            names_list.append(ev1_input['name'])

        for id_str, ev_input in self.ElectricVehicle2.items():
            # should we have a check for time series data?
            ev_input.update({'binary': binary,
                             'dt': dt})
            names_list.append(ev_input['name'])
            ev_input.update({'EV_baseline': self.get_single_series(time_series, f'EV fleet/{id_str}', time_series_nan_count, 'EV Load')})

        if len(self.CHP):
            for id_str, chp_inputs in self.CHP.items():
                chp_inputs.update({'dt': dt})
                # add time series, monthly data, and any scenario case parameters to CHP parameter dictionary
                # TODO: we allow for multiple CHPs to be defined -- and if there were -- then they all would share the same data.
                #       Is this correct? --HN; yes --AE
                chp_inputs.update({'site_steam_load':
                    self.get_single_series(time_series, 'Site Steam Thermal Load (MMBtu/hr)', time_series_nan_count, bypass_key_error=True, unit_conversion=KW_PER_MMBTU_HR)})
                chp_inputs.update({'site_hotwater_load':
                    self.get_single_series(time_series, 'Site Hot Water Thermal Load (MMBtu/hr)', time_series_nan_count, bypass_key_error=True, unit_conversion=KW_PER_MMBTU_HR)})

                # report error when thermal load has neither steam nor hotwater components
                if chp_inputs.get('site_steam_load') is None and chp_inputs.get('site_hotwater_load') is None:
                    self.record_input_error("CHP is missing required site heating load time series input data. 'Site Steam Thermal Load (MMBtu/hr)' and/or 'Site Hot Water Thermal Load (MMBtu/hr)' must exist as time series data inputs.")
                # when only one thermal load exists (steam or hotwater), make the other one with zeroes and warn
                elif chp_inputs.get('site_steam_load') is None or chp_inputs.get('site_hotwater_load') is None:
                    if chp_inputs.get('site_steam_load') is None:
                        all_zeroes = chp_inputs['site_hotwater_load'].copy()
                        all_zeroes.values[:] = 0
                        chp_inputs.update({'site_steam_load': all_zeroes})
                        TellUser.warning('Thermal heating loads are input as separate hot water and steam time series. Since we only found site hot water thermal load data, we assume a site steam thermal load time series of all zeroes.')
                    if chp_inputs.get('site_hotwater_load') is None:
                        all_zeroes = chp_inputs['site_steam_load'].copy()
                        all_zeroes.values[:] = 0
                        chp_inputs.update({'site_hotwater_load': all_zeroes})
                        TellUser.warning('Thermal heating loads are input as separate hot water and steam time series. Since we only found site steam thermal load data, we assume a site hot water thermal load time series of all zeroes.')

        if len(self.CT):
            for id_str, ct_inputs in self.CT.items():
                ct_inputs.update({'dt': dt})

        if len(self.ElectrolyzerSystem):
            for id_str, electrolyzer_inputs in self.ElectrolyzerSystem.items():
                electrolyzer_inputs.update({'binary': binary,
                                  'dt': dt})
                names_list.append(electrolyzer_inputs['name'])
                # min rating should not be greater than the max rating for power
                if electrolyzer_inputs['min_rated_power'] > electrolyzer_inputs['max_rated_power']:
                    self.record_input_error(f"ElectrolyzerSystem #{id_str} max_rated_power < min_rated_power. max_rated_power ({electrolyzer_inputs['max_rated_power']}) should be greater than min_rated_power ({electrolyzer_inputs['min_rated_power']})")
                # upload a hydrogen schedule timeseries if the flag is set to 1
                if electrolyzer_inputs['hydrogen_schedule_mode'] == 1:
                    single_instance = True
                    if len(self.ElectrolyzerSystem) > 1:
                        single_instance = False
                    electrolyzer_inputs.update({'ts_hydrogen_schedule': self.get_single_series(time_series, 'ElectrolyzerSystem Schedule (kW)', time_series_nan_count, 'ElectrolyzerSystem Load', id_str=id_str, single_instance=single_instance)})
                    # do not allow sizing in this scenario
                    if electrolyzer_inputs['rated_power'] == 0:
                        self.record_input_error(f'ElectrolyzerSystem #{id_str} power sizing is turned ON, but we have declared a fixed operation of the system by setting hydrogen_schedul_mode to 1. This is incompatible; one or the other must be changed.')
                # ensure that the ElectrolyzerSystem size is large enough to meet the
                #   hydrogen production quota at 100 percent capacity factor
                if electrolyzer_inputs['hydrogen_schedule_mode'] == 2 and electrolyzer_inputs['rated_power']:
                    if electrolyzer_inputs['hydrogen_production_quota'] > \
                        electrolyzer_inputs['rated_power'] * electrolyzer_inputs['efficiency'] * get_min_opt_period_hours():
                        self.record_input_error(f'ElectrolyzerSystem #{id_str} is unable to meet the hydrogen production quota for the minimum optimization period (hours) while operating at 100% capacity factor: hydrogen_production_quota > rated_power * efficiency * min_opt_period_hours. Here, the min_opt_period_hours = {get_min_opt_period_hours()}, while the hydrogen_production_quota = {electrolyzer_inputs["hydrogen_production_quota"]} kg')
                # warn if there are POI electric power import restrictions
                #   as these could conflict with hydrogen production scheduling/quotas
                #   resulting in an infeasibility of the optimization problem
                if self.Scenario['apply_interconnection_constraints'] and \
                    electrolyzer_inputs['hydrogen_schedule_mode'] in [1,2]:
                    TellUser.warning('There are POI electric power import restrictions set (see the Scenario input max_import) which may conflict with hydrogen production scheduling/quotas. If such a conflict occurs, the optimization problem becomes infeasible, and must be modified to get a result.')
                # upload a variable efficiency curve if the flag is set to 1
                if electrolyzer_inputs['incl_variable_efficiency']:
                    try:
                        electrolyzer_inputs['variable_efficiency_data'] = self.referenced_data['variable_efficiency_filename'][electrolyzer_inputs['variable_efficiency_filename']]
                    except KeyError:
                        self.record_input_error(f"Missing variable_efficiency_filename for ElectrolyzerSystem #{id_str}: {electrolyzer_inputs['variable_efficiency_filename']} Please include a variable_efficiency_filename")
                    # make sure we have the required columns
                    if electrolyzer_inputs['variable_efficiency_data'] is not None:
                        for required_column in ['Efficiency (kg/kWh)', 'Fractional Power (%)']:
                            if required_column not in electrolyzer_inputs['variable_efficiency_data'].columns:
                                self.record_input_error(
                                        f"'{required_column}' is missing from {electrolyzer_inputs['variable_efficiency_filename']}")
                # upload a degradation curve if the incl_degradation flag is set to 1
                if electrolyzer_inputs['incl_degradation']:
                    try:
                        electrolyzer_inputs['cycle_life_data'] = self.referenced_data['cycle_life_filename'][electrolyzer_inputs['cycle_life_filename']]
                    except KeyError:
                        self.record_input_error(f"Missing cycle_life_filename for ElectrolyzerSystem #{id_str}: {electrolyzer_inputs['cycle_life_filename']} Please include a cycle_life_filename")
                    # make sure we have the required columns
                    if electrolyzer_inputs['cycle_life_data'] is not None:
                        for required_column in ['Power Fluctuation (%)', 'Cycle Life Value (cycles)']:
                            if required_column not in electrolyzer_inputs['cycle_life_data'].columns:
                                self.record_input_error(
                                        f"'{required_column}' is missing from {electrolyzer_inputs['cycle_life_filename']}")
                # electrolyzer degradation-related warnings
                if not electrolyzer_inputs['rated_power'] and electrolyzer_inputs['incl_degradation']:
                    self.record_input_error(
                        f'Error: ELECTROLYZERSYSTEM {id_str}: Degradation with power sizing is still under ' +
                        f'development. Please choose to do one or the other.')
        if len(self.DieselGenset):
            for id_str, inputs in self.DieselGenset.items():
                inputs.update({'dt': dt})

        if len(self.Chiller):
            for id_str, chiller_input in self.Chiller.items():
                chiller_input.update({'dt': dt})
                # add time series to Chiller parameter dictionary
                chiller_input.update({'site_cooling_load':
                    self.get_single_series(time_series, 'Site Cooling Thermal Load (tons)', time_series_nan_count, bypass_key_error=True, unit_conversion=KW_PER_TON)})
                # report error when thermal load does not have a cooling load
                if chiller_input.get('site_cooling_load') is None:
                    self.record_input_error("Chiller is missing a site cooling load ('Site Cooling Thermal Load (tons)' from timeseries data input")
                    self.record_input_error("Chiller is missing required site cooling load time series input data. 'Site Cooling Thermal Load (tons)' must exist as a time series data input.")
                # if the chiller is powered by heat, then an active Boilder or CHP is required
                if chiller_input['power_source'] == 'heat' and not (len(self.Boiler) + len(self.CHP)):
                    self.record_input_error(f"Chiller ({chiller_input['name']}) is powered by heat, but there is not an active Boiler or CHP to provide this heat.")
        else:
            # without an active Chiller, report error if a cooling load time series exists
            #     (instead of silently ignoring it)
            if non_zero_time_series_data_exists('Site Cooling Thermal Load (tons)', time_series):
                self.record_input_error("A non-zero site cooling load time series ('Site Cooling Thermal Load (tons)') exists, but there are no active Chillers. Please activate a Chiller, or remove these data from the input time series.")

        if len(self.Boiler):
            for id_str, boiler_inputs in self.Boiler.items():
                boiler_inputs.update({'dt': dt})
                # add time series, monthly data, and any scenario case parameters to boiler parameter dictionary
                # TODO: we allow for multiple boilers to be defined -- and if there were -- then they all would share the same data.
                #       Is this correct? --HN; yes --AE
                boiler_inputs.update({'site_steam_load':
                    self.get_single_series(time_series, 'Site Steam Thermal Load (MMBtu/hr)', time_series_nan_count, bypass_key_error=True, unit_conversion=KW_PER_MMBTU_HR)})
                boiler_inputs.update({'site_hotwater_load':
                    self.get_single_series(time_series, 'Site Hot Water Thermal Load (MMBtu/hr)', time_series_nan_count, bypass_key_error=True, unit_conversion=KW_PER_MMBTU_HR)})
                # report error when thermal load has neither steam nor hotwater components
                if boiler_inputs.get('site_steam_load') is None and boiler_inputs.get('site_hotwater_load') is None:
                    self.record_input_error("Boiler is missing required site heating load time series input data. 'Site Steam Thermal Load (MMBtu/hr)' and/or 'Site Hot Water Thermal Load (MMBtu/hr)' must exist as time series data inputs.")
                # when only one thermal load exists (steam or hotwater), make the other one with zeroes and warn
                elif boiler_inputs.get('site_steam_load') is None or boiler_inputs.get('site_hotwater_load') is None:
                    if boiler_inputs.get('site_steam_load') is None:
                        all_zeroes = boiler_inputs['site_hotwater_load'].copy()
                        all_zeroes.values[:] = 0
                        boiler_inputs.update({'site_steam_load': all_zeroes})
                        TellUser.warning('Thermal heating loads are input as separate hot water and steam time series. Since we only found site hot water thermal load data, we assume a site steam thermal load time series of all zeroes.')
                    if boiler_inputs.get('site_hotwater_load') is None:
                        all_zeroes = boiler_inputs['site_steam_load'].copy()
                        all_zeroes.values[:] = 0
                        boiler_inputs.update({'site_hotwater_load': all_zeroes})
                        TellUser.warning('Thermal heating loads are input as separate hot water and steam time series. Since we only found site steam thermal load data, we assume a site hot water thermal load time series of all zeroes.')

        # without an active Boiler or CHP, report error if a heating load time series exists
        #     (instead of silently ignoring it)
        if not (len(self.Boiler) + len(self.CHP)):
            if non_zero_time_series_data_exists('Site Steam Thermal Load (MMBtu/hr)', time_series):
                self.record_input_error("A non-zero site heating (steam) load time series ('Site Steam Thermal Load (MMBtu/hr)') exists, but there is no active Boiler or CHP. Please activate a Boiler and/or a CHP, or remove these data from the input time series.")
            if non_zero_time_series_data_exists('Site Hot Water Thermal Load (MMBtu/hr)', time_series):
                self.record_input_error("A non-zero site heating (hot water) load time series ('Site Hot Water Thermal Load (MMBtu/hr)') exists, but there is no active Boiler or CHP. Please activate a Boiler and/or a CHP, or remove these data from the input time series.")

        super().load_technology(names_list)

    def load_ts_limits(self, id_str, inputs_dct, tag, measurement, unit, time_series, time_series_nan_count):
        # check for inconsistencies in timeseries limit columns and values

        def fill_ts_limits(ts_max, ts_min, inputs_dct, measurement, ts_max_name, ts_min_name):
            # fill in empty/nan values, from ts_max and ts_min, with model parameter scalars here
            # if ts is None, then create it and fill it as appropriate
            # report details of fills in log file
            if inputs_dct.get(f'incl_ts_{measurement.lower()}_limits'):
                # create a timeseries with all NaN values using a copy of the first time series available
                empty_ts = next(item for item in [ts_max, ts_min] if item is not None).copy()
                empty_ts.iloc[:] = np.nan
                # determine the fill value; use the required model parameter scalar
                fill_max_value, fill_min_value = None, None
                if measurement.lower() == 'energy':
                    fill_max_value = inputs_dct.get('ene_max_rated')
                    fill_min_value = 0
                elif measurement.lower() == 'charge':
                    fill_max_value = inputs_dct.get('ch_max_rated')
                    fill_min_value = 0
                elif measurement.lower() == 'discharge':
                    fill_max_value = inputs_dct.get('dis_max_rated')
                    fill_min_value = 0
                if ts_max is None:
                    ts_max = empty_ts.copy()
                    ts_max.name = ts_max_name
                if ts_min is None:
                    ts_min = empty_ts.copy()
                    ts_min.name = ts_min_name
                # fill all missing values with the scalar, if necessary, and report on it in the log
                ts_max_nan_count = time_series_nan_count.get(ts_max_name, len(ts_max))
                ts_min_nan_count = time_series_nan_count.get(ts_min_name, len(ts_min))
                if ts_max_nan_count != 0:
                    ts_max.fillna(fill_max_value, inplace=True)
                    TellUser.warning(f"We have filled in {ts_max_nan_count} empty/NaN value(s) from "
                                     f"'{ts_max_name}' with the value: {fill_max_value}")
                if ts_min_nan_count != 0:
                    ts_min.fillna(fill_min_value, inplace=True)
                    TellUser.warning(f"We have filled in {ts_min_nan_count} empty/NaN value(s) from "
                                     f"'{ts_min_name}' with the value: {fill_min_value}")
            # return 2 series
            return ts_max, ts_min

        ts_max_name = f'{tag}: {measurement} Max ({unit})/{id_str}'
        ts_min_name = f'{tag}: {measurement} Min ({unit})/{id_str}'
        ts_max = self.get_single_series(time_series, ts_max_name, time_series_nan_count, bypass_key_error=True, allow_nans=True)
        ts_min = self.get_single_series(time_series, ts_min_name, time_series_nan_count, bypass_key_error=True, allow_nans=True)
        if ts_max is None and ts_min is None:
            # disallow both ts_max and ts_min to be None; at least one must exist to continue
            self.record_timeseries_missing_error(f"'{ts_max_name}' or '{ts_min_name}' is missing " +
                                    "from timeseries input. User indicated one needs to be applied. " +
                                    f"Please include or turn incl_ts_{measurement.lower()}_limits off.")
            return None

        # fill in empty/nan values, from ts_max and ts_min, with model parameter scalars here
        ts_max, ts_min = fill_ts_limits(ts_max, ts_min, inputs_dct, measurement, ts_max_name, ts_min_name)

        if unit == 'kW':
            # perform the following checks on the values in the timeseries
            if ts_max.max() * ts_max.min() < 0:
                # then the max and min are not both positive or both negative -- so error
                self.record_timeseries_data_error(f"'{ts_max_name}' should be all positive or all negative. " +
                                        "Please fix and rerun.")
            if ts_min.max() * ts_min.min() < 0:
                # then the max and min are not both positive or both negative -- so error
                self.record_timeseries_data_error(f"'{ts_min_name}' should be all positive or all negative. " +
                                        "Please fix and rerun.")
        if unit == 'kWh':
            # perform the following checks on the values in the timeseries
            if ts_max.min() < 0:
                self.req_all_non_negative(ts_max.values, ts_max_name)
            if ts_min.min() < 0:
                self.req_all_non_negative(ts_min.values, ts_min_name)

        inputs_dct.update({f'ts_{measurement.lower()}_max': ts_max,
                           f'ts_{measurement.lower()}_min': ts_min})

    def get_single_series(self, time_series, column_name, nan_count, description=None, bypass_key_error=False, allow_nans=False, id_str='', unit_conversion=None, single_instance=False):
        # build on existing method in storagevet
        single_ts = None
        # first attempt to load the time series without the id_str, if single_instance=True
        if single_instance:
            single_ts = super(ParamsDER, self).get_single_series(time_series, column_name, nan_count, description=description, bypass_key_error=True, allow_nans=allow_nans)
        # add id_str to column_name if one is present
        if single_ts is None:
            # attempt to load the time series with the id_str
            if id_str != '':
                column_name = f"{column_name}/{id_str}"
            single_ts = super(ParamsDER, self).get_single_series(time_series, column_name, nan_count, description=description, bypass_key_error=bypass_key_error, allow_nans=allow_nans)
        # perform a unit conversion on all values if called for
        if single_ts is not None and unit_conversion is not None:
            single_ts = single_ts * unit_conversion
        return single_ts

    @classmethod
    def read_referenced_data(cls):
        """ This function makes a unique set of filename(s) based on grab_value_lst.
        It applies for time series filename(s), monthly data filename(s), customer tariff filename(s), and cycle
        life filename(s). For each set, the corresponding class dataset variable (ts, md, ct, cl) is loaded with the data.

        Preprocess monthly data files

        """
        super().read_referenced_data()
        cls.referenced_data['load_shed_percentage'] = dict()
        cls.referenced_data['variable_efficiency_filename'] = dict()
        cls.referenced_data['cycle_life_filename'] = dict()
        rel_files = cls.grab_value_set('Reliability', 'load_shed_perc_filename')
        ee_files = cls.grab_value_set('ElectrolyzerSystem', 'variable_efficiency_filename') # electrolyzer efficiency files
        ed_files = cls.grab_value_set('ElectrolyzerSystem', 'cycle_life_filename') # electrolyzer degradation files

        for rel_file in rel_files:
            cls.referenced_data['load_shed_percentage'][rel_file] = \
                cls.read_from_file('load_shed_percentage', rel_file,'Outage Length (hrs)')
        for ee_file in ee_files:
            cls.referenced_data['variable_efficiency_filename'][ee_file] = \
                cls.read_from_file('variable_efficiency_filename', ee_file)
        for ed_file in ed_files:
            cls.referenced_data['cycle_life_filename'][ed_file] = \
                cls.read_from_file('cycle_life_filename', ed_file)

        return True

    def load_services(self):
        """ Interprets user given data and prepares it for each ValueStream (dispatch and pre-dispatch).

        """
        super().load_services()

        if self.Reliability is not None:
            self.Reliability["dt"] = self.Scenario["dt"]
            try:
                self.Reliability.update({'critical load': self.Scenario['time_series'].loc[:, 'Critical Load (kW)']})
            except KeyError:
                self.record_input_error("Missing 'Critial Load (kW)' from timeseries input. Please include a critical load.")
            if self.Reliability['load_shed_percentage']:
                try:
                    self.Reliability['load_shed_data'] = self.referenced_data["load_shed_percentage"][self.Reliability['load_shed_perc_filename']]
                except KeyError:
                    self.record_input_error("Missing 'Load shed percentage' file . Please include a load_shed_perc_filename") #--TODO length of the data
        # TODO add try statements around each lookup to time_series
        if self.FR is not None:
            if self.FR['u_ts_constraints']:
                self.FR.update({'regu_max': self.Scenario['time_series'].loc[:, 'FR Reg Up Max (kW)'],
                                'regu_min': self.Scenario['time_series'].loc[:, 'FR Reg Up Min (kW)']})
            if self.FR['u_ts_constraints']:
                self.FR.update({'regd_max': self.Scenario['time_series'].loc[:, 'FR Reg Down Max (kW)'],
                                'regd_min': self.Scenario['time_series'].loc[:, 'FR Reg Down Min (kW)']})

        if self.SR is not None:
            if self.SR['ts_constraints']:
                self.SR.update({'max': self.Scenario['time_series'].loc[:, 'SR Max (kW)'],
                                'min': self.Scenario['time_series'].loc[:, 'SR Min (kW)']})

        if self.NSR is not None:
            if self.NSR['ts_constraints']:
                self.NSR.update({'max': self.Scenario['time_series'].loc[:, 'NSR Max (kW)'],
                                 'min': self.Scenario['time_series'].loc[:, 'NSR Min (kW)']})

        if self.LF is not None:
            if self.LF['u_ts_constraints']:
                self.LF.update({'lf_u_max': self.Scenario['time_series'].loc[:, 'LF Reg Up Max (kW)'],
                                'lf_u_min': self.Scenario['time_series'].loc[:, 'LF Reg Up Min (kW)']})
            if self.LF['u_ts_constraints']:
                self.LF.update({'lf_d_max': self.Scenario['time_series'].loc[:, 'LF Reg Down Max (kW)'],
                                'lf_d_min': self.Scenario['time_series'].loc[:, 'LF Reg Down Min (kW)']})

        TellUser.debug("Successfully prepared the value-streams")
