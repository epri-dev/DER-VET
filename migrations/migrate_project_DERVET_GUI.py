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
migrate_project_DERVET_GUI.py

This Python script is used to convert a DER-VET GUI Project file
from version 1 (used in GUI v1.1) to version 2 (used in GUI v1.2)
"""

import copy, os, json, argparse, sys
from pathlib import Path

PROJECT_FILENAME = 'project.json'

SCHEMA_VERSION = '1.0.0'

COLLECTIONS = [
    "technologySpecsBattery",
    "technologySpecsControllableLoad",
    "technologySpecsDieselGen",
    "technologySpecsFleetEV",
    "technologySpecsICE",
    "technologySpecsSingleEV",
    "technologySpecsSolarPV",
    "externalIncentives",
    "retailTariffBillingPeriods"
]

COLLECTION_FIELDS_TO_REMOVE = [
    "active",
    "associatedInputsComplete",
    "complete",
    "componentSpecsComplete",
    "errorList",
    "id",
    "path",
    "submitted",
    "tag",
    "technologyType"
]

PROJECT_TIMESERIES = [
    "mtsBackupEnergyPrice",
    "mtsBackupEnergyReservation",
    "tsCriticalLoad",
    "tsDaPrice",
    "mtsDrCapacityPrice",
    "mtsDrCapacityReservation",
    "mtsDrEnergyPrice",
    "tsDeferralLoad",
    "tsFrPrice",
    "tsFrUpPrice",
    "tsFrDownPrice",
    "tsLfEOU",
    "tsLfEOD",
    "tsLfPrice",
    "tsLfUpPrice",
    "tsLfDownPrice",
    "tsNsrPrice",
    "tsSiteLoad",
    "tsSrPrice",
    "tsSystemLoad",
    "tsRaActive",
    "mtsRaCapacityPrice",
    "tsUserEnergyMax",
    "tsUserEnergyMin",
    "tsUserPowerExportMax",
    "tsUserPowerExportMin",
    "tsUserPowerImportMax",
    "tsUserPowerImportMin"
]

def dict_from_json_file(filepath: str) -> dict:
    with open(filepath) as f:
        try:
            dct = json.load(f)
        except Exception as e:
            sys.exit(f'\nFAIL: The {PROJECT_FILENAME} file has one or more formatting errors that need to be resolved.\n      JSON Error: {e}\n')
    return dct

def dict_to_json_file(filepath: str, dct: dict):
    with open(filepath, "w") as f:
        json.dump(dct, f, indent=2)

def make_v2_filepath(v1_filepath):
    # make new directory
    v2_path = str(v1_filepath.parent) + '_v2'
    if Path(v2_path).resolve():
        os.makedirs(v2_path, exist_ok=True)
    else
        raise Exception('\n\nFAIL: Invalid input folder was specified')
    return Path(v2_path) / PROJECT_FILENAME

def convert(v1):
    result = copy.deepcopy(v1)

    def remove_field(field, values):
        values.pop(field, None)

    def mod_collection(coll):
        for index, item in enumerate(result[coll]):
            keys = item.keys()
            keys = [f for f in keys if f not in ("id", "active", )]
            result[coll][index]["values"] = copy.copy(result[coll][index])
            pointer = result[coll][index]["values"]
            [remove_field(f, pointer) for f in COLLECTION_FIELDS_TO_REMOVE]

            for key in keys:
                remove_field(key, result[coll][index])

            if coll == "technologySpecsSolarPV":
                pointer["tsSolarPVGenerationProfile"] = pointer["associatedInputs"][0]["ts"]["data"]
                remove_field("associatedInputs", pointer)

            if coll == "technologySpecsControllableLoad":
                pointer["tsControllableLoadProfile"] = pointer["associatedInputs"][0]["ts"]["data"]
                remove_field("associatedInputs", pointer)

            if coll == "technologySpecsFleetEV":
                pointer["tsFleetEVBaselineLoadProfile"] = pointer["associatedInputs"][0]["ts"]["data"]
                remove_field("associatedInputs", pointer)

            if coll == "technologySpecsBattery":
                pointer["cycleLifeCurve"] = pointer["associatedInputs"][0]["dataRows"]
                remove_field("associatedInputs", pointer)

            if coll == "technologySpecsICE" or coll == "technologySpecsDieselGen":
                remove_field("fuelCost", pointer)
                pointer["fuelType"] = "fuel_type must be selected";

    def mod_timeseries(field, dct):
        dct[field] = dct[field]["data"]

    # Collection modifications
    [mod_collection(coll) for coll in COLLECTIONS]

    # For each timeseries, use data as entire field (strip metadata)
    [mod_timeseries(ts, result) for ts in PROJECT_TIMESERIES]

    # Renamed properties
    result["mtsDrMonthsApplied"] = copy.deepcopy(result["drMonthsAppliedLabels"])
    del result["drMonthsAppliedLabels"]

    # Remove deleted project properties
    del result["listOfActiveTechnologies"]
    del result["listOfActiveServices"]
    del result["externalIncentivesFileImportNotes"]
    del result["retailTariffFileImportNotes"]

    # Add new fuel price properties
    result["fuelPriceGas"] = None
    result["fuelPriceLiquid"] = None
    result["fuelPriceOther"] = None

    # Add new Scenario field
    result["activate_electricity_load_dump"] = None

    # add a schema version
    result['schemaVersion'] = SCHEMA_VERSION

    return result

def main(v1_filepath):
    v1 = dict_from_json_file(v1_filepath)
    if v1.get('schemaVersion') == SCHEMA_VERSION:
        sys.exit('\nNOTHING WAS DONE: This project is already compatible with DER-VET GUI version 1.2\n')
    v2 = convert(v1)
    v2_filepath = make_v2_filepath(v1_filepath)
    dict_to_json_file(v2_filepath, v2)
    print(f'\nA v2 {PROJECT_FILENAME} was created:\n  {v2_filepath.resolve()}\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='migrate_project_dervet_GUI.py',
        description='This script converts an existing version 1.1 DER-VET GUI Project into a ' +
          f'new \'{PROJECT_FILENAME}\' file for import into version 1.2 of the DER-VET GUI. ' +
          f'A single argument (a directory/path which contains a {PROJECT_FILENAME} file) is ' +
          'required. This directory will not be altered. ' +
          'A new directory is created (with \'_v2\' appended to the name) which will ' +
          f'contain the new converted \'{PROJECT_FILENAME}\' file. ' +
          'Should be used with Python 3.2 or greater',
        epilog='Copyright 2023. Electric Power Research Institute (EPRI). All Rights Reserved.',
    )
    parser.add_argument('v1_directory_name', type=str, help='specify the directory name to work on')
    args = parser.parse_args()

    if (Path(args.v1_directory_name) / PROJECT_FILENAME).is_file():
        v1_filepath = Path(args.v1_directory_name) / PROJECT_FILENAME
    else:
        raise Exception(f'\n\nFAIL: You must specify a valid directory name in the argument to this script that has the "{PROJECT_FILENAME}" file.\n')
    main(v1_filepath)
