# Changelog
Notable changes to DER-VET are documented in this CHANGELOG.md.

Questions and feedback can be submitted to the Electric Power Research Institute (EPRI) from the Survey Monkey feedback linked on the DER-VET website (https://www.der-vet.com/).

The format is based on [Keep a Changelog] (https://keepachangelog.com/en/1.0.0/).

## [1.3.0] - 2024-12-02
### Fixed
- removed the 'Overall Tax Burden' column from the proforma
  - it was causing a bug in the 'Yearly Net Value' tally
- fixed issue where the log files get jumbled
  - we now close the log file upon any raised Exception,
      or when the run completes without error
- fixed issue where RTE losses were being 2x counted in FR service
  - this was a Regulation Energy Throughput for low-RTE tech bug
- fixed infeasibility issues from positive non-zero min power levels
    for charging or discharging in ESSs
  - min power constraints are included in get-schedule ESS methods
- Spelling errors fixed in default Model Parameters CSV
- fixed fixed_om calc so that it gets multiplied by the rated capacity
  - this fixed for the following technologies:
    PV, ICE, CHP, Chiller, Boiler, Electrolyzer
    (it was already implemented in ES and CAES)
- Addresses many Python warnings
  - reduces the amount of User, Future, Deprecation, and other Warnings
    as a result of upgrading to using pandas 2.2
  - suppress the UserWarning about date format parsing here
    we want a flexible date parser, and do not need to warn
  - suppress the UserWarning about the problem not being DPP
    the basic DA service objective_function() is always not DPP
  - addresses some cvx UserWarnings
- Fixes bugs with DCM and tariffs
  - The methods used to create the billing periods was flawed
    because it used an object that was getting updated as it went.
    This was corrected to use a copy of the original tariff file,
    across several scripts.
  - The tariff growth rate in customer bill was being applied wrong.
    It had an extra divide by 100 so that the growth rate was much
    smaller than what is should be. This was fixed.
### Changed
- Upgraded to Python 3.11
- Moved away from having storagevet be a submodule of dervet
  - all code is now contained in a single repository
  - --recurse_submodules is no longer used
- All code updates are now seen in this CHANGELOG.md file
- removed the requirements-dev.txt file
- requirements.txt file includes a local install of storagevet
  - many packages are updated to latest stable versions
- Try cvx solve with another solver if the first one fails
  - Toggles between GLPK_MI and ECOS_BB
- changes to payback.csv
  - result format is transposed
  - columns are Metric, Units, Value
- changed the DA ETS so that it uses Net Load from the Results TS
- Reduces the decimal precision to 2 on dollar vlaues
  - For all results CSVs that show dollar amounts
     - cost_benefit
     - payback (just Lifetime Net Present Value)
     - npv
     - pro_forma
     - tax_breakdown (dervet only)
- IRR calc is replaced by MIRR calc (modified IRR)
  - MIRR is modified Internal Rate of Return and will appear
    in the payback.csv file with units of percent
- Adds fixed_om costs from PV and ElectrolyzerSystem to be
  included in the cost function
- Fixes the nomenclature regarding fixed_om_cost
  - fixed_om is now fixed_om_cost for an ElectrolyzerSystem
    this is to distinguish it from fixed_om (used for Loads)
- Cleans up the messaging wrt Battery degradation
  - incl_degradation replaces incl_cycle_degrade,
    but there is backward compatibility (using alternate keys)
  - makes incl_degradation the default in both Model Params files
  - cleans up the messaging for Battery degradation
  - fixes the use of optional/alternate keys in parameters
    - keys can now be optional
    - keys can have an alternate key
  - these parameters are now entirely optional:
    - Finance-external_incentives
    - Results-label
    - Finance-ecc_mode
- Removed duration_max from CAES
- Cleaned up DR tests and handling of length of events
  - DR-length and DR-program_end_hour handling were cleaned up
- Disallows Deferral Service without an active Battery
- Removes all mention of XML in the code
  - DER-VET used to convert input CSV model parameters into XML
    This behavior has been replaced by JSON
    XML files are no longer supported
    The read_and_validate() scripts in Params class have diverged
### Added
- Adds a requirements-conda.txt file
- New results file: capacity_factor.csv
  - shows the fractional capacity factor for each active technology
    except for EV1, EV2 and ControllableLoad
- Adds input error when user selects both DA and retailTimeShift.
  You cannot have energy prices from 2 sources in DER-VET.
- Adds DEBUG boolean for printing constraints/costs in:
  - dervet/MicrogridScenario.py
  - dervet/MidcrogridPOI.py
- Adds input error when DR or RA is active with technology being sized
- Adds checks for required columns in battery_cycle_life.csv data file
- Adds a new ElectrolyzerSystem technology
  - adds 5 variables to the output time series
    - Power (kW)
    - Original Load (kW)
    - On (y/n)
    - Efficiency (kg/kWh)
    - Hydrogen Production (kg)
  - adds Hydrogen value ($) to financial handling and results files
    - hydrogen value = power * efficiency[input] * unit_value[input]
  - all required inputs added to the default Model Parameters CSV and JSON
  - adds cost (objective) function based on hydrogen value
  - substantial pytests added to test capabilities
  - adds hydrogen production quotas (meet or exceed)
    - Adds the following two parameters to the Electrolyzer technology
      - hydrogen_production_quota: the quantity of hydrogen (kg) to
        meet or exceed in each optimization window
      - hydrogen_schedule_mode: Defines how to control hydrogen production:
        0=no control,
        1=apply a fixed schedule according to an input time series of
          'ElectrolyzerSystem Schedule (kW)',
        2=use hydrogen_production_quota parameter to set the quantity of
          hydrogen (kg) that must be produced in each optimization window
  - adds LCOH: Levelized Cost of Hydrogen to payback.csv
    - reports out when Electrolyzers are operating
    - LCOH = NPV(cash flows exclusive of hydrogen value)/NPV(hydrogen produced)
  - Adds electrolyzer variable efficiency post-optimiztion
    - Creates 2 new ElectrolyzerSystem model parameters:
      - incl_variable_efficiency
      - variable_efficiency_filename
    - When the boolean incl_variable_efficiency is True,
      variable efficiency replaces the constant scalar efficiency,
      and 2 new columns will appear in the results time series:
      - Efficiency Actual (kg/kWh)
      - Hydrogen Production Actual (kg)
    - The new values of Hydrogen Production Actual are then used
      in performance and financial results.
    - The reason for this treatment is that with variable efficiency ON,
      using it will create a nonlinear constraint and will break
      our optimization approach. Instead, we handle it post-optimization.
    - A warning appears when variable_efficiency is ON and
      hydrogen production quotas as being set in constraints
  - Adds a check on scheduled electrolyzer power
    - When not sizing, and hydrogen_schedule_mode is 1,
      values of scheduled power cannot exceed rated_power,
      and we cause an error in this case.
  - Adds a min_power constraint
  - disallow ElectrolyzerSystems to participate in market services
  - Adds a grow_drop_data() method
    - When hydrogen_schedule_mode is 1, we upload a hydrogen
      schedule time series. these values may change by the default
      growth parameter (in Scenario) when optimizing for multiple years
  - Adds ElectrolyzerSystem Variable O&M Costs, with units of $/kg
    - special handling was needed for when efficiency is variable
  - Adds degradation modeling for ElectrolyzerSystems
    This builds off of what we already have for Battery degradation.
    However, for ElectrolyzerSystems, it is simpler.
    - Three new input parameters:
      incl_degradation
      cycle_life_filename
      yearly_degrade
    - New cycle-life file in data:
      - electrolyzersystem_degradation_curve.csv
    - We do not consider user inputs for state_of_health or
      cycle_life_table_eol_condition, but instead treat these as zero.
      The end-of-life is when the SOH goes to zero.
    - We use power fluctuation (as a percentage of rated_power), and not energy.
    - We disallow sizing and degradation, but this can go away with future work.
    - Adds checks for required header rows for these 3 data files:
      - electrolyzersystem_efficiency_curve.csv
      - electrolyzersystem_degradation_curve.csv
  - Adds continuous ElectrolyzerSystem sizing
    - Future work is required to achieve Discrete Sizing

## [1.2.3] - 2023-01-26
### Fixed
- Simplify the README
- improvements to the Reliability Sizing module
  - define a minimum 72-hour continuous window for infeasibility checking to capture day and night
  - fix the size of non-ESS DERs first and then iterate on ESS sizing
  - adds a new sizing method that will unset the size for iterating through the reliability sizing
  - allow reliability sizing for sub-hourly timesteps
- better error messaging when an infeasibility is encountered (avoid infinite loop)
- when a size optimization warning occurs due to the horizon-mode being 2 or 3,
    also have the code fail with a useful error message (this part was previously ignored)
- all output values labeled in percent should range from 0 to 100
- remove triple equals sign from line in the requirements file
- allow om_cost to end up in the proforma output CSV file for technologies
- allow decommissioning-costs to be negative for all technologies
- bypass check on valid project start and end year when analysis-horizon-mode is not set to 1
- fix the Controllable Load technology class so that is will recognize its input parameters
### Added
- pytests for GUI pre-defined use cases
  - test number of results files, proforma values, npv values, and load-coverage-probability values
- pytests that mimic new storagevet tests, for testing with run_dervet
- add dervet sizing information to log file
- make the testing library methods more robust, easier to use, and aligned more with storagevet tests
- include a second call to calculate_system_requirements for dervet, after reliability sizing occurs
- have dervet-only technologies (EVs, thermal, ControllableLoad) add to der-dispatch-net-power
- add 4 standard input parameters to Controllable Load technology
### Changed
- when optimally sizing, include Warning if there are negative DA energy prices input
- clean up the Warnings and Error reporting on the binary parameter and a DCP error
- change the default ts_constraints parameters to 0 for all services

## [1.2.2] - 2022-05-05 to 2022-07-07
### Added
- pytests for small tweaks to the default model parameters CSV
  - create instances of infeasibility with bad data in the input time series
  - setting ts-limit booleans ON should not fail
### Changed
- improved handling of input time series data
  - new method: get_single_series()
  - introduce more specific error classes: TimeseriesDataError and TimeseriesMissingError
  - better error and warning messaging and readability
- sets all min and max battery constraint values to empty in the default input time series
- adds column: LF Price ($/kW) to default time series
- improvements to the migration script
  - when the input is already v1.2, exit with a clear message
  - have better error messages for poorly formatted json inputs
### Fixed
- the broken reliability service, when post_facto_only is OFF, has been fixed
  - Adds back the SysEneMinReq constraint
  - NOTE: this was mistakenly removed in a commit from August 2021
  - adds a pytest to ensure that when an optimization loop is run, with Reliability active,
    the first X hours in the load_coverage_prob.csv file are 1 meaning 100 percent covered

## [1.2.1] - 2022-03-31 to 2022-05-04
### Added
- adds warning message to ignore the duration_max parameter when not sizing a battery

## [1.2.0] - 2021-09-10 to 2022-03-30
### Added
- added a migrations/migrate_project_DERVET_GUI.py script
  - this will transform a project exported from the GUI v1.1.2 into GUI v1.2.0
- pytests to ensure that the default model parameter CSV file runs when input into run_DERVET.py
- pytests to ensure that with each technology active along with a battery, the default model parameter CSV runs
- adds required rows for all technologies in the default model parameter csv
- adds a new scenario tag input to allow/disallow an electric load_dump
- a copy of the model parameters input CSV is now copied to the Results folder for each run
- adds thermal technologies: CHP, Boiler, and Chiller
  - a Chiller can serve a cooling load only, and can be powered by electricity, natural gas, or heat
  - a Boiler can serve a heating load (hot water and/or steam), and can be powered by electricity or natural gas
  - a CHP can serve a heating load (hot water and/or steam), and an electrical load
  - an active thermal technology requires the appropriate thermal input time series data

### Changed
- upgrade supported/recommended Python version to 3.8.13
  - Python package requirements have been updated
  - Update installation instructions: Python environment creation, conda-route, pip-route
- re-structures how fuel costs are handled (see storagevet CHANGELOG)
- force use of the GLPK_MI solver when a project has an active thermal technology
- limit MACRS term to no greater than 20

### Removed
- remove incl_thermal_load boolean from model parameter inputs

### Fixed
- disallow sizing of CAES since it has not been validated

## [1.1.2] - 2021-08-04 to 2021-09-09
### Changed
- Changed the expected type to float for yearly_degrade battery input

### Fixed
- Degradation Fix: more descriptive column header names on Results files
- Simplifies system_requirements infeasibility checks
- Fix to allow minimum battery sizing user constraints to work

## [1.1.1] - 2021-07-09 to 2021-08-03
### Fixed
- Removed comma from soc_target description in the Model Parameters CSV

## [1.1.0] - 2021-04-14 to 2021-07-09
### Added
- this CHANGELOG.md file
- useful error messaging and warning for extreme soc_target values with reliability
- all growth rates have a minimum value of -100 percent
- Fleet EV will output the Baseline Load time series

### Changed
- description of battery soc_target updated for reliability based ES sizing
- modified the README.md with better and more thorough instructions for Installation
- increased the max limit (hours) on optimization window to be 8784

### Fixed
- corrected the logic and docstrings in ParamsDER class bad_active_combo method
- load_technology bug regarding names_list was fixed
