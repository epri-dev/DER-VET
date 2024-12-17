# DER-VET™

[DER-VET™](https://der-vet.com) provides a free, publicly accessible, open-source platform for calculating, understanding, and optimizing the value
of distributed
energy resources (DER) based on their technical merits and constraints. An extension of EPRI's [StorageVET®](./storagevet) tool, DER-VET supports
site-specific assessments of energy storage and additional DER technologies—including solar, wind, demand response, electric vehicle charging,
internal combustion engines, and combined heat and power—in different configurations, such as microgrids. It uses load and other data to determine
optimal size, duration, and other characteristics for maximizing benefits based on site conditions and the value that can be extracted from targeted
use cases. Customers, developers, utilities, and regulators across the industry can apply this tool to inform project-level decisions based on sound
technical understanding and unbiased cost-performance data.

DER-VET was developed with funding from the California Energy Commission. EPRI plans to support continuing updates and enhancements.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for
notes on how to deploy the project on a live system.

### New Installation

Follow these steps to run DER-VET through your command-line on your local computer.

1. #### Open your favorite command-line prompt, in administrator mode
   >You will need administrator access on your computer.

    **On Windows**

    Choose from git bash, powershell, command prompt, or anaconda prompt.

    **On Mac**

    Use terminal, iterm2, or anaconda prompt.

2. #### Clone/download this repository (repo) onto your local computer
    Navigate to a folder on your computer where you place git repositories. Normally this is a folder named `git` that
    lives in your home directory, but it can be any folder.

    Enter the following command from a terminal/prompt to 'clone' the repo and download it to your computer.

    ```
    git clone https://github.com/epri-dev/DER-VET.git
    ```

    When cloning a repo with 'git clone', if you do not specify a new directory as the last argument (as shown above),
    it will be named `DER-VET`. Alternatively, you can specify this and name it as you please. Regardless of what the
    downloaded folder is named or located, this new directory becomes the 'root directory' of dervet.

    >The *root directory* refers to the folder with folders/files such as
   > `dervet`, `data`, `test`, `README.md` (this file), `Model_Parameters_Template_DER.csv`, and `run_DERVET.py`.

3. #### Install system requirements
    **On Windows**

    Install the Build Tools for [Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/). When prompted by the
    installer, select C++ build tools and the appropriate Windows SDK specified in the table below and install.

    | Windows OS   | SDK        |
    |--------------|------------|
    | Windows 7    | Windows 8.1|
    | Windows 8.1  | Windows 8.1|
    | Windows 10   | Windows 10 |

   **On Mac**

    Install [Xcode](https://developer.apple.com/xcode/) and [GLPK](https://formulae.brew.sh/formula/glpk)

4. #### Navigate to your `dervet` folder
   This is the location of the repository or downloaded folder on your local computer. We refer to this location as the
    "root directory"

5. #### Create the Python virtual environment
    It is recommended that the latest Python 3.11 version be used. As of this writing, that version is Python 3.11.9

    We give the user two routes to create a Python environment for Python 3.11.9

    Each route results in a siloed Python environment, but with different properties.
    Choose the conda OR pip route and stick to it. Commands are not interchangeable.
    >Please remember the route which created the python environment in order to activate it again later.

   > **You will need to activate the python environment to run the model, always.**

    **Conda Route**

    This route requires you to install [Anaconda](https://www.anaconda.com/products/individual). After doing so, the anaconda prompt may be used. Note that Conda is both a package manager, and an environment manager.

    Enter the following command:

    ```
    conda create -n dervet-venv python=3.11.9
    ```

    **Pip Route**

	Install Python version 3.11.9 directly on your computer. Note that Pip is the package management system for Python, while Virtualenv is the environment manager.

    Enter the following commands:

    ```
    pip install virtualenv
    virtualenv dervet-venv
    ```
    >The `pip` command used here should be associated with **Python 3.11.9**

6. #### Activate Python 3.11.9 environment
    **Conda Route**

    Enter the following command into the open prompt:

    ```
    conda activate dervet-venv
    ```

    **Pip Route**

    Enter the corresponding command into the open prompt:

    *On Mac*

    ```
    source dervet-venv/bin/activate
    ```

    *On Windows*

    ```
    "./dervet-venv/Scripts/activate"
    ```

7. #### Install project dependencies
    **Conda Route**

    Enter the following commands into the open prompt:

    ```
    conda install --channel conda-forge --file requirements-conda.txt
    pip install -e ./storagevet
    ```

    **Pip Route**

    Enter the following commands into the open prompt:

    ```
    pip install -r requirements.txt
    ```

## Running Your First Case
Follow these steps to run DER-VET from the command-line:

1. ####  Open your favorite command-line prompt and activate Python environment
    Skip this step if your python environment is already active.

   >Refer to *New Installation* steps above for activation instructions.

2. ####  Navigate to the root "dervet" folder

3. ####  Enter the following into the open prompt:

    ```
    python run_DERVET.py Model_Parameters_Template_DER.csv
    ```

## Running the tests

1. #### Activate Python environment
    Skip this step if your python environment is already active.

   >Refer to *New Installation* steps above for activation instructions.

2. #### Enter the following into the open prompt from inside the root "dervet" folder:
    ```
    python -m pytest test
    ```

## Deployment

To use this project as a dependency in your own, clone this repo directly into the root of your project.
Open a command-line prompt from your project root, and input the following command:

```
pip install -e ./dervet
```

## Notes on updating your local repository

After each `git update` or `git pull` from remote, you will need to re-install the storagevet module locally to realize the code changes:
```
pip install -e ./storagevet
```

## Migrating DER-VET GUI project.json files

We have created a new folder titled "migrations" that is in the root "dervet" folder.
In this folder, we have provided a command-line Python script which will convert an existing version 1.1 DER-VET GUI project.json file into a version 1.2 DER-VET GUI project.json file. This script should be used with Python 3.2 or greater.

To view the usage statement for this script, open a command-line prompt from your project root, and input the following command:

```
python migrations/migrate_project_dervet_GUI.py -h
```

The script accepts a single positional argument: the name of a folder which must contain a "project.json" file

## Versioning

For the versions available, please
see the [list of releases](https://github.com/epri-dev/DER-VET/releases) on out GitHub repository.
This is version 1.2.3

## Authors

* **Halley Nathwani**
* **Miles Evans**
* **Suma Jothibasu**
* **Ramakrishnan Ravikumar**
* **Andrew Etringer**
* **Andres Cortes**
* **Evan Giarta**
* **Thien Nguyen**
* **Micah Botkin-Levy**
* **Yekta Yazar**
* **Kunle Awojinrin**
* **Arindam Maitra**
* **Giovanni Damato**

## Contributing
Pull requests are welcome. For major changes, please contact the team to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

This project is licensed under the 3-clause BSD License - see [LICENSE.txt](./LICENSE.txt).

DER-VET v1.3.0

Copyright © 2024 Electric Power Research Institute, Inc. All Rights Reserved.

Permission to use, copy, modify, and distribute this software for any purpose
with or without fee is hereby granted, provided that the above copyright
notice and this permission notice appear in all copies.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL EPRI BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Third-Party Software
EPRI does not own any portion of the software that is attributed
below.

<CVXPY/1.4.2> - &lt;Steven Diamond&gt;, <diamond@cs.stanford.edu>
Copyright © 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

CVXPY is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the
implied warranties of merchantability and fitness for a particular
purpose are disclaimed.

This software relies on CVXPY to interface with work(s) covered by the
following copyright and permission notice(s): 

GLPK 5.0 - Andrew Makhorin, mao@gnu.org
Copyright © 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
2010, 2011, 2012 Andrew Makhorin, Department for Applied Informatics,
Moscow Aviation Institute, Moscow, Russia. All rights reserved.

Licensed under GNU Public License v3.0; you may not use GLPK except in
compliance with the License. You may obtain a copy of the License at
https://www.gnu.org/licenses/gpl-3.0.en.html.

GLPK is a free program and is provided by the copyright holders and
contributors "as is" and any express or implied warranties, including,
but not limited to, the implied warranties of merchantability and fitness
for a particular purpose are disclaimed.
