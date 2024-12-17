# StoragetVET 2.0

StorageVET 2.0 is a valuation model for analysis of energy storage technologies and some other energy resources paired with storage. The tool can be used as a standalone model, or integrated with other power system models, thanks to its open-source Python framework. Download the executable environment and learn more at https://www.der-vet.com.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites & Installing

* Please refer to the latest [DER-VET README.md file](https://github.com/epri-dev/DER-VET/blob/master/README.md) for installing the proper environment.

>If you wish to only install StorageVET, then only clone the StorageVET repository, and have the `storagevet` folder as your 'root directory'. All other steps are the same.

## Running Your First Case
Follow these steps to run StorageVET from the command-line:

1. ####  Open your favorite command-line prompt and activate Python environment

2. ####  Navigate to the root "storagevet" folder

3. ####  Enter the following into your open prompt:

    ```
    python run_StorageVET.py Model_Parameters_Template.csv
    ```

## Running the tests

To run tests, activate Python environment, and navigate to the root "storagevet" folder. Then enter the following into your open prompt:

```
python -m pytest test
```

## Deployment

To use this project as a dependency in your own, clone this repo directly into the root of your project.
Open a command-line prompt from your project root, and input the following command:
```
pip install -e ./storagevet
```

## Versioning

For the versions available, please
see the [list of releases](https://github.com/epri-dev/StorageVET/releases) on out GitHub repository.
This is version 1.2.3

## Authors

* **Miles Evans**
* **Andres Cortes**
* **Halley Nathwani**
* **Ramakrishnan Ravikumar**
* **Evan Giarta**
* **Thien Nguyen**
* **Micah Botkin-Levy**
* **Yekta Yazar**
* **Kunle Awojinrin**
* **Giovanni Damato**
* **Andrew Etringer**

## License

This project is licensed under the BSD (3-clause) License - see the [LICENSE.txt](./LICENSE.txt) file for details

