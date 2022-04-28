# Assignment 2: Simple Image Classifications
This assignment solves the problem of creating rudimentary network analysis based on edgelists saved in tsv-format. The project is structured as a mini "package" with helpler functions in `src/`, and the main scripts in the main folder (`nn_classifier.py` and `logistic_regression.py`). 

## Software Design
The main challenge in this assignments is creating a design that avoids code duplication between the two different models. I found that the main commonalites between the use cases were a) loading the data and b) printing the report. This is why I moved the functionality for these two tasks into separate files (`src/load_data.py` and `src/report_performance.py`). The actual scripts can then be higher level and focus on the specifics of the two models. Translated into the SOLID-principles this means: 
- **Single responsibility**: Splitting functionality into separate files and functions. 
- **Interface segregation**: I make the main scripts independent of the implementation details in each other and in loading the data / writing the reports. 
- **Liskov substitution**: I make sure that each dataset / model uses structural typing which makes them more cleanly separated.
- **Open-closed**: I have tried making it easy to add more datasets / models by having structural expectations
- **Dependency Inversion**: This is also to use the power of typing to avoid low-level coupling

## Usage 
TL;DR: An example of the entire setup, testing, running pipeline can be run using the bash-script `run_project.sh`. 

### Setting up
The project uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html). Setup can be done as easily as `pipenv install` (after pipenv has been installed) and activating the environment is `pipenv shell`. NB: Make sure that you have python 3.9 (or later) installed on your system!

### Using the script
The script has a single argument `--data-path` that can be either a) a .csv-file (tab-delimited) or b) a folder containing .csv-files (tab delimited). An example can be seen below:

```console
$ python logistic_regression.py --dataset cifar10
```

