# Simple Network Analysis
This assignment solves the problem of creating rudimentary network analysis based on edgelists saved in tsv-format. The project is structured as a mini "package" with helpler functions in `src/`, tests in `tests/`, and the main script in the main folder (`network_analysis.py`). 

## Software Design
The goal of this package is to create a python script to do network analysis and -visualisation for arbitrary data. The general design tries to follow the SOLID principles in the ways: 
- **Single responsibility**: Each method only does one thing i.e. graphing is separated from analysis which is separated from writing
- **Interface segregation**: I try to do as much calculation on the graph objects to increase the flexibility
- **Liskov substitution**: Not applicable as the implementation is more functional than object-oriented. 
- **Open-closed**: The implementation makes it easy to add/remove network metrics without having to change the course - only a single dictionary needs to be changed
- **Dependency Inversion**: Typing is used to maximize interoperability between different functions. 
Furthermore, I have tried to make it clear whether the input is correctly formatted through doc-strings and failing early. 

## Usage 
TL;DR: An example of the entire setup, testing, running pipeline can be run using the bash-script `run_project.sh`. 
### Setting up
The project uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html). Setup can be done as easily as `pipenv install` (after pipenv has been installed) and activating the environment is `pipenv shell`. NB: Make sure that you have python 3.9 (or later) installed on your system!

### Using the script
The script has a single argument `--data-path` that can be either a) a .csv-file (tab-delimited) or b) a folder containing .csv-files (tab delimited). An example can be seen below:

```console
$ python network_analysis.py --data-dir ../input/edgelist.csv
```

## Other files
Other files not previously described include
- `experiments.ipynb`: for interactively testing the functionality and designing (not up to date)

## Testing
The scripts were developed using a TDD-methodology using [pytest](https://docs.pytest.org/en/7.0.x/). To execute the test suite run `python -m pytest` from the main directory.


# TODO
- [ ] Create sizes in visualization
- [ ] Refactor helper functions
- [ ] Separate calculation from dataframe creation (see notebook)
