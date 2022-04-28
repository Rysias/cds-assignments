# Assignment 1: Most similar images
Here are solutions for finding most similar images as described in `assignment1.md`. All solutions are python scripts with cmd-arguments. Documentation for each script can be found using the `--help` flag. In addition to the scripts, there is a jupyter notebook for experimenting with the different solutions. Most similar images are defined using chi-squared from the `compareHist`-function in `cv2`. Main packages used are `numpy`, `pandas`, and `cv2`. To increase processing speed, multiprocessing is also used (per default all cores-1 are utilized)

## Software Design
The main challenge in this assignments is creating a design that avoids code duplication between the two different models. I found that the main commonalites between the use cases were a) loading the data and b) printing the report. This is why I moved the functionality for these two tasks into separate files (`src/load_data.py` and `src/report_performance.py`). The actual scripts can then be higher level and focus on the specifics of the two models. Translated into the SOLID-principles this means: 
- **Single responsibility**: Splitting functionality into separate files and functions. 
- **Interface segregation**: I make the main scripts independent of the implementation details in each other and in loading the data / writing the reports. 
- **Liskov substitution**: I make sure that each dataset / model uses structural typing which makes them more cleanly separated.
- **Open-closed**: I have tried making it easy to add more datasets / models by having structural expectations
- **Dependency Inversion**: This is also to use the power of typing to avoid low-level coupling

## Files
Below is a description of the different files.

### Helper files 
To cleanly separate functionality, I have created a `src` directory with the following files. This allows us to segregate the interfaces: 
- `calculate_dists.py`: functionality for doing all the heavy lifting and calculations
- `format_output.py`: functionality for creating nice and readable outputs
- `img_help.py`: functionality shared between the two main scripts.

### Scripts
1. `find_similar_imgs.py`: a script for completing the basic task as described in `assignment1.md`. Takes a filename (the source image) as argument and, optionally, a directory and number of cores to use. Outputs an image with the 3 most similar images with their distances as well as a csv with similarities. 
2. `dir_similarity.py`: Find closest 3 images for every image in the supplied directory.

### Other files
- `experiments.ipynb`: code sketches and experiments

## Usage 
TL;DR: An example of the entire setup and running pipeline can be run using the bash-script `run_project.sh`. 

### Setting up
The project uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html). Setup can be done as easily as `pipenv install` (after pipenv has been installed) and activating the environment is `pipenv shell`. NB: Make sure that you have python 3.9 (or later) installed on your system!

### Using the script(s)
The scripts are based around strong defaults. However, `find_similar_imgs.py` has a required argument of a path to an image (`--img-path`). Complete documentation can be found using `<name of script>.py --help`. For an example, see below:

```console
$ python find_similar_imgs.py --img-name testimg.jpg --ncores 1000
```


## TODO
- [] create reproducible pipenv 
- [] create `run_project.sh`
- [X] refactor to output / src 
- [] Update README with template from the other ones