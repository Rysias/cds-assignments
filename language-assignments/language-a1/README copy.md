# Assignment 3 - Transfer learning + CNN classification
[GITHUB LINK](https://github.com/Rysias/cds-assignments/tree/main/language-assignments/language-a1)


#TODO: Create abstract 


## Table of Content
- [Assignment Description](#assignment-description)
    * [Personal learning goals](#personal-learning-goals)
- [Methods and design](#methods-and-design)
    * [Software design](#software-design)
- [Usage](#usage)
    * [Setting up](#setting-up)
    * [Using the script(s)](#using-the-scripts)
- [Results and Discussion](#results-and-discussion)
    * [Results](#results)
    * [Discussion](#discussion)

## Assignment Description
For this assignment, you will write a small Python program to perform collocational analysis using the string processing and NLP tools you've already encountered. Your script should do the following:

- Take a user-defined search term and a user-defined window size.
- Take one specific text which the user can define.
- Find all the context words which appear ± the window size from the search term in that text.
- Calculate the mutual information score for each context word.
- Save the results as a CSV file with (at least) the following columns: the collocate term; how often it appears as a collocate; how often it appears in the text; the mutual information score.

**Additional bonus tasks completed**
- Create a program which does the above for every novel in the corpus, saving one output CSV per novel
- Create a program which does this for the whole dataset, creating a CSV with one set of results, showing the mutual information scores for collocates across the whole set of texts
- Create a program which allows a user to define a number of different collocates at the same time, rather than only one

### Personal Learning Goals
Apart from challenging myself with the bonus tasks, I want to see if I can make the assignment follow the  [SOLID principles](https://www.digitalocean.com/community/conceptual_articles/s-o-l-i-d-the-first-five-principles-of-object-oriented-design) to the extent that it makes sense. There is a lot of room for this, as many of the tasks follow the same basic structure. I will explain more about this in the [software design section](#software-design). 

## Methods and Design
- Uses Spacy for tokenisation
    - lower case and punctuation removed
- other than that a vectorized approach using numpy and pandas to calculate Mutual information 
- Link to MI calculation + short explainer

# TODO WRITE THIS OUT

### Software Design
- Share core functionality through src/collocate.py 
    - Might have been nicer to split more out
- Still uses a functional paradigm (lower complexity)
- Focus on single responsibility and interface segregation

- **Single responsibility**: Each function does one thing and one thing only, which makes them a) easier to debug and b) easier to refactor.
- **Open-closed**: Not too applicable as we have little functionality to add.
- **Liskov substitution**: Not applicable as we don't work with classes. 
- **Interface segregation**: Each file (e.g. [`collocate_corpus.py`](./collocate_corpus.py)) has more or less only base python dependencies. This makes it easier to change implementation details.
- **Dependency Inversion**: Not super relevant for this project, as the assignment is relatively set so we don't need to juggle different back-ends.

## Usage 
TL;DR: An example of the entire setup and running the pipeline can be run using the bash-script `run_project.sh`. 

### Setting up
The project uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html). Setup can be done as easily as `pipenv install` (after pipenv has been installed) and activating the environment is `pipenv shell`. NB: Make sure that you have python 3.9 (or later) installed on your system!

### Using the script (TODO: THIS!)


Parameter | Type | Required | Description
---- | ---- | ---- | ----
`--learning-rate` | `float` | No | The learning rate (defaults to 0.001)
`--learning-rate` | `int` | No | Batch size for each iteration of SGD (defaults to 32)
`--epochs` | `int` | No | How many epochs to train for (defaults to 10).

#### Example usage (TODO: THIS!)
```console
$ python TODO
```

## Discussion
TODO: THIS