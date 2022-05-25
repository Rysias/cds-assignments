# Assignment 3 - Transfer learning + CNN classification
[GITHUB LINK](https://github.com/Rysias/cds-assignments/tree/main/language-assignments/fooled-by-llm)

# TODO 

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
For this assignment the goal is to write software for generating the beginning of news articles based on headlines using GPT-models. The software should do the following: 

1. Download (and clean) a suitable dataset
    - This script can then be used as a ground-truth dataset for the turing
2. Create a script for generating news articles  
    - The script should take as input a) the source dataset and b) the model to use
3. Format the data to be ready for a News Turing test

### Personal Learning Goals
- Prompt engineering 
- Working with large language models

## Methods and Design
### Data Collection
- https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php
- Downloaded via kaggle

### Cleaning News
- Length is important (as you pay per token + off)
- Removing URLs (from both --> tendency to generate false ones)

### Generating news / prompt engineering 
- 6b.eleuther.ai <-- used for testing prompts 
- About Goose.ai 
    - API based on EleutherAI
    - Works with OpenAI's API (with different endpoint)
- More on that in discussion
- Notes on manual selection

### Software Design
# TODO 
- **Single responsibility**: 
- **Open-closed**: 
- **Liskov substitution**: 
- **Interface segregation**: 
- **Dependency Inversion**: 

## Usage 
TL;DR: An example of the entire setup and running the pipeline can be run using the bash-script `run_project.sh`. 
# TODO: ADD Description of how to get API key.
### Setting up
The project uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html). Setup can be done as easily as `pipenv install` (after pipenv has been installed) and activating the environment is `pipenv shell`. NB: Make sure that you have python 3.9 (or later) installed on your system!

### Using the script
# TODO 

Parameter | Type | Required | Description
---- | ---- | ---- | ----


#### Example usage
```console
$ python 
```
## Discussion and Results
- Examples (in a table; semi manual)
    - Comments on the examples
- Notes on the process
    - Fake URLs 
    - Tendency to repeat itself
- The role of manual selectoin

# TODO List
- [ ] Fill out TODO's in this readme 
- [ ] Add download data (kaggle) --> data/raw
- [ ] Find way to visualize.