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
There are two learning goals in particular that have shaped my development of this assignment. One is to gain experience working with large language models, which I (and other smarter people (LINK)) believe might be the future of Artificial (General) Intelligence. The second is to create a workflow and structure that supports rapid iterations and reproducibility. This is particularly important when exploring new technologies but iteration speed and reproducibility are key skills across all of data science (and science!). 

## Methods and Design
### Data Collection
# TODO: IMPLEMENT KAGGLE SCRIPT
The dataset I use is the [ISOT Fake News Dataset](https://www.uvic.ca/ecs/ece/isot/assets/docs/ISOT_Fake_News_Dataset_ReadMe.pdf). It is a large collection of "real" news scraped from Reuters.com and fake news scraped from a wide variety of internet sources. Furthermore, the data is easily accessible via kaggle.com and the script [`download_data.sh`]()

### Cleaning News
While the ISOT dataset is in general relatively clean, it is nevertheless an NLP dataset which necessitates more cleaning. The cleaning pipeline consists of both general and domain specific steps. 

The general cleaning steps are quite standard and minimal: basically it consists of normalising whitespace. As the end-consumer are human readers most of the standard preprocessing pipeline such as removing punctuation, lower-casing and lemmatisation have been left out. 

On the domain-specific side, there are a few steps that needed to be taken to ensure good prompt generation and fair comparison: 
- **Removing 'tags'**: The headlines and texts would have different markers that would bias the prompt generation and make them easier to distinguish. This could be words like "VIDEO" in the title, or the fact that all Reuters articles start with the location of the story (e.g. "WASHINGTON (REUTUERS)...")
- **Removing URLs**: While GPT models can generate URLs these are most often invalid. Therefore, I have removed these from the texts as a) it makes the comparisons more fair and b) actual fake news generation would probably do the same. 
- **Limiting length of articles**: I have limited the text of the ground-truth and generated articles to a maximum of two sentences (or <75 tokens). I will discuss the reasons for this in other sections. 


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