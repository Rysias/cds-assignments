# Assignment 3 - Transfer learning + CNN classification
[GITHUB LINK](https://github.com/Rysias/cds-assignments/tree/main/language-assignments/language-a2)

# TODO: Abstract

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
I have chosen subtask 2 for this assignment: 

2. Using the corpus of Fake vs Real news, write some code which does the following
   - Split the data into two datasets - one of Fake news and one of Real news
   - For every headline
     - Get the sentiment scores
     - Find all mentions of geopolitical entites
     - Save a CSV which shows the text ID, the sentiment scores, and column showing all GPEs in that text
   - Find the 20 most common geopolitical entities mentioned across each dataset - plot the results as a bar charts

**Bonus tasks completed**
- Repeat experiments using both sentiment analysis techniques, in order to compare results.

### Personal Learning Goals
# TODO: Create this!

## Methods and Design
The overall approach to this task is quite similar to [assignment 2](../vision-a2/) with a few different notes. the main difference is that we use a much more powerful system in relying on vgg16 as our base model.

Other than that the flow goes as follows (as shown in [transfer_cnn.py](./transfer_cnn.py)): 
1. specify the hyperparameters using the command line
2. load the data 
3. initialize the model
4. train the model
5. evaluate (both in terms of the training plot and the classification report)

### Software Design (TODO THIS)
- **Single responsibility**: 
- **Open-closed**: 
- **Liskov substitution**: 
- **Interface segregation**: 
- **Dependency Inversion**: 

## Usage 
TL;DR: An example of the entire setup and running the pipeline can be run using the bash-script `run_project.sh`. 

### Setting up
The project uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html). Setup can be done as easily as `pipenv install` (after pipenv has been installed) and activating the environment is `pipenv shell`. NB: Make sure that you have python 3.9 (or later) installed on your system!

### Using the script (TODO THIS)

Parameter | Type | Required | Description
---- | ---- | ---- | ----
`--learning-rate` | `float` | No | The learning rate (defaults to 0.001)
`--learning-rate` | `int` | No | Batch size for each iteration of SGD (defaults to 32)
`--epochs` | `int` | No | How many epochs to train for (defaults to 10).

#### Example usage
```console
$ python TODO
```
## Discussion and Results
### Results
![Fake entities](./output/Fake_top_ents.png)

*Figure 1: Fake entities*
#### Real entities
![Real entities](./output/Real_top_ents.png)
*Figure 2: Real entities*

### Discussion (TODO)
- Discuss the entities
- Create the plot of the sentiment 
- Discuss the pros and cons of the models

## TODO
- [ ] Split textblob and spacy into separate files