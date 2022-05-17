# Assignment 2: Simple Image Classifications
[GITHUB LINK](https://github.com/Rysias/cds-assignments/tree/main/vision-assignments/vision-a1)

While [finding similar images is useful](../vision-a1/README.md#assignment-1-most-similar-images), we also want to know what they depict. This can be either because you want to build a self-driving car, or simply want to find out whether something is a [hotdog or not](https://www.youtube.com/watch?v=vIci3C4JkL0&ab_channel=vietanhle). 

In ye olden days of pre-2012, this task was approached through classical machine learning. Here domain-specific feature extraction was combined with algorithms such as logistic regression or SVMs to produce a (hopefully) correct prediction. 

However, this all changed with the introduction of [AlexNet in 2012](https://paperswithcode.com/paper/imagenet-classification-with-deep). Here Hinton and his gang showed that deep neural networks accelerated by GPUs could perform remarkably well just by throwing huge amounts of compute and data at the problem. 

In this assignment we will pit the two methods up against each other. However, for the real powers of deep learning we will have to wait until the [next assignment](../vision-a3/)

## Table of Content
- [Assignment Description](#assignment-description)
    * [Personal learning goals](#personal-learning-goals)
- [Methods and design](#methods-and-design)
    * [Software design](#software-design)
- [Usage](#usage)
    * [Setting up](#setting-up)
    * [Using the script(s)](#using-the-scripts)
- [Results and Discussion](#results-and-discussion)
    * [Classification reports](#classification-reports)
    * [Discussion](#discussion)

## Assignment Description
For this assignment, you will take the classifier pipelines we covered in lecture 7 and turn them into *two separate ```.py``` scripts*. Your code should do the following:

- One script should be called ```logistic_regression.py``` and should do the following:
  - Load either the **MNIST_784** data or the **CIFAR_10** data
  - Train a Logistic Regression model using ```scikit-learn```
  - Print the classification report to the terminal **and** save the classification report to ```out/lr_report.txt```
- Another scripts should be called ```nn_classifier.py``` and should do the following:
  - Load either the **MNIST_784** data or the **CIFAR_10** data
  - Train a Neural Network model using the premade module in ```neuralnetwork.py```
  - Print output to the terminal during training showing epochs and loss
  - Print the classification report to the terminal **and** save the classification report to ```out/nn_report.txt```

### Personal learning goals
For this assignment I have two goals: 1) explore the powers of [Github Copilot](https://copilot.github.com/) which I believe will fundamentally change the way we do programming (see my arguments in [this post](https://medium.com/codex/github-copilot-is-a-game-changer-cd0a2bbe6de8)), 2) further hone my software design skills in applying SOLID principles. 

## Methods and Design

### Software Design
The main challenge in this assignments is creating a design that avoids code duplication between the two different models. I found that the main commonalites between the use cases were a) loading the data and b) printing the report. This is why I moved the functionality for these two tasks into separate files (`src/load_data.py` and `src/report_performance.py`). The actual scripts can then be higher level and focus on the specifics of the two models. Translated into the SOLID-principles this means: 
- **Single responsibility**: Splitting functionality into separate files and functions. 
- **Interface segregation**: I make the main scripts independent of the implementation details in each other and in loading the data / writing the reports. 
- **Liskov substitution**: I make sure that each dataset / model uses structural typing which makes them more cleanly separated.
- **Open-closed**: I have tried making it easy to add more datasets / models by having structural expectations
- **Dependency Inversion**: This is also to use the power of typing to avoid low-level coupling

## Usage 
TL;DR: An example of the entire setup, testing, running pipeline can be run using the bash-script [`run_project.sh`](./run_project.sh).

### Setting up
The project uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html). Setup can be done as easily as `pipenv install` (after pipenv has been installed) and activating the environment is `pipenv shell`. NB: Make sure that you have python 3.9 (or later) installed on your system!

### Using the script(s)
Below is a description of the usage parameters for the two main scripts [`nn_classifier.py`](./nn_classifier.py) and [`logistic_regression.py`](./logistic_regression.py). The scripts are documented using argparse so a full description can be found using `python <name of script>.py --help`. 

#### logistic_regression.py
Parameter | Type | Required | Description
---- | ---- | ---- | ----
`--dataset` | `str` |  | The dataset to train on. Choose between `mnist_784` or `cifar10`.

#### nn_classifier.py
Parameter | Type | Required | Description
---- | ---- | ---- | ----
`--dataset` | `str` | Yes | The dataset to train on. Choose between `mnist_784` or `cifar10`.
`--epochs` | `int` | no | How many epochs to train for.

#### Example
```console
$ python nn_classifier.py --dataset cifar10 --epochs 200
```

## Results and Discussion
### Classification reports
#### Logistic Regression
| | precision | recall | f1-score | support |
|---|---|---|---|---|
|macro avg | 0.40 | 0.41 | 0.40 | 10000|
|weighted avg | 0.40 | 0.41 | 0.40 | 10000|

#### Neural Network 
| | precision | recall | f1-score | support |
|---|---|---|---|---|
|macro avg | 0.12 | 0.08 | 0.03 | 10000|
|weighted avg | 0.12 | 0.08 | 0.03 | 10000|

### Discussion
At first glance, there is not much to discuss: logistic regression bleew the neural network out of the water. Simple machine learning is superior, cue the next [AI Winter](https://en.wikipedia.org/wiki/AI_winter). 

However, there are several caveats to this conclusion. The first (and most important one) is that we only trained the neural network for 3 epochs. To reach comparable performance we would have needed to train for perhaps ~200 epochs.

This was a quite deliberate choice on my end. Deep learning is incredibly energy intensive as [this report from MIT shows](https://www.technologyreview.com/2019/06/06/239031/training-a-single-ai-model-can-emit-as-much-carbon-as-five-cars-in-their-lifetimes/). Going against the IPCC with the goal of predicting digits seemed imprudent. The Danish Data Science Community has compiled a great list of carbon impact of data science in [this repository](https://github.com/Dansk-Data-Science-Community/sustainable-data-science).

The reason for this is that it takes a looong time to train a deep learning system. Currently, each epoch took \~70 seconds. Multiplying this by 200 gives approximately 4 hours. The reasons for this are partially that python is a slow language, and that deep learning is works suboptimally on CPUs. There are many ways to speed this up from using tensorflow and GPU-acceleration to using transfer learning.

We know that Deep Learning can perform quite a lot better as the current [SOTA on CIFAR10 is ~99%](https://paperswithcode.com/sota/image-classification-on-cifar-10) and the logistic regression is currently performing at 40%. However, we will explore all of that in the [next assignment](../vision-a3/). 
